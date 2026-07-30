"""Audio decoding and resampling.

Container parsing is delegated entirely to ``ffmpeg`` in a subprocess. That keeps
every untrusted byte out of the API process, which is the single most valuable
security property in this codebase, and it means wav, mp3, m4a, webm, ogg and flac
all work without per-format Python libraries.

The pure-numpy resampler exists for the library path, where a caller hands
``identify()`` a bare array at some other sample rate. Requests coming through the
HTTP API are resampled by ffmpeg during decode and skip it.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TARGET_SAMPLE_RATE, Settings

logger = logging.getLogger(__name__)

#: Advertised in /v1/about and used to reject obviously wrong uploads early. ffmpeg
#: handles more than this; these are the formats covered by tests.
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/vnd.wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/aac",
        "audio/webm",
        "audio/ogg",
        "audio/opus",
        "audio/flac",
        "audio/x-flac",
        "video/mp4",  # phones often record .m4a with an mp4 container type
        "video/webm",
        "application/octet-stream",  # many mobile HTTP clients send this
    }
)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".mp4", ".webm", ".ogg", ".oga", ".opus", ".flac"}
)


class AudioError(Exception):
    """Base class for audio problems that are the caller's fault, not the server's.

    Every subclass carries a stable ``code`` so the API can map it to a 4xx body
    without string matching.
    """

    code = "audio_error"
    http_status = 422

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class UnsupportedFormatError(AudioError):
    code = "unsupported_format"
    http_status = 415


class AudioDecodeError(AudioError):
    code = "audio_decode_failed"


class AudioTooShortError(AudioError):
    code = "audio_too_short"


class AudioTooLongError(AudioError):
    code = "audio_too_long"
    http_status = 413


class AudioSilentError(AudioError):
    code = "audio_silent"


class AudioEmptyError(AudioError):
    code = "audio_empty"


class FfmpegUnavailableError(RuntimeError):
    """ffmpeg or ffprobe is not installed. A deployment problem, not a request problem."""


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """What the uploaded file looked like before we touched it."""

    sample_rate: int
    channels: int
    duration_seconds: float
    codec: str


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Mono float32 PCM at :data:`TARGET_SAMPLE_RATE`, plus the source's properties."""

    samples: np.ndarray
    sample_rate: int
    source: SourceInfo

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def ensure_ffmpeg(settings: Settings) -> None:
    """Raise if the required binaries are missing, with an actionable message."""
    for binary in (settings.ffmpeg_path, settings.ffprobe_path):
        if shutil.which(binary) is None:
            raise FfmpegUnavailableError(
                f"{binary!r} was not found on PATH. Install ffmpeg "
                "(`brew install ffmpeg` or `apt-get install ffmpeg`) or set "
                "WILDECHO_FFMPEG_PATH / WILDECHO_FFPROBE_PATH."
            )


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess with no shell, capturing output and enforcing a timeout."""
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(
            f"Audio processing timed out after {timeout}s. The file may be corrupt or too complex."
        ) from exc
    except OSError as exc:
        raise FfmpegUnavailableError(f"Could not execute {argv[0]!r}: {exc}") from exc


def probe(path: Path, settings: Settings) -> SourceInfo:
    """Read stream properties without decoding the whole file.

    Called before decoding so an over-long clip is rejected before we allocate
    hundreds of megabytes of PCM for it.
    """
    result = _run(
        [
            settings.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        settings.ffmpeg_timeout_seconds,
    )
    if result.returncode != 0:
        raise UnsupportedFormatError(
            "Could not read this file as audio. Supported formats are wav, mp3, m4a, "
            "webm, ogg and flac."
        )
    try:
        payload = json.loads(result.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise AudioDecodeError("Could not interpret the audio file's metadata.") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise UnsupportedFormatError("The uploaded file contains no audio stream.")
    stream = streams[0]

    # Duration can live on the stream, on the format, or nowhere at all (some webm
    # recordings from browsers omit it). A missing duration is not fatal: we fall
    # back to measuring the decoded sample count.
    raw_duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
    try:
        duration = float(raw_duration)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration < 0:
        duration = 0.0

    try:
        sample_rate = int(stream.get("sample_rate") or 0)
        channels = int(stream.get("channels") or 0)
    except (TypeError, ValueError):
        sample_rate, channels = 0, 0

    return SourceInfo(
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=duration,
        codec=str(stream.get("codec_name") or "unknown"),
    )


def decode_file(path: Path, settings: Settings) -> DecodedAudio:
    """Decode any ffmpeg-readable file to mono float32 at 32 kHz.

    ffmpeg does downmixing and resampling with its own high-quality resampler, so
    the HTTP path never needs :func:`resample`.
    """
    ensure_ffmpeg(settings)
    source = probe(path, settings)

    if source.duration_seconds > settings.max_duration_seconds:
        raise AudioTooLongError(
            f"Clip is {source.duration_seconds:.1f}s; the maximum is "
            f"{settings.max_duration_seconds:.0f}s. Trim it and try again."
        )

    result = _run(
        [
            settings.ffmpeg_path,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-f",
            "f32le",
            # Cap the decoded stream too. probe() can be fooled by a bad header;
            # this is the backstop that bounds memory for real.
            "-t",
            str(settings.max_duration_seconds),
            "-",
        ],
        settings.ffmpeg_timeout_seconds,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        reason = stderr[-1] if stderr else "unknown error"
        logger.warning("ffmpeg decode failed: %s", reason)
        raise AudioDecodeError(
            "Could not decode this audio file. It may be truncated or corrupt. "
            "Supported formats are wav, mp3, m4a, webm, ogg and flac."
        )

    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise AudioEmptyError("The decoded audio contains no samples.")

    # np.frombuffer is a read-only view over the subprocess buffer; downstream code
    # normalises in place, so take ownership here.
    samples = np.array(samples, dtype=np.float32, copy=True)
    validate_samples(samples, TARGET_SAMPLE_RATE, settings)

    if source.sample_rate == 0 or source.channels == 0:
        source = SourceInfo(
            sample_rate=source.sample_rate or TARGET_SAMPLE_RATE,
            channels=source.channels or 1,
            duration_seconds=source.duration_seconds or len(samples) / TARGET_SAMPLE_RATE,
            codec=source.codec,
        )

    return DecodedAudio(samples=samples, sample_rate=TARGET_SAMPLE_RATE, source=source)


def validate_samples(samples: np.ndarray, sample_rate: int, settings: Settings) -> None:
    """Reject clips that are empty, too short, too long, silent, or not finite."""
    if samples.size == 0:
        raise AudioEmptyError("The audio contains no samples.")

    duration = samples.size / sample_rate
    if duration < settings.min_duration_seconds:
        raise AudioTooShortError(
            f"Clip is {duration:.2f}s; the minimum is {settings.min_duration_seconds:.2f}s. "
            "Record a longer sample."
        )
    if duration > settings.max_duration_seconds:
        raise AudioTooLongError(
            f"Clip is {duration:.1f}s; the maximum is {settings.max_duration_seconds:.0f}s."
        )

    finite = np.isfinite(samples)
    if not bool(finite.all()):
        if not bool(finite.any()):
            raise AudioDecodeError("The audio contains no usable samples (all NaN or infinite).")
        logger.warning("audio contained non-finite samples; they were zeroed")
        samples[~finite] = 0.0

    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms < settings.silence_rms_threshold:
        raise AudioSilentError(
            f"The audio is silent (RMS {rms:.2e}). Check the microphone permission and "
            "recording level, then try again."
        )


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average any channel layout down to mono float32.

    Accepts ``(n,)``, ``(n, channels)`` and ``(channels, n)``. For the ambiguous
    square case the second axis is treated as channels, matching the interleaved
    layout every decoder produces.
    """
    array = np.asarray(samples)
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    if array.ndim != 2:
        raise AudioDecodeError(f"Expected 1-D or 2-D audio, got shape {array.shape}.")
    if array.shape[0] < array.shape[1]:
        array = array.T  # (channels, n) -> (n, channels)
    return array.mean(axis=1).astype(np.float32, copy=False)


def resample(
    samples: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Resample mono audio using a band-limited Fourier method.

    This is the same approach as ``scipy.signal.resample``: take the real FFT,
    truncate or zero-pad the spectrum to the new length, and transform back. It is
    O(n log n), needs no dependency beyond numpy, and is band-limited so
    downsampling does not alias. Because it assumes the signal is periodic it can
    ring slightly at the very start and end of a clip; that is inaudible to a
    classifier looking at 5-second windows.

    ffmpeg already resamples everything arriving over HTTP, so this runs only for
    direct library calls.
    """
    if source_rate <= 0:
        raise AudioDecodeError(f"Invalid source sample rate: {source_rate}.")
    mono = to_mono(samples)
    if source_rate == target_rate:
        return np.ascontiguousarray(mono, dtype=np.float32)
    if mono.size == 0:
        raise AudioEmptyError("Cannot resample empty audio.")

    target_length = round(mono.size * target_rate / source_rate)
    if target_length < 1:
        raise AudioTooShortError(
            f"Resampling {mono.size} samples from {source_rate} Hz to {target_rate} Hz "
            "leaves nothing to analyse."
        )

    spectrum = np.fft.rfft(mono.astype(np.float64))
    keep = min(spectrum.size, target_length // 2 + 1)
    resized = np.zeros(target_length // 2 + 1, dtype=complex)
    resized[:keep] = spectrum[:keep]

    # When the new length is even, the Nyquist bin is shared between the positive
    # and negative frequency it represents; halve it so amplitude is preserved.
    if target_length % 2 == 0 and keep == resized.size and spectrum.size > keep:
        resized[-1] *= 0.5

    out = np.fft.irfft(resized, n=target_length) * (target_length / mono.size)
    return np.ascontiguousarray(out, dtype=np.float32)
