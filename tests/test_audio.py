"""Audio decoding, resampling and validation."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from conftest import noise, tone

from wildecho_api.audio import (
    AudioDecodeError,
    AudioEmptyError,
    AudioSilentError,
    AudioTooLongError,
    AudioTooShortError,
    UnsupportedFormatError,
    decode_file,
    probe,
    resample,
    to_mono,
    validate_samples,
)
from wildecho_api.config import TARGET_SAMPLE_RATE, Settings

ffmpeg_available = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg/ffprobe not installed")


# ---------------------------------------------------------------------------
# to_mono
# ---------------------------------------------------------------------------
def test_to_mono_passes_through_mono() -> None:
    mono = tone(1.0)
    assert to_mono(mono) is not None
    np.testing.assert_array_equal(to_mono(mono), mono)


def test_to_mono_averages_interleaved_stereo() -> None:
    left = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    right = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    stereo = np.stack([left, right], axis=1)  # (n, channels)
    np.testing.assert_allclose(to_mono(stereo), [0.0, 0.5, 1.0])


def test_to_mono_handles_channels_first() -> None:
    channels_first = np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]], dtype=np.float32)  # (2, 3)
    np.testing.assert_allclose(to_mono(channels_first), [0.0, 0.5, 1.0])


def test_to_mono_rejects_3d() -> None:
    with pytest.raises(AudioDecodeError):
        to_mono(np.zeros((2, 2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------
def test_resample_is_a_noop_at_target_rate() -> None:
    original = tone(1.0)
    result = resample(original, TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    np.testing.assert_array_equal(result, original)


@pytest.mark.parametrize("source_rate", [8_000, 16_000, 22_050, 44_100, 48_000, 96_000])
def test_resample_produces_expected_length(source_rate: int) -> None:
    original = tone(2.0, frequency=440.0, sample_rate=source_rate)
    result = resample(original, source_rate, TARGET_SAMPLE_RATE)
    expected = round(original.size * TARGET_SAMPLE_RATE / source_rate)
    assert abs(result.size - expected) <= 1
    assert result.dtype == np.float32


@pytest.mark.parametrize("source_rate", [16_000, 44_100, 48_000])
def test_resample_preserves_tone_frequency(source_rate: int) -> None:
    """The whole point of resampling correctly: a 1 kHz tone stays a 1 kHz tone."""
    frequency = 1_000.0
    original = tone(1.0, frequency=frequency, sample_rate=source_rate)
    result = resample(original, source_rate, TARGET_SAMPLE_RATE)

    spectrum = np.abs(np.fft.rfft(result))
    peak_hz = np.fft.rfftfreq(result.size, 1 / TARGET_SAMPLE_RATE)[int(np.argmax(spectrum))]
    assert abs(peak_hz - frequency) < 5.0


def test_resample_preserves_amplitude() -> None:
    original = tone(1.0, frequency=500.0, sample_rate=48_000, amplitude=0.5)
    result = resample(original, 48_000, TARGET_SAMPLE_RATE)
    # Trim the edges, where the Fourier method's periodicity assumption rings.
    interior = result[TARGET_SAMPLE_RATE // 10 : -TARGET_SAMPLE_RATE // 10]
    assert 0.45 < float(np.abs(interior).max()) < 0.55


def test_resample_downsampling_does_not_alias() -> None:
    """A 15 kHz tone sampled at 48 kHz must not fold below Nyquist when moved to 32 kHz."""
    original = tone(1.0, frequency=15_000.0, sample_rate=48_000)
    result = resample(original, 48_000, TARGET_SAMPLE_RATE)
    spectrum = np.abs(np.fft.rfft(result))
    frequencies = np.fft.rfftfreq(result.size, 1 / TARGET_SAMPLE_RATE)
    low_band = spectrum[frequencies < 10_000]
    assert float(low_band.max()) < float(spectrum.max()) * 0.05


def test_resample_rejects_invalid_rate() -> None:
    with pytest.raises(AudioDecodeError):
        resample(tone(1.0), 0, TARGET_SAMPLE_RATE)


def test_resample_rejects_empty() -> None:
    with pytest.raises(AudioEmptyError):
        resample(np.array([], dtype=np.float32), 44_100, TARGET_SAMPLE_RATE)


def test_resample_downmixes_stereo() -> None:
    stereo = np.stack([tone(1.0, sample_rate=44_100), tone(1.0, sample_rate=44_100)], axis=1)
    result = resample(stereo, 44_100, TARGET_SAMPLE_RATE)
    assert result.ndim == 1


# ---------------------------------------------------------------------------
# validate_samples
# ---------------------------------------------------------------------------
def test_validate_rejects_empty(settings: Settings) -> None:
    with pytest.raises(AudioEmptyError):
        validate_samples(np.array([], dtype=np.float32), TARGET_SAMPLE_RATE, settings)


def test_validate_rejects_too_short(settings: Settings) -> None:
    with pytest.raises(AudioTooShortError) as excinfo:
        validate_samples(tone(0.2), TARGET_SAMPLE_RATE, settings)
    assert "minimum" in str(excinfo.value)


def test_validate_rejects_too_long(settings: Settings) -> None:
    over = int((settings.max_duration_seconds + 1) * TARGET_SAMPLE_RATE)
    samples = np.full(over, 0.3, dtype=np.float32)
    with pytest.raises(AudioTooLongError):
        validate_samples(samples, TARGET_SAMPLE_RATE, settings)


def test_validate_rejects_digital_silence(settings: Settings) -> None:
    with pytest.raises(AudioSilentError):
        validate_samples(
            np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32), TARGET_SAMPLE_RATE, settings
        )


def test_validate_rejects_near_silence(settings: Settings) -> None:
    whisper = (noise(2.0) * 1e-6).astype(np.float32)
    with pytest.raises(AudioSilentError):
        validate_samples(whisper, TARGET_SAMPLE_RATE, settings)


def test_validate_rejects_all_nan(settings: Settings) -> None:
    with pytest.raises(AudioDecodeError):
        validate_samples(
            np.full(TARGET_SAMPLE_RATE, np.nan, dtype=np.float32), TARGET_SAMPLE_RATE, settings
        )


def test_validate_zeroes_partial_nan_and_passes(settings: Settings) -> None:
    samples = tone(2.0).copy()
    samples[100:200] = np.nan
    validate_samples(samples, TARGET_SAMPLE_RATE, settings)
    assert bool(np.isfinite(samples).all())


def test_validate_accepts_normal_audio(settings: Settings) -> None:
    validate_samples(tone(3.0), TARGET_SAMPLE_RATE, settings)


# ---------------------------------------------------------------------------
# decode_file (requires ffmpeg)
# ---------------------------------------------------------------------------
@requires_ffmpeg
def test_decode_wav_roundtrip(tmp_path, wav_bytes, settings: Settings) -> None:
    path = tmp_path / "clip.wav"
    path.write_bytes(wav_bytes(tone(3.0, sample_rate=44_100), sample_rate=44_100))

    decoded = decode_file(path, settings)
    assert decoded.sample_rate == TARGET_SAMPLE_RATE
    assert decoded.source.sample_rate == 44_100
    assert decoded.source.channels == 1
    assert 2.9 < decoded.duration_seconds < 3.1
    assert decoded.samples.dtype == np.float32
    assert decoded.samples.flags.writeable, "downstream code normalises in place"


@requires_ffmpeg
def test_decode_mp3_fixture(settings: Settings) -> None:
    from conftest import NIGHTJAR_CLIP

    decoded = decode_file(NIGHTJAR_CLIP, settings)
    assert decoded.sample_rate == TARGET_SAMPLE_RATE
    assert 9.5 < decoded.duration_seconds < 10.5
    assert float(np.abs(decoded.samples).max()) > 0.01


@requires_ffmpeg
def test_decode_rejects_non_audio(tmp_path, settings: Settings) -> None:
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"this is definitely not a wav file" * 100)
    with pytest.raises((UnsupportedFormatError, AudioDecodeError)):
        decode_file(path, settings)


@requires_ffmpeg
def test_decode_rejects_silent_wav(tmp_path, wav_bytes, settings: Settings) -> None:
    path = tmp_path / "silence.wav"
    path.write_bytes(wav_bytes(np.zeros(TARGET_SAMPLE_RATE * 2, dtype=np.float32)))
    with pytest.raises(AudioSilentError):
        decode_file(path, settings)


@requires_ffmpeg
def test_decode_rejects_too_short_wav(tmp_path, wav_bytes, settings: Settings) -> None:
    path = tmp_path / "blip.wav"
    path.write_bytes(wav_bytes(tone(0.1)))
    with pytest.raises(AudioTooShortError):
        decode_file(path, settings)


@requires_ffmpeg
def test_decode_rejects_over_long_clip(tmp_path, wav_bytes, settings: Settings) -> None:
    short_settings = settings.model_copy(update={"max_duration_seconds": 2.0})
    path = tmp_path / "long.wav"
    path.write_bytes(wav_bytes(tone(5.0)))
    with pytest.raises(AudioTooLongError):
        decode_file(path, short_settings)


@requires_ffmpeg
def test_decode_downmixes_stereo(tmp_path, settings: Settings) -> None:
    import io
    import wave

    left = tone(2.0, frequency=440.0)
    right = tone(2.0, frequency=880.0)
    interleaved = np.empty(left.size * 2, dtype="<i2")
    interleaved[0::2] = (left * 32767).astype("<i2")
    interleaved[1::2] = (right * 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(interleaved.tobytes())

    path = tmp_path / "stereo.wav"
    path.write_bytes(buffer.getvalue())

    decoded = decode_file(path, settings)
    assert decoded.source.channels == 2
    assert decoded.samples.ndim == 1


@requires_ffmpeg
def test_probe_reports_source_properties(tmp_path, wav_bytes, settings: Settings) -> None:
    path = tmp_path / "probe.wav"
    path.write_bytes(wav_bytes(tone(1.5, sample_rate=22_050), sample_rate=22_050))
    info = probe(path, settings)
    assert info.sample_rate == 22_050
    assert info.channels == 1
    assert 1.4 < info.duration_seconds < 1.6
