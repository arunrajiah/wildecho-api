# Security Policy

## Supported versions

This project is pre-1.0. Only the latest release on the `main` branch receives
security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues.**

Report privately through either channel:

1. **GitHub Security Advisories** (preferred):
   [Report a vulnerability](https://github.com/arunrajiah/wildecho-api/security/advisories/new)
2. **Email**: arunrajiah@gmail.com with `[wildecho-api security]` in the subject.

Please include:

* A description of the issue and why you believe it is a security problem.
* Steps to reproduce, ideally a minimal request or audio file.
* The version or commit you tested.
* Any suggested fix, if you have one.

### What to expect

* Acknowledgement within 5 business days.
* An assessment and rough remediation timeline within 14 days.
* Credit in the release notes and advisory, unless you prefer to stay anonymous.

This is a volunteer-maintained project with no paid security team, so please be
patient. There is no bug bounty.

## Threat model for self-hosters

Read this before exposing an instance to the internet. It matters more than any
individual CVE in the dependency tree.

**This service ships no authentication.** That is deliberate and documented as
out of scope. Anything you deploy publicly is an open, unauthenticated
compute-heavy endpoint.

Known risk areas, and what the project does about them:

| Risk | Mitigation in this project | What you must still do |
| --- | --- | --- |
| CPU exhaustion from long clips | Uploads capped by `WILDECHO_MAX_UPLOAD_BYTES` (default 25 MB) and duration capped by `WILDECHO_MAX_DURATION_SECONDS` (default 300s) | Lower both if you run on small hardware |
| Request flooding | Per-IP rate limit, `WILDECHO_RATE_LIMIT` (default `20/hour`) | Tune it, and put a real reverse proxy in front |
| Malicious media files | Decoding is delegated to `ffmpeg` in a subprocess with a wall-clock timeout, no shell interpolation, and output size caps. The API process never parses container formats itself | Keep `ffmpeg` patched. It is the largest attack surface here |
| Untrusted `X-Forwarded-For` | Rate limiting uses the socket peer address by default | Only enable proxy header trust if you actually run behind a proxy you control |
| Open CORS | Defaults to `*` so mobile apps work out of the box | Set `WILDECHO_CORS_ORIGINS` to your app's origins in production |
| Path traversal via filenames | Uploaded filenames are never used for filesystem paths. Audio is streamed to a temp file with a generated name | Nothing |

If you need authentication, terminate it at a reverse proxy or API gateway in
front of this service.

## Out of scope

The following are not treated as vulnerabilities in this repository:

* **Misidentified species.** That is a model accuracy limitation. See the
  "Accuracy and limitations" section of the README.
* **Vulnerabilities in Perch 2.0 itself or in its weights.** Report those to
  [google-research/perch](https://github.com/google-research/perch).
* **Denial of service against an instance you deployed without a rate limit or
  reverse proxy.** Configure it as documented above.
* **Missing authentication.** Documented as out of scope, not a defect.
