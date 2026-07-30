## Summary

<!-- What this PR changes and why. One or two sentences is fine. -->

## Related issue

<!-- e.g. Closes #12. Write "none" for small self-evident fixes. -->

## Type of change

- [ ] `fix` bug fix (no API change)
- [ ] `feat` new capability
- [ ] `docs` documentation only
- [ ] `test` tests only
- [ ] `refactor` no behaviour change
- [ ] `perf` performance
- [ ] `chore` / `ci` tooling, deps, workflows
- [ ] Breaking change

## Checks

Run these locally before requesting review. They are exactly what CI runs.

```bash
ruff check . && ruff format --check . && mypy src scripts && pytest
```

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `mypy src scripts` passes
- [ ] `pytest` passes
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

## Project rules

- [ ] No model weight files added (`.onnx`, `.tflite`, `.pb`, `.safetensors`)
- [ ] No new dependency added, or if one was, it is justified in the summary above
- [ ] Any bundled audio has a stated permissive license and is credited in `NOTICE`
- [ ] I did not weaken the accuracy or limitations wording in the README or
      `/v1/about`
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` if this is user-facing

## If this touches inference

- [ ] I verified predictions against the bundled test clip (`pytest -m model`)
- [ ] I noted any change in output confidences in the summary above

## Notes for the reviewer

<!-- Anything you want a second opinion on, or tradeoffs you made. -->
