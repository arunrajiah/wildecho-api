---
name: Feature request
about: Suggest a capability or improvement
title: "feat: "
labels: enhancement
assignees: ''
---

## The problem

<!--
What are you trying to do that this service makes hard or impossible? Describe
the use case, not just the proposed solution.
-->

## Proposed solution

<!-- What you would like the service to do. -->

## Alternatives you considered

<!-- Including "I could do this in my own client instead" if that applies. -->

## Scope check

The following are deliberately out of scope for this project. If your request
involves one of them, please explain why it belongs here rather than in a layer
in front of the service:

- [ ] My request is **not** about authentication or API keys (terminate those at
      a reverse proxy)
- [ ] My request is **not** about usage analytics or telemetry
- [ ] My request is **not** about deployment to a specific host or platform
- [ ] My request does **not** require committing model weights to this repo

## Would this change the model or just the wrapper?

<!--
Retraining, fine-tuning, or extending species coverage is upstream work at
https://github.com/google-research/perch, not something this wrapper can do.
Adding bat coverage, for instance, needs a different model.
-->

- [ ] Wrapper only (API surface, preprocessing, packaging, docs)
- [ ] Needs a different or additional model

## Additional context

<!-- Links, papers, similar projects, screenshots. -->
