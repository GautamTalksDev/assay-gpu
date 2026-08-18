# Contributing

Contributions are welcome after you have read the Kill Tests in `README.md`.
If a kill test has already failed, do not add features; stop.

## Requirements

- Python 3.11+
- `uv`
- Deterministic, unit-testable changes only
- No new dependency that is not already in `pyproject.toml` without prior
  agreement from a maintainer
- Prefer the standard library and `torch`; do not add LLM calls, heuristic
  scores, machine-learning models, or network requests the user did not
  explicitly request
- Every numerical threshold must be traceable to measured data in this
  repository

## Developer Certificate of Origin (DCO)

Every commit **must** be signed off with `git commit -s`. Unsigned commits
will be rejected.

Signing off certifies that you agree to the Developer Certificate of Origin
version 1.1:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

The sign-off is a line in the commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

Git adds this when you use `-s`:

```bash
git commit -s -m "Your message"
```

Configure `user.name` and `user.email` so the sign-off matches your identity.

## Workflow

1. Fork and branch from the default branch.
2. Keep changes small and testable.
3. Run `uv sync`, `uv run ruff check .`, `uv run mypy --strict src/`, and
   `uv run pytest -m cpu`. Do not expect GPU-marked tests to run in CI.
4. Commit with `git commit -s`.
5. Open a pull request that explains *why*, not only *what*.
