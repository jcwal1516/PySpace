# Contributing

Changes must preserve the scientific behavior of the pinned R SPACE reference.
For behavior changes, first add a failing test, implement the smallest correction,
and run the relevant parity and regression suites.

Before submitting a change, run:

```bash
make check
make parity
make build
make audit
```

When changing release files, including GitHub workflows, regenerate
`ARTIFACT_MANIFEST.json` with `python scripts/write_artifact_manifest.py`
and include it in the same change. Dependabot updates need this regeneration
before they can pass the artifact audit. Keep the CodeQL `init` and `analyze`
actions pinned to the same commit; Dependabot groups their updates together.

Do not commit clinical data, raw images, annotations, manuscripts, generated study
results, credentials, caches, or compiled files.
