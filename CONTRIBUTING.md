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

Do not commit clinical data, raw images, annotations, manuscripts, generated study
results, credentials, caches, or compiled files.
