# Examples

All examples use synthetic data, expose a `main()` function, and run in the test
suite.

- `examples/table_workflow.py` performs a deterministic coordinate-table census.
- `examples/image_workflow.py` measures a synthetic object image and saves a plot.
- `examples/safe_bundle.py` writes and reloads the safe result-bundle format.

Run one from the repository root:

```bash
python examples/table_workflow.py
```

The default output directory is `example-output/`, which is local working data
and must not be committed.
