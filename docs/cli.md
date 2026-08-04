# Command-line interface

The installed command is `pyspace`. Errors use exit code 2 and write an
actionable message to stderr; successful commands return 0.

## `census`

Collect neighborhoods from CSV, TSV, XLSX, parquet, TIFF, PNG, or JPEG input.
Unknown extensions are rejected instead of being guessed.

```bash
pyspace census cells.csv --radii 10,20 --sample-size 100,100 --variables marker --output census.pyspace
```

## `analyze`

Run pattern learning or cisMI. Strict cisMI requires a patch list unless
`--allow-permutation-fallback` is explicitly supplied.

## `plot`

Render a saved safe bundle to PNG, PDF, or SVG.

## `convert`

Convert explicit safe table, JSON, NPZ, and bundle formats. Pickle input and
output are always rejected.

```bash
pyspace convert table.csv --format bundle --output table.pyspace
```
