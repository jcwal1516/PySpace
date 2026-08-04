# Installation

PySpace 0.1.0 supports CPython 3.11 through 3.14 on Linux, macOS, and Windows.

```bash
python -m pip install pyspace-analysis
```

This installs NumPy/pandas/SciPy, image and table readers, static and Plotly
visualization, NetworkX, progress reporting, and opt-in resource inspection.

Community detection with igraph, Leiden, or Infomap is a separate extra because
those packages use copyleft licenses:

```bash
python -m pip install 'pyspace-analysis[community]'
```

For a source checkout:

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,docs]'
make check
```

R is required only for `make parity`'s live-oracle job and R/Python benchmarks.
