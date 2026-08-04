# PySpace

PySpace is a Python implementation of [SPACE](https://github.com/eschrom/SPACE),
Spatial Patterning Analysis of Cellular Ensembles. It provides spatial census,
cisMI/transMI, pattern learning, diversity analysis, visualization, and
Python-native workflow tools for multiplex tissue imaging.

Scientific compatibility is pinned to upstream SPACE commit
`94f0a0f9311e2ee2b406ebc45c84a4e93f2f38f8`. Numerical and tabular parity is
tested against live R-generated oracles; Python-only interfaces delegate to the
same computational core.

## Installation

```bash
python -m pip install pyspace-analysis
```

Optional GPL-licensed community-detection backends are separate:

```bash
python -m pip install 'pyspace-analysis[community]'
```

## Quick start

```python
import pandas as pd
from pyspace import SpacePipeline

cells = pd.DataFrame(
    {
        "X": [0.0, 1.0, 2.0],
        "Y": [0.0, 0.0, 0.0],
        "Object": [1, 1, 2],
    }
)

pipeline = SpacePipeline().load_table(cells, validate=True)
result = pipeline.census(radii=[1.0], n_neighborhoods=1)
```

The project is under active pre-1.0 development. See the versioned documentation
for the R-to-Python API map, reproducibility contract, and complete examples.

## Data policy

This repository contains source code, synthetic fixtures, and provenance-reviewed
derived tutorial outputs only. Clinical spreadsheets, raw images, annotations,
manuscripts, and generated study results are prohibited.

## License and attribution

Apache-2.0. See `LICENSE` and `NOTICE`. PySpace is not the canonical R SPACE
package; cite the original SPACE project and the PySpace release used in an
analysis.
