# Quick start

The table API accepts a DataFrame directly; it does not write an intermediate
file or require a `Radius` column before census collection.

```python
import pandas as pd
from pyspace import census_table

cells = pd.DataFrame(
    {
        "X": [0.0, 1.0, 2.0, 3.0],
        "Y": [0.0, 0.0, 0.0, 0.0],
        "Z": [0.0, 0.0, 0.0, 0.0],
        "Object": [1, 1, 2, 2],
    }
)

result = census_table(
    cells,
    radii=[1.5],
    sample_size=[2],
    seed_points=[1, 2],
    sample_indices=[0, 2],
)
print(result.census)
```

Use `random_state` for repeatable Python runs. For cross-language stochastic
parity, pass the function-specific explicit random plan described in
[reproducibility](reproducibility.md).
