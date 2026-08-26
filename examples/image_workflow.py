"""Run a deterministic object-image census on a synthetic NumPy array."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pyspace import census_image, plot_image


def main(output_dir: str | Path = "example-output/image") -> dict[str, Any]:
    """Measure one explicit neighborhood and save its encoded image view."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    image = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 2, 2],
            [0, 2, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    result = census_image(
        {"O1": image},
        radii=[1.5],
        sample_size=[1],
        seed_points=np.array([[1, 1]]),
        random_state=11,
    )
    encoded, figure = plot_image(image[:, :, None, None], "O", ["#4C78A8", "#F58518"])
    figure.savefig(destination / "objects.png", dpi=100)
    return {
        "rows": 0 if result.census is None else len(result.census),
        "shape": encoded.shape,
        "figure": destination / "objects.png",
    }


if __name__ == "__main__":
    print(main())
