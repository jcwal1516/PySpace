"""R-compatible whole-image diversity measurements."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_image_type(img_type: str) -> None:
    if img_type not in {"O", "S"}:
        raise ValueError("img_type must be 'O' or 'S'")


def _unwrap_single_image(image: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    if isinstance(image, (list, tuple)):
        if len(image) != 1:
            raise ValueError("Alpha diversity can be measured for only one image at a time")
        image = image[0]
    return np.asarray(image)


def _unwrap_single_palette(palette: Sequence[str] | Sequence[Sequence[str]]) -> list[str]:
    if len(palette) == 1 and isinstance(palette[0], (list, tuple, np.ndarray)):
        return [str(color) for color in palette[0]]
    if any(isinstance(color, (list, tuple, np.ndarray)) for color in palette):
        raise ValueError("Alpha diversity requires one color palette")
    return [str(color) for color in palette]


def alpha_diversity(
    img: np.ndarray | Sequence[np.ndarray],
    img_type: str,
    col_pal: Sequence[str] | Sequence[Sequence[str]],
    plot_bkgd: str = "W",
) -> tuple[pd.DataFrame, float]:
    """Measure whole-image composition and Shannon entropy as SPACE does.

    ``plot_bkgd`` is accepted for API parity. Plotting is deliberately separate
    in Python so this numerical function has no display or global-style side
    effects.
    """
    _validate_image_type(img_type)
    if plot_bkgd not in {"W", "B"}:
        raise ValueError("plot_bkgd must be 'W' or 'B'")

    image = _unwrap_single_image(img)
    palette = _unwrap_single_palette(col_pal)
    if not palette:
        raise ValueError("col_pal must contain at least one color")

    if img_type == "O":
        counts = np.asarray([np.count_nonzero(image == index) for index in range(1, len(palette) + 1)], dtype=float)
    else:
        if image.ndim < 2 or image.shape[-1] != len(palette):
            raise ValueError("A scalar image must have one final-axis channel per palette color")
        counts = np.sum(image, axis=tuple(range(image.ndim - 1)), dtype=float)

    total = float(counts.sum())
    proportions = counts / total if total else np.full(counts.shape, np.nan)
    positive = proportions > 0
    entropy = float(-np.sum(proportions[positive] * np.log2(proportions[positive])))
    composition = pd.DataFrame({"V": np.arange(1, len(palette) + 1), "P": 100.0 * proportions})
    return composition, entropy


def beta_diversity(
    img: Sequence[np.ndarray],
    img_type: str,
    col_pal: Sequence[Sequence[str]],
    plot_bkgd: str = "W",
) -> tuple[pd.DataFrame, float]:
    """Measure SPACE's weighted regional KL divergence and compositions."""
    _validate_image_type(img_type)
    if plot_bkgd not in {"W", "B"}:
        raise ValueError("plot_bkgd must be 'W' or 'B'")
    if len(img) != 2:
        raise ValueError("Exactly two images must be provided: parent objects and constituent variables")
    if len(col_pal) != 2:
        raise ValueError("Exactly two color palettes must be provided")

    parent_image = np.asarray(img[0])
    variable_image = np.asarray(img[1])
    parent_palette = list(col_pal[0])
    variable_palette = list(col_pal[1])
    n_parents = len(parent_palette)
    n_variables = len(variable_palette) if img_type == "O" else variable_image.shape[-1]
    if not n_parents or not n_variables:
        raise ValueError("Both palettes must contain at least one color")

    expected_shape = variable_image.shape if img_type == "O" else variable_image.shape[:-1]
    if parent_image.shape != expected_shape:
        raise ValueError("Parent and constituent images must have matching spatial dimensions")

    counts = np.zeros((n_parents, n_variables), dtype=float)
    parent_sizes = np.zeros(n_parents, dtype=float)
    for parent_offset in range(n_parents):
        parent_id = parent_offset + 1
        parent_mask = parent_image == parent_id
        parent_sizes[parent_offset] = np.count_nonzero(parent_mask)
        if not parent_sizes[parent_offset]:
            continue
        if img_type == "O":
            counts[parent_offset] = [
                np.count_nonzero(variable_image[parent_mask] == variable_id)
                for variable_id in range(1, n_variables + 1)
            ]
        else:
            counts[parent_offset] = np.sum(variable_image[parent_mask], axis=0, dtype=float)

    row_totals = counts.sum(axis=1)
    positive_rows = row_totals > 0
    proportions = np.zeros_like(counts)
    proportions[positive_rows] = counts[positive_rows] / row_totals[positive_rows, None]
    entropy = np.zeros(n_parents, dtype=float)
    for row_index, row in enumerate(proportions):
        positive = row > 0
        entropy[row_index] = -np.sum(row[positive] * np.log2(row[positive]))

    if not np.any(positive_rows):
        raise ValueError("No parent object contains a measured constituent variable")
    region_weights = parent_sizes[positive_rows]
    region_weights /= region_weights.sum()
    beta_rows = proportions[positive_rows]
    average = np.sum(beta_rows * region_weights[:, None], axis=0)
    average /= average.sum()
    tiny_probability = 0.000001 / n_variables
    if np.any(average == 0):
        average[average == 0] = tiny_probability
        average /= average.sum()

    divergences = np.empty(len(beta_rows), dtype=float)
    for row_index, region in enumerate(beta_rows):
        smoothed = region.copy()
        if np.any(smoothed == 0):
            smoothed[smoothed == 0] = tiny_probability
            smoothed /= smoothed.sum()
        divergences[row_index] = np.sum(smoothed * np.log2(smoothed / average))
    beta = float(np.sum(divergences * region_weights))

    columns: dict[str, np.ndarray] = {"O": np.arange(1, n_parents + 1)}
    columns.update({f"V{index + 1}": 100.0 * proportions[:, index] for index in range(n_variables)})
    columns["E"] = entropy
    return pd.DataFrame(columns), beta


__all__ = ["alpha_diversity", "beta_diversity"]
