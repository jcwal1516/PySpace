"""Explicit TIFF loading for the R-compatible public image interface."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

import matplotlib.colors as mcolors
import numpy as np
import tifffile


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Unsupported image format: {path.suffix or '<none>'}")
    return path


def _pages(path: Path) -> list[np.ndarray]:
    with tifffile.TiffFile(path) as image:
        pages = [np.asarray(page.asarray()) for page in image.pages]
    if not pages:
        raise ValueError(f"TIFF contains no image pages: {path}")
    return pages


def _as_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        if np.any(~np.isfinite(array)) or np.min(array) < 0 or np.max(array) > 1:
            raise ValueError("Floating TIFF samples must be finite values in [0, 1]")
        return np.rint(array * 255).astype(np.uint8)
    if np.issubdtype(array.dtype, np.integer):
        maximum = np.iinfo(array.dtype).max
        return np.rint(array.astype(float) * (255 / maximum)).astype(np.uint8)
    raise TypeError(f"Unsupported TIFF sample dtype: {array.dtype}")


def _hex_volume(rgb_pages: list[np.ndarray]) -> np.ndarray:
    if any(page.ndim != 3 or page.shape[-1] not in {3, 4} for page in rgb_pages):
        raise ValueError("Object TIFF pages must contain RGB or RGBA pixels")
    spatial_shapes = {page.shape[:2] for page in rgb_pages}
    if len(spatial_shapes) != 1:
        raise ValueError("All TIFF pages must have the same spatial dimensions")
    rgb = np.stack([_as_uint8(page)[..., :3] for page in rgb_pages], axis=2)
    flat = rgb.reshape(-1, 3)
    encoded = np.asarray([f"#{red:02X}{green:02X}{blue:02X}" for red, green, blue in flat])
    return encoded.reshape(rgb.shape[:-1])


def _background_hex(color: str) -> str:
    try:
        return mcolors.to_hex(color, keep_alpha=False).upper()
    except ValueError as exc:
        raise ValueError(f"Unknown background color: {color}") from exc


@overload
def load_image(
    in_file: str | Path,
    img_type: Literal["O"],
    bkgd_col: str | None,
    num_chs: int,
    keep_chs: list[int] | None = None,
) -> tuple[np.ndarray, list[str]]: ...


@overload
def load_image(
    in_file: str | Path,
    img_type: Literal["S"],
    bkgd_col: str | None,
    num_chs: int,
    keep_chs: list[int] | None = None,
) -> np.ndarray: ...


def load_image(
    in_file: str | Path,
    img_type: str,
    bkgd_col: str | None,
    num_chs: int,
    keep_chs: list[int] | None = None,
) -> tuple[np.ndarray, list[str]] | np.ndarray:
    """Load an object or scalar TIFF into SPACE's ``X,Y,Z,C`` layout."""
    path = _path(in_file)
    if img_type not in {"O", "S"}:
        raise ValueError("img_type must be 'O' or 'S'")
    if not isinstance(num_chs, int) or num_chs <= 0:
        raise ValueError("num_chs must be a positive integer")
    pages = _pages(path)

    if img_type == "O":
        hex_volume = _hex_volume(pages)
        palette = list(dict.fromkeys(hex_volume.ravel(order="F").tolist()))
        background = _background_hex(bkgd_col) if bkgd_col is not None else None
        if background is not None:
            palette = [background, *[color for color in palette if color != background]]
        color_ids = {color: index + 1 for index, color in enumerate(palette)}
        labels = np.asarray([color_ids[color] for color in hex_volume.ravel()], dtype=np.int64).reshape(
            hex_volume.shape
        )
        if background is not None:
            labels -= 1
            palette = palette[1:]
        return labels[..., None], palette

    if len(pages) % num_chs:
        raise ValueError("The number of TIFF pages must be divisible by num_chs")
    if any(page.ndim != 2 for page in pages):
        raise ValueError("Scalar TIFF pages must be two-dimensional grayscale images")
    spatial_shapes = {page.shape for page in pages}
    if len(spatial_shapes) != 1:
        raise ValueError("All TIFF pages must have the same spatial dimensions")
    normalized_pages = [_as_uint8(page).astype(np.int64) for page in pages]
    z_count = len(pages) // num_chs
    height, width = normalized_pages[0].shape
    result = np.empty((height, width, z_count, num_chs), dtype=np.int64)
    for channel in range(num_chs):
        result[..., channel] = np.stack(normalized_pages[channel::num_chs], axis=2)
    if keep_chs is not None:
        if not keep_chs or any(channel < 1 or channel > num_chs for channel in keep_chs):
            raise ValueError("keep_chs uses one-based channel indices within num_chs")
        result = result[..., [channel - 1 for channel in keep_chs]]
    return result


def read_image_array(path: str | Path) -> np.ndarray:
    """Read a supported image without imposing SPACE object/scalar semantics."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Image file not found: {source}")
    if source.suffix.lower() in {".tif", ".tiff"}:
        return np.asarray(tifffile.imread(source))
    from PIL import Image

    if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image format: {source.suffix or '<none>'}")
    with Image.open(source) as image:
        return np.asarray(image)


__all__ = ["load_image", "read_image_array"]
