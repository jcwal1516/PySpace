"""Scoped publication styling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib as mpl


@contextmanager
def publication_style(*, font_size: float = 11, dpi: int = 150) -> Iterator[None]:
    """Temporarily apply a compact publication style and restore all settings."""
    with mpl.rc_context():
        mpl.rcParams["figure.dpi"] = dpi
        mpl.rcParams["font.size"] = font_size
        mpl.rcParams["axes.labelsize"] = font_size
        mpl.rcParams["axes.titlesize"] = font_size + 1
        mpl.rcParams["legend.fontsize"] = max(8, font_size - 1)
        mpl.rcParams["savefig.bbox"] = "tight"
        yield


__all__ = ["publication_style"]
