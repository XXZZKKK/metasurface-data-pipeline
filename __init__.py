"""Model-agnostic metasurface data extraction pipeline."""

from __future__ import annotations

import importlib
import sys


_LEGACY_MODULES = (
    "aprilgrid",
    "base_dataset",
    "basis",
    "board_constrained_dataset",
    "board_prior",
    "dense_observations",
    "geometry",
    "paths",
    "real_metasurface",
    "response",
    "roi",
    "sparse_observations",
    "spectral_pixel_observations",
)


for _module_name in _LEGACY_MODULES:
    _module = importlib.import_module(f".compat.{_module_name}", __name__)
    sys.modules[f"{__name__}.{_module_name}"] = _module
    globals()[_module_name] = _module


__all__ = list(_LEGACY_MODULES)
