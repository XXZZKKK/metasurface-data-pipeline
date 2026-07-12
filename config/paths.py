from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


def default_output_dir(output_name: str) -> Path:
    name = str(output_name).strip()
    if not name:
        raise ValueError("output_name must not be empty")
    if Path(name).name != name:
        raise ValueError("output_name must be a single directory name")
    return DEFAULT_OUTPUT_ROOT / name

