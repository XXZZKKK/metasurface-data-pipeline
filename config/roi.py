"""Utilities for selecting and combining multiple metasurface ROIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROI_SIZE = 160


@dataclass(frozen=True)
class MetasurfaceRoi:
    name: str
    x0: int
    y0: int
    width: int = ROI_SIZE
    height: int = ROI_SIZE

    @property
    def x1(self) -> int:
        return int(self.x0 + self.width)

    @property
    def y1(self) -> int:
        return int(self.y0 + self.height)

    def to_dict(self) -> dict:
        return {
            "name": str(self.name),
            "x0": int(self.x0),
            "y0": int(self.y0),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetasurfaceRoi":
        return cls(
            name=str(data["name"]),
            x0=int(data["x0"]),
            y0=int(data["y0"]),
            width=int(data.get("width", ROI_SIZE)),
            height=int(data.get("height", ROI_SIZE)),
        )


def fixed_roi_from_center(
    name: str,
    center_x: float,
    center_y: float,
    image_width: int,
    image_height: int,
    size: int = ROI_SIZE,
) -> MetasurfaceRoi:
    size = int(size)
    if size <= 0:
        raise ValueError("ROI size must be positive")
    if size > int(image_width) or size > int(image_height):
        raise ValueError("ROI size must fit inside the image")
    x0 = int(round(float(center_x) - size * 0.5))
    y0 = int(round(float(center_y) - size * 0.5))
    x0 = max(0, min(int(image_width) - size, x0))
    y0 = max(0, min(int(image_height) - size, y0))
    return MetasurfaceRoi(str(name), x0, y0, size, size)


def validate_rois(
    rois: Iterable[MetasurfaceRoi],
    image_width: int | None = None,
    image_height: int | None = None,
    expected_size: int = ROI_SIZE,
) -> list[MetasurfaceRoi]:
    checked = [roi if isinstance(roi, MetasurfaceRoi) else MetasurfaceRoi.from_dict(roi) for roi in rois]
    if not checked:
        raise ValueError("at least one ROI is required")
    names = [roi.name for roi in checked]
    if len(names) != len(set(names)):
        raise ValueError("ROI names must be unique")
    for roi in checked:
        if int(roi.width) != int(expected_size) or int(roi.height) != int(expected_size):
            raise ValueError(f"{roi.name} must be {expected_size}x{expected_size}")
        if int(roi.x0) < 0 or int(roi.y0) < 0:
            raise ValueError(f"{roi.name} has negative origin")
        if image_width is not None and roi.x1 > int(image_width):
            raise ValueError(f"{roi.name} exceeds image width")
        if image_height is not None and roi.y1 > int(image_height):
            raise ValueError(f"{roi.name} exceeds image height")
    return checked


def union_crop(rois: Iterable[MetasurfaceRoi]) -> tuple[int, int, int, int]:
    checked = validate_rois(rois)
    left = min(roi.x0 for roi in checked)
    top = min(roi.y0 for roi in checked)
    right = max(roi.x1 for roi in checked)
    bottom = max(roi.y1 for roi in checked)
    return int(left), int(top), int(right - left), int(bottom - top)


def save_rois_json(path: str | Path, rois: Iterable[MetasurfaceRoi], image_path: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checked = validate_rois(rois)
    payload = {
        "roi_size": ROI_SIZE,
        "image_path": image_path,
        "rois": [roi.to_dict() for roi in checked],
        "union_crop": list(union_crop(checked)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rois_json(path: str | Path) -> list[MetasurfaceRoi]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "rois" not in data:
        raise ValueError(f"{path} missing rois")
    return validate_rois([MetasurfaceRoi.from_dict(item) for item in data["rois"]])


def roi_array(rois: Iterable[MetasurfaceRoi]) -> np.ndarray:
    checked = validate_rois(rois)
    return np.asarray([[roi.x0, roi.y0, roi.width, roi.height] for roi in checked], dtype=np.int32)


def concatenate_roi_observations(
    roi_arrays: list[dict],
    rois: Iterable[MetasurfaceRoi],
    units_per_roi: int = 256,
) -> dict:
    checked = validate_rois(rois)
    if len(roi_arrays) != len(checked):
        raise ValueError("roi_arrays and rois must have the same length")
    row_counts = [int(np.asarray(arr["obs_value"]).shape[0]) for arr in roi_arrays]
    merged: dict[str, np.ndarray] = {}
    row_keys = set()
    for arrays, row_count in zip(roi_arrays, row_counts):
        for key, value in arrays.items():
            value = np.asarray(value)
            if value.shape[:1] == (row_count,):
                row_keys.add(key)

    for key in sorted(row_keys):
        parts = []
        for roi_index, arrays in enumerate(roi_arrays):
            value = np.asarray(arrays[key])
            if key == "unit_id":
                parts.append(value.astype(np.int64) + int(roi_index) * int(units_per_roi))
            else:
                parts.append(value)
        merged[key] = np.concatenate(parts, axis=0)

    local_unit = []
    roi_ids = []
    roi_names = []
    for roi_index, (arrays, count) in enumerate(zip(roi_arrays, row_counts)):
        local_unit.append(np.asarray(arrays["unit_id"], dtype=np.int64))
        roi_ids.append(np.full((count,), roi_index, dtype=np.int64))
        roi_names.append(np.asarray([checked[roi_index].name] * count))
    merged["local_unit_id"] = np.concatenate(local_unit, axis=0)
    merged["roi_id"] = np.concatenate(roi_ids, axis=0)
    merged["roi_name"] = np.concatenate(roi_names, axis=0)

    for key, value in roi_arrays[0].items():
        if key not in merged:
            merged[key] = np.asarray(value)

    n_obs = int(merged["obs_value"].shape[0])
    train_end = int(round(n_obs * 0.8))
    val_end = int(round(n_obs * 0.9))
    merged["i_train"] = np.arange(0, train_end, dtype=np.int64)
    merged["i_val"] = np.arange(train_end, val_end, dtype=np.int64)
    merged["i_test"] = np.arange(val_end, n_obs, dtype=np.int64)
    merged["metasurface_rois"] = roi_array(checked)
    merged["metasurface_roi_names"] = np.asarray([roi.name for roi in checked])
    merged["union_roi"] = np.asarray(union_crop(checked), dtype=np.int32)
    return merged
