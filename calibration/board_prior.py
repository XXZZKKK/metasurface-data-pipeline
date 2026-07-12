"""Metric lookup for the printed AprilGrid board texture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardPrior:
    image: np.ndarray
    marker_origin_px: np.ndarray
    pixels_per_meter: float

    @classmethod
    def from_files(cls, json_path: str | Path, png_path: str | Path) -> "BoardPrior":
        metadata = json.loads(Path(json_path).read_text(encoding="utf-8"))
        try:
            marker_length_m = float(metadata["marker_length_m"])
            board = metadata["board"]
            marker_length_px = float(board["marker_length_px"])
            marker_origin_px = np.asarray(
                board["marker_origin_in_canvas_px"], dtype=np.float32
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("missing board metric metadata") from exc

        if marker_length_m <= 0.0 or marker_length_px <= 0.0:
            raise ValueError("board metric lengths must be positive")
        if marker_origin_px.shape != (2,) or not np.isfinite(marker_origin_px).all():
            raise ValueError("board marker origin must contain two finite pixel coordinates")

        image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"failed to read board image: {png_path}")
        if image.ndim != 2:
            raise ValueError("board image must be grayscale")

        immutable_image = np.asarray(image, dtype=np.uint8)
        immutable_image.setflags(write=False)
        marker_origin_px.setflags(write=False)
        return cls(
            image=immutable_image,
            marker_origin_px=marker_origin_px,
            pixels_per_meter=marker_length_px / marker_length_m,
        )

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy_array = np.asarray(xy, dtype=np.float32)
        if xy_array.ndim != 2 or xy_array.shape[1] != 2:
            raise ValueError("world xy must have shape [N, 2]")
        if not np.isfinite(xy_array).all():
            raise ValueError("world xy must be finite")
        return self.marker_origin_px[None, :] + xy_array * self.pixels_per_meter

    def sample(self, xy: np.ndarray) -> np.ndarray:
        pixels = self.world_to_pixel(xy)
        height, width = self.image.shape
        inside = (
            (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= width - 1)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= height - 1)
        )
        if not np.all(inside):
            count = int((~inside).sum())
            raise ValueError(f"{count} board coordinates are outside board canvas")

        values = np.empty(len(pixels), dtype=np.float32)
        chunk_size = 32_000
        for start in range(0, len(pixels), chunk_size):
            chunk = pixels[start : start + chunk_size]
            remapped = cv2.remap(
                self.image,
                chunk[:, 0].reshape(-1, 1),
                chunk[:, 1].reshape(-1, 1),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            values[start : start + len(chunk)] = remapped.reshape(-1)
        return values / 255.0
