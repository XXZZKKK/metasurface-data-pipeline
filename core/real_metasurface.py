"""Real-image helpers shared by metasurface observation builders."""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np

from .response import extract_center2x2_response_table


@dataclass(frozen=True)
class RoiConfig:
    """Fixed MATLAB ROI converted to Python crop coordinates."""

    matlab_x: int = 1747
    matlab_y: int = 1020
    size: int = 160
    unit_size: int = 20

    @property
    def x0(self) -> int:
        return self.matlab_x - 1

    @property
    def y0(self) -> int:
        return self.matlab_y - 1


MATLAB_ROI = RoiConfig()
POLAR_SUBCELL_ANGLES = {2: 0.0, 3: 45.0, 6: 90.0, 7: 135.0}


def _require_scipy_io():
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError("loading A_matrix_normalized.mat requires scipy") from exc
    return sio


def crop_metasurface_roi(image, roi=MATLAB_ROI):
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    y1 = roi.y0 + roi.size
    x1 = roi.x0 + roi.size
    if arr.shape[0] < y1 or arr.shape[1] < x1:
        raise ValueError(f"image shape {arr.shape} is smaller than ROI ending at {(x1, y1)}")
    return arr[roi.y0:y1, roi.x0:x1]


def extract_center2x2_observation_values(roi_image, unit_size=20, subcells_per_side=4):
    roi = np.asarray(roi_image, dtype=np.float32)
    if roi.ndim != 2:
        raise ValueError("roi_image must be a 2D grayscale image")
    if roi.shape[0] != roi.shape[1]:
        raise ValueError("roi_image must be square")
    if roi.shape[0] % unit_size != 0:
        raise ValueError("ROI side length must be divisible by unit_size")
    if unit_size % subcells_per_side != 0:
        raise ValueError("unit_size must be divisible by subcells_per_side")

    units_per_side = roi.shape[0] // unit_size
    subcell_size = unit_size // subcells_per_side
    if subcell_size < 4:
        raise ValueError("center2x2 extraction requires subcell blocks at least 4x4")

    values = []
    unit_ids = []
    subcell_ids = []
    center_uv = []
    for unit_row in range(units_per_side):
        for unit_col in range(units_per_side):
            unit_id = unit_row * units_per_side + unit_col
            unit_y = unit_row * unit_size
            unit_x = unit_col * unit_size
            for sub_row in range(subcells_per_side):
                for sub_col in range(subcells_per_side):
                    subcell_id = sub_row * subcells_per_side + sub_col
                    sub_y = unit_y + sub_row * subcell_size
                    sub_x = unit_x + sub_col * subcell_size
                    yy = np.array([sub_y + 2, sub_y + 3], dtype=np.int64)
                    xx = np.array([sub_x + 2, sub_x + 3], dtype=np.int64)
                    values.append(float(roi[np.ix_(yy, xx)].mean()))
                    unit_ids.append(unit_id)
                    subcell_ids.append(subcell_id)
                    center_uv.append((float(xx.mean()), float(yy.mean())))

    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(unit_ids, dtype=np.int64),
        np.asarray(subcell_ids, dtype=np.int64),
        np.asarray(center_uv, dtype=np.float32),
    )


def rays_from_uv(center_uv, poses_c2w, intrinsics):
    """Build world rays for OpenCV camera poses: +Z forward and +Y down."""
    uv = np.asarray(center_uv, dtype=np.float32)
    poses = np.asarray(poses_c2w, dtype=np.float32)
    k = np.asarray(intrinsics, dtype=np.float32)
    if k.ndim == 2:
        k = np.broadcast_to(k[None, ...], (uv.shape[0], 3, 3))

    u = uv[:, 0]
    v = uv[:, 1]
    dirs = np.stack(
        [
            (u - k[:, 0, 2]) / k[:, 0, 0],
            (v - k[:, 1, 2]) / k[:, 1, 1],
            np.ones_like(u),
        ],
        axis=-1,
    )
    rays_d = np.sum(dirs[:, None, :] * poses[:, :3, :3], axis=-1)
    rays_d = rays_d / np.maximum(np.linalg.norm(rays_d, axis=-1, keepdims=True), 1e-8)
    rays_o = poses[:, :3, 3]
    return np.stack([rays_o, rays_d], axis=1).astype(np.float32)


def intersect_rays_with_z0_plane(rays):
    rays = np.asarray(rays, dtype=np.float32)
    origins = rays[:, 0, :]
    dirs = rays[:, 1, :]
    denom = dirs[:, 2]
    safe = np.where(
        np.abs(denom) < 1e-8,
        np.sign(denom) * 1e-8 + (denom == 0) * 1e-8,
        denom,
    )
    t = -origins[:, 2] / safe
    points = origins + dirs * t[:, None]
    return points[:, :2].astype(np.float32)


def build_analyzer_vectors(subcell_ids):
    analyzers = np.zeros((len(subcell_ids), 3), dtype=np.float32)
    obs_types = np.asarray(["spectral"] * len(subcell_ids), dtype="<U16")
    for row, subcell_id in enumerate(subcell_ids):
        angle = POLAR_SUBCELL_ANGLES.get(int(subcell_id))
        if angle is None:
            continue
        theta = math.radians(angle)
        analyzers[row] = 0.5 * np.asarray(
            [1.0, math.cos(2.0 * theta), math.sin(2.0 * theta)],
            dtype=np.float32,
        )
        obs_types[row] = "polarization"
    return obs_types, analyzers


def build_real_observation_arrays(
    frame_ids,
    roi_images,
    poses_c2w,
    intrinsics,
    response_table,
    wavelengths,
    roi_x0,
    roi_y0,
    near=0.0,
    far=1.0,
):
    frame_values = []
    unit_values = []
    subcell_values = []
    obs_type_values = []
    channel_values = []
    obs_values = []
    center_values = []
    pose_values = []
    k_values = []
    response_values = []
    analyzer_values = []
    ray_values = []
    plane_uv_values = []

    response_table = np.asarray(response_table, dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    calib_units = response_table.shape[2]
    calib_side = int(round(math.sqrt(calib_units)))

    for frame_id, roi, pose in zip(frame_ids, roi_images, poses_c2w):
        values, unit_ids, subcell_ids, local_centers = extract_center2x2_observation_values(roi)
        centers = local_centers + np.asarray([roi_x0, roi_y0], dtype=np.float32)[None, :]
        obs_types, analyzers = build_analyzer_vectors(subcell_ids)
        rays = rays_from_uv(
            centers,
            np.broadcast_to(np.asarray(pose, dtype=np.float32), (len(centers), 4, 4)),
            intrinsics,
        )
        plane_uv = intersect_rays_with_z0_plane(rays)

        for idx in range(len(values)):
            unit_id = int(unit_ids[idx])
            unit_row = unit_id // 8
            unit_col = unit_id % 8
            calib_id = (unit_row % calib_side) * calib_side + (unit_col % calib_side)
            frame_values.append(int(frame_id))
            unit_values.append(unit_id)
            subcell_values.append(int(subcell_ids[idx]))
            obs_type_values.append(str(obs_types[idx]))
            channel_values.append(int(subcell_ids[idx]))
            obs_values.append(float(values[idx]) / 255.0)
            center_values.append(centers[idx])
            pose_values.append(pose)
            k_values.append(intrinsics)
            response_values.append(response_table[int(subcell_ids[idx]), :, calib_id])
            analyzer_values.append(analyzers[idx])
            ray_values.append(rays[idx])
            plane_uv_values.append(plane_uv[idx])

    n_obs = len(obs_values)
    train_end = int(round(n_obs * 0.8))
    val_end = int(round(n_obs * 0.9))
    return {
        "frame_id": np.asarray(frame_values, dtype=np.int64),
        "unit_id": np.asarray(unit_values, dtype=np.int64),
        "subcell_id": np.asarray(subcell_values, dtype=np.int64),
        "obs_type": np.asarray(obs_type_values),
        "channel_id": np.asarray(channel_values, dtype=np.int64),
        "obs_value": np.asarray(obs_values, dtype=np.float32),
        "center_uv": np.asarray(center_values, dtype=np.float32),
        "pose_c2w": np.asarray(pose_values, dtype=np.float32),
        "K_roi": np.asarray(k_values, dtype=np.float32),
        "near": np.asarray([near], dtype=np.float32),
        "far": np.asarray([far], dtype=np.float32),
        "response": np.asarray(response_values, dtype=np.float32),
        "wavelengths": wavelengths,
        "analyzer": np.asarray(analyzer_values, dtype=np.float32),
        "rays": np.asarray(ray_values, dtype=np.float32),
        "plane_uv": np.asarray(plane_uv_values, dtype=np.float32),
        "i_train": np.arange(0, train_end, dtype=np.int64),
        "i_val": np.arange(train_end, val_end, dtype=np.int64),
        "i_test": np.arange(val_end, n_obs, dtype=np.int64),
    }


def load_center2x2_response_from_mat(mat_path):
    mat = _require_scipy_io().loadmat(str(mat_path))
    if "A_matrix" not in mat:
        raise ValueError(f"{mat_path} does not contain A_matrix")
    wavelengths = np.asarray(mat.get("wavelengths")).reshape(-1).astype(np.float32)
    response = np.asarray(extract_center2x2_response_table(mat["A_matrix"]), dtype=np.float32)
    return response, wavelengths


def write_json(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
