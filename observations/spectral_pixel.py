"""Derive independent spectral-pixel observations from the accepted cube2 poses."""

from __future__ import annotations

import json
import pathlib

import numpy as np

from ..core.basis import (
    build_gaussian_basis,
    response_basis_projection,
)
from ..core.response import extract_pixel_response_vectors
from ..core.real_metasurface import (
    MATLAB_ROI,
    crop_metasurface_roi,
    intersect_rays_with_z0_plane,
    rays_from_uv,
)


SPECTRAL_SUBCELL_IDS = (0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15)
CENTER_PIXEL_OFFSETS = ((2, 2), (3, 2), (2, 3), (3, 3))


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cube2 pixel dataset construction requires opencv-python") from exc
    return cv2


def _require_scipy_io():
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError("cube2 pixel dataset construction requires scipy") from exc
    return sio


def _observation_template(unit_size=20, subcells_per_side=4, units_per_side=8):
    if unit_size % subcells_per_side != 0:
        raise ValueError("unit_size must be divisible by subcells_per_side")
    subcell_size = unit_size // subcells_per_side
    if subcell_size < 4:
        raise ValueError("center 2x2 pixels require subcells at least 4x4")

    unit_ids = []
    subcell_ids = []
    local_uv = []
    pixel_offsets = []
    unit_pixel_rows = []
    unit_pixel_cols = []
    for unit_row in range(units_per_side):
        for unit_col in range(units_per_side):
            unit_id = unit_row * units_per_side + unit_col
            for subcell_id in SPECTRAL_SUBCELL_IDS:
                sub_row = subcell_id // subcells_per_side
                sub_col = subcell_id % subcells_per_side
                for offset_x, offset_y in CENTER_PIXEL_OFFSETS:
                    local_x = unit_col * unit_size + sub_col * subcell_size + offset_x
                    local_y = unit_row * unit_size + sub_row * subcell_size + offset_y
                    unit_ids.append(unit_id)
                    subcell_ids.append(subcell_id)
                    local_uv.append((local_x, local_y))
                    pixel_offsets.append((offset_x, offset_y))
                    unit_pixel_rows.append(sub_row * subcell_size + offset_y)
                    unit_pixel_cols.append(sub_col * subcell_size + offset_x)

    return {
        "unit_id": np.asarray(unit_ids, dtype=np.int64),
        "subcell_id": np.asarray(subcell_ids, dtype=np.int64),
        "local_uv": np.asarray(local_uv, dtype=np.float32),
        "pixel_offset": np.asarray(pixel_offsets, dtype=np.int64),
        "unit_pixel_row": np.asarray(unit_pixel_rows, dtype=np.int64),
        "unit_pixel_col": np.asarray(unit_pixel_cols, dtype=np.int64),
    }


def build_independent_spectral_rows(
    roi,
    frame_id,
    pose_c2w,
    intrinsics,
    a_matrix,
    wavelengths,
    roi_x0,
    roi_y0,
):
    """Build four independent center-pixel observations for 12 spectral subcells."""
    roi = np.asarray(roi, dtype=np.float32)
    if roi.shape != (MATLAB_ROI.size, MATLAB_ROI.size):
        raise ValueError(f"roi must have shape {(MATLAB_ROI.size, MATLAB_ROI.size)}")
    pose = np.asarray(pose_c2w, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    a_matrix = np.asarray(a_matrix, dtype=np.float32)
    if pose.shape != (4, 4):
        raise ValueError("pose_c2w must have shape [4,4]")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3,3]")
    if a_matrix.ndim != 3 or a_matrix.shape[0] != 400:
        raise ValueError("a_matrix must have shape [400,L,U]")
    if a_matrix.shape[1] != wavelengths.size:
        raise ValueError("A-matrix wavelength dimension must match wavelengths")

    template = _observation_template()
    local_uv = template["local_uv"]
    full_uv = local_uv + np.asarray([roi_x0, roi_y0], dtype=np.float32)
    count = local_uv.shape[0]
    poses = np.broadcast_to(pose, (count, 4, 4))
    rays = rays_from_uv(full_uv, poses, intrinsics)
    responses = extract_pixel_response_vectors(
        a_matrix,
        unit_ids=template["unit_id"],
        unit_rows=template["unit_pixel_row"],
        unit_cols=template["unit_pixel_col"],
    )

    local_x = local_uv[:, 0].astype(np.int64)
    local_y = local_uv[:, 1].astype(np.int64)
    return {
        "frame_id": np.full(count, int(frame_id), dtype=np.int64),
        "unit_id": template["unit_id"],
        "subcell_id": template["subcell_id"],
        "channel_id": template["subcell_id"].copy(),
        "obs_type": np.full(count, "spectral", dtype="<U16"),
        "obs_value": (roi[local_y, local_x] / 255.0).astype(np.float32),
        "center_uv": full_uv.astype(np.float32),
        "response": responses,
        "wavelengths": wavelengths,
        "analyzer": np.zeros((count, 3), dtype=np.float32),
        "rays": rays,
        "plane_uv": intersect_rays_with_z0_plane(rays),
        "pixel_offset": template["pixel_offset"],
        "calibration_pixel_id": (
            template["unit_pixel_row"] * 20 + template["unit_pixel_col"]
        ).astype(np.int64),
    }


def _load_real_response(path):
    data = _require_scipy_io().loadmat(str(path))
    missing = [name for name in ("A_matrix", "wavelengths") if name not in data]
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
    a_matrix = np.asarray(data["A_matrix"], dtype=np.float32)
    wavelengths = np.asarray(data["wavelengths"], dtype=np.float32).reshape(-1)
    if a_matrix.shape != (400, wavelengths.size, 64):
        raise ValueError(
            f"expected A_matrix shape (400,{wavelengths.size},64), got {a_matrix.shape}"
        )
    if not np.isfinite(a_matrix).all() or not np.isfinite(wavelengths).all():
        raise ValueError("A_matrix or wavelengths contains non-finite values")
    return a_matrix, wavelengths


def _resolve_image_path(saved_name, source_image_dir):
    saved = pathlib.Path(str(saved_name))
    if saved.is_file():
        return saved
    candidate = pathlib.Path(source_image_dir) / saved.name
    if not candidate.is_file():
        raise FileNotFoundError(f"cube2 source image not found: {saved}")
    return candidate


def _write_json(path, payload):
    pathlib.Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def derive_independent_spectral_dataset(
    dataset_dir,
    source_image_dir,
    response_mat,
    output_dir,
    basis_count=12,
    fwhm_nm=20.0,
    max_frames=None,
):
    """Create an observations.npz-compatible real basis-S0 dataset."""
    dataset_dir = pathlib.Path(dataset_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_path = dataset_dir / "poses_bounds.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"missing accepted pose file: {pose_path}")

    with np.load(pose_path, allow_pickle=True) as pose_data:
        required = ("poses_c2w", "frame_ids", "image_names", "K")
        missing = [name for name in required if name not in pose_data.files]
        if missing:
            raise ValueError(f"{pose_path} is missing fields: {', '.join(missing)}")
        poses = np.asarray(pose_data["poses_c2w"], dtype=np.float32)
        frame_ids = np.asarray(pose_data["frame_ids"], dtype=np.int64)
        image_names = np.asarray(pose_data["image_names"])
        intrinsics = np.asarray(pose_data["K"], dtype=np.float32)

    if not (len(poses) == len(frame_ids) == len(image_names)):
        raise ValueError("poses, frame_ids, and image_names must have matching lengths")
    if poses.shape[1:] != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("invalid pose or intrinsic matrix shape")
    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        poses = poses[:max_frames]
        frame_ids = frame_ids[:max_frames]
        image_names = image_names[:max_frames]

    a_matrix, wavelengths = _load_real_response(response_mat)
    cv2 = _require_cv2()
    row_chunks = []
    response_template = None
    for index, (frame_id, pose, image_name) in enumerate(zip(frame_ids, poses, image_names)):
        image_path = _resolve_image_path(image_name, source_image_dir)
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"failed to read cube2 image: {image_path}")
        roi = crop_metasurface_roi(gray, MATLAB_ROI)
        rows = build_independent_spectral_rows(
            roi=roi,
            frame_id=int(frame_id),
            pose_c2w=pose,
            intrinsics=intrinsics,
            a_matrix=a_matrix,
            wavelengths=wavelengths,
            roi_x0=MATLAB_ROI.x0,
            roi_y0=MATLAB_ROI.y0,
        )
        if response_template is None:
            response_template = rows.pop("response")
        else:
            rows.pop("response")
        rows.pop("wavelengths")
        row_chunks.append(rows)
        if (index + 1) % 50 == 0 or index + 1 == len(frame_ids):
            print(f"[pixel4] processed {index + 1}/{len(frame_ids)} frames", flush=True)

    if not row_chunks or response_template is None:
        raise ValueError("no independent observations were generated")
    rows_per_frame = int(response_template.shape[0])
    arrays = {}
    for key in row_chunks[0]:
        arrays[key] = np.concatenate([chunk[key] for chunk in row_chunks], axis=0)
    spectral_centers = np.linspace(
        float(wavelengths[0]),
        float(wavelengths[-1]),
        int(basis_count),
        dtype=np.float32,
    )
    basis = build_gaussian_basis(wavelengths, spectral_centers, fwhm_nm=fwhm_nm)
    response_basis_template = response_basis_projection(response_template, basis)
    arrays["response_basis"] = np.tile(
        response_basis_template,
        (len(row_chunks), 1),
    ).astype(np.float32)
    arrays["wavelengths"] = wavelengths
    arrays["near"] = np.asarray([0.0], dtype=np.float32)
    arrays["far"] = np.asarray([1.0], dtype=np.float32)
    count = int(arrays["obs_value"].shape[0])
    indices = np.arange(count, dtype=np.int64)
    arrays["i_train"] = indices
    arrays["i_val"] = indices
    arrays["i_test"] = indices

    observation_path = output_dir / "observations_pixel4_spectral.npz"
    response_path = output_dir / "response_pixel4_basis_metadata.npz"
    np.savez_compressed(observation_path, **arrays)
    np.savez_compressed(
        response_path,
        A_matrix=a_matrix,
        wavelengths=wavelengths,
        spectral_centers_nm=spectral_centers,
        fwhm_nm=np.asarray(float(fwhm_nm), dtype=np.float32),
    )

    report = {
        "source_dataset_dir": str(dataset_dir),
        "source_image_dir": str(source_image_dir),
        "response_mat": str(response_mat),
        "observation_path": str(observation_path),
        "response_path": str(response_path),
        "frame_count": int(len(frame_ids)),
        "rows_per_frame": rows_per_frame,
        "observation_count": count,
        "spectral_subcell_ids": list(SPECTRAL_SUBCELL_IDS),
        "pixel_offsets": [list(value) for value in CENTER_PIXEL_OFFSETS],
        "basis_count": int(basis_count),
        "fwhm_nm": float(fwhm_nm),
        "response_source": "real_A_matrix_pixel_specific",
        "response_storage": "preprojected_normalized_response_times_basis",
    }
    report_path = output_dir / "pixel4_dataset_report.json"
    _write_json(report_path, report)
    return observation_path, response_path, report
