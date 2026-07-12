"""Build dense all-subcell multi-ROI metasurface observations.

The dense dataset keeps per-row geometry and gray values, but stores spectral
response projections through a compact table plus row indices.  This avoids
replicating the same 12-D response-basis vector for every frame.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ..core.basis import build_gaussian_basis, response_basis_projection
from ..core.response import extract_pixel_response_vectors
from ..core.real_metasurface import build_analyzer_vectors, intersect_rays_with_z0_plane, rays_from_uv

from .spectral_pixel import _load_real_response, _resolve_image_path
from ..config.roi import load_rois_json, save_rois_json, union_crop, validate_rois
from ..config.paths import default_output_dir


ALL_SUBCELL_IDS = tuple(range(16))
POLAR_SUBCELL_IDS = (2, 3, 6, 7)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("dense multi-ROI construction requires opencv-python") from exc
    return cv2


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def center_window_offsets(center_window: int = 3, subcell_size: int = 5) -> tuple[tuple[int, int], ...]:
    center_window = int(center_window)
    subcell_size = int(subcell_size)
    if center_window <= 0 or center_window > subcell_size:
        raise ValueError("center_window must be in [1, subcell_size]")
    if center_window % 2 == 0:
        raise ValueError("center_window must be odd so it is centered")
    start = (subcell_size - center_window) // 2
    values = range(start, start + center_window)
    return tuple((x, y) for y in values for x in values)


def dense_observation_template(
    center_window: int = 3,
    unit_size: int = 20,
    subcells_per_side: int = 4,
    units_per_side: int = 8,
) -> dict[str, np.ndarray]:
    if int(unit_size) % int(subcells_per_side) != 0:
        raise ValueError("unit_size must be divisible by subcells_per_side")
    subcell_size = int(unit_size) // int(subcells_per_side)
    offsets = center_window_offsets(center_window=center_window, subcell_size=subcell_size)

    unit_ids = []
    subcell_ids = []
    local_uv = []
    pixel_offsets = []
    pixel_offset_ids = []
    unit_pixel_rows = []
    unit_pixel_cols = []
    for unit_row in range(int(units_per_side)):
        for unit_col in range(int(units_per_side)):
            unit_id = unit_row * int(units_per_side) + unit_col
            for subcell_id in ALL_SUBCELL_IDS:
                sub_row = subcell_id // int(subcells_per_side)
                sub_col = subcell_id % int(subcells_per_side)
                for offset_id, (offset_x, offset_y) in enumerate(offsets):
                    local_x = unit_col * int(unit_size) + sub_col * subcell_size + offset_x
                    local_y = unit_row * int(unit_size) + sub_row * subcell_size + offset_y
                    unit_ids.append(unit_id)
                    subcell_ids.append(subcell_id)
                    local_uv.append((local_x, local_y))
                    pixel_offsets.append((offset_x, offset_y))
                    pixel_offset_ids.append(offset_id)
                    unit_pixel_rows.append(sub_row * subcell_size + offset_y)
                    unit_pixel_cols.append(sub_col * subcell_size + offset_x)

    subcell_array = np.asarray(subcell_ids, dtype=np.int64)
    obs_type, analyzer = build_analyzer_vectors(subcell_array)
    is_polar = obs_type == "polarization"
    analyzer_id = np.full(subcell_array.shape, -1, dtype=np.int64)
    for row, subcell_id in enumerate(subcell_array):
        if int(subcell_id) in POLAR_SUBCELL_IDS:
            analyzer_id[row] = POLAR_SUBCELL_IDS.index(int(subcell_id))

    return {
        "unit_id": np.asarray(unit_ids, dtype=np.int64),
        "subcell_id": subcell_array,
        "local_uv": np.asarray(local_uv, dtype=np.float32),
        "pixel_offset": np.asarray(pixel_offsets, dtype=np.int64),
        "pixel_offset_id": np.asarray(pixel_offset_ids, dtype=np.int64),
        "unit_pixel_row": np.asarray(unit_pixel_rows, dtype=np.int64),
        "unit_pixel_col": np.asarray(unit_pixel_cols, dtype=np.int64),
        "calibration_pixel_id": (
            np.asarray(unit_pixel_rows, dtype=np.int64) * int(unit_size)
            + np.asarray(unit_pixel_cols, dtype=np.int64)
        ).astype(np.int64),
        "obs_type": obs_type,
        "analyzer": analyzer.astype(np.float32),
        "is_polar": is_polar.astype(bool),
        "analyzer_id": analyzer_id,
        "response_basis_index": np.arange(len(unit_ids), dtype=np.int64),
    }


def build_dense_all_subcell_rows(
    roi,
    frame_id,
    pose_c2w,
    intrinsics,
    template,
    roi_x0,
    roi_y0,
):
    roi = np.asarray(roi, dtype=np.float32)
    if roi.shape != (160, 160):
        raise ValueError("dense ROI must have shape [160,160]")
    pose = np.asarray(pose_c2w, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    local_uv = np.asarray(template["local_uv"], dtype=np.float32)
    full_uv = local_uv + np.asarray([roi_x0, roi_y0], dtype=np.float32)
    count = local_uv.shape[0]
    rays = rays_from_uv(full_uv, np.broadcast_to(pose, (count, 4, 4)), intrinsics)
    local_x = local_uv[:, 0].astype(np.int64)
    local_y = local_uv[:, 1].astype(np.int64)
    return {
        "frame_id": np.full(count, int(frame_id), dtype=np.int64),
        "unit_id": np.asarray(template["unit_id"], dtype=np.int64),
        "local_unit_id": np.asarray(template["unit_id"], dtype=np.int64),
        "subcell_id": np.asarray(template["subcell_id"], dtype=np.int64),
        "is_polar": np.asarray(template["is_polar"], dtype=bool),
        "analyzer_id": np.asarray(template["analyzer_id"], dtype=np.int64),
        "obs_value": (roi[local_y, local_x] / 255.0).astype(np.float32),
        "plane_uv": intersect_rays_with_z0_plane(rays).astype(np.float32),
        "pixel_offset_id": np.asarray(template["pixel_offset_id"], dtype=np.int64),
        "calibration_pixel_id": np.asarray(template["calibration_pixel_id"], dtype=np.int64),
        "response_basis_index": np.asarray(template["response_basis_index"], dtype=np.int64),
    }


def _load_pose_data(dataset_dir: Path):
    pose_path = dataset_dir / "poses_bounds.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)
    with np.load(pose_path, allow_pickle=True) as data:
        required = ("poses_c2w", "frame_ids", "image_names", "K")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{pose_path} is missing fields: {', '.join(missing)}")
        arrays = {name: np.asarray(data[name]) for name in data.files}
    return pose_path, arrays


def _save_pose_with_union_crop(source_pose_arrays: dict, rois, output_path: Path):
    output = {name: np.asarray(value) for name, value in source_pose_arrays.items()}
    output["roi"] = np.asarray(union_crop(rois), dtype=np.int32)
    output["metasurface_rois"] = np.asarray(
        [[roi.x0, roi.y0, roi.width, roi.height] for roi in rois], dtype=np.int32
    )
    output["metasurface_roi_names"] = np.asarray([roi.name for roi in rois])
    np.savez_compressed(output_path, **output)


def _draw_roi_overlay(image, rois):
    cv2 = _require_cv2()
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for index, roi in enumerate(rois):
        color = (0, 0, 255) if index == 0 else (0, 255, 255)
        cv2.rectangle(canvas, (roi.x0, roi.y0), (roi.x1 - 1, roi.y1 - 1), color, 3)
        cv2.putText(canvas, roi.name, (roi.x0, max(20, roi.y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return canvas


def _open_memmaps(tmp_dir: Path, total_count: int):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    from numpy.lib.format import open_memmap

    specs = {
        "frame_id": (np.int32, (total_count,)),
        "roi_id": (np.uint8, (total_count,)),
        "local_unit_id": (np.uint8, (total_count,)),
        "subcell_id": (np.uint8, (total_count,)),
        "is_polar": (bool, (total_count,)),
        "analyzer_id": (np.int8, (total_count,)),
        "pixel_offset_id": (np.uint8, (total_count,)),
        "obs_value": (np.float32, (total_count,)),
        "plane_uv": (np.float32, (total_count, 2)),
        "sensor_key": (np.int32, (total_count,)),
        "response_basis_index": (np.uint16, (total_count,)),
    }
    return {
        name: open_memmap(tmp_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in specs.items()
    }


def _flush_memmaps(memmaps: dict[str, np.ndarray]) -> None:
    for value in memmaps.values():
        if hasattr(value, "flush"):
            value.flush()


def derive_dense_multi_roi_dataset(
    dataset_dir,
    source_image_dir,
    response_mat,
    rois_json,
    output_dir,
    basis_count=12,
    fwhm_nm=20.0,
    center_window=3,
    max_frames=None,
    expected_roi_count=4,
    debug_image_limit=20,
):
    dataset_dir = Path(dataset_dir)
    source_image_dir = Path(source_image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2 = _require_cv2()

    _, pose_arrays = _load_pose_data(dataset_dir)
    poses = np.asarray(pose_arrays["poses_c2w"], dtype=np.float32)
    frame_ids = np.asarray(pose_arrays["frame_ids"], dtype=np.int64)
    image_names = np.asarray(pose_arrays["image_names"])
    intrinsics = np.asarray(pose_arrays["K"], dtype=np.float32)
    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        poses = poses[:max_frames]
        frame_ids = frame_ids[:max_frames]
        image_names = image_names[:max_frames]
        for key in ("poses_c2w", "frame_ids", "image_names"):
            pose_arrays[key] = np.asarray(pose_arrays[key])[:max_frames]

    rois = load_rois_json(rois_json)
    if expected_roi_count is not None and len(rois) != int(expected_roi_count):
        raise ValueError(f"expected {expected_roi_count} ROIs, got {len(rois)}")

    a_matrix, wavelengths = _load_real_response(response_mat)
    spectral_centers = np.linspace(float(wavelengths[0]), float(wavelengths[-1]), int(basis_count), dtype=np.float32)
    basis = build_gaussian_basis(wavelengths, spectral_centers, fwhm_nm=fwhm_nm)
    template = dense_observation_template(center_window=center_window)
    response = extract_pixel_response_vectors(
        a_matrix,
        unit_ids=template["unit_id"],
        unit_rows=template["unit_pixel_row"],
        unit_cols=template["unit_pixel_col"],
    )
    response_basis_table = response_basis_projection(response, basis).astype(np.float32)

    rows_per_roi = int(template["unit_id"].shape[0])
    rows_per_frame = rows_per_roi * len(rois)
    total_count = int(len(frame_ids) * rows_per_frame)
    tmp_dir = output_dir / "_dense_memmap_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    memmaps = _open_memmaps(tmp_dir, total_count)
    write_cursor = 0
    for index, (frame_id, pose, image_name) in enumerate(zip(frame_ids, poses, image_names)):
        image_path = _resolve_image_path(image_name, source_image_dir)
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"failed to read image: {image_path}")
        if index == 0:
            rois = validate_rois(rois, image_width=gray.shape[1], image_height=gray.shape[0])
            save_rois_json(output_dir / "metasurface_rois.json", rois, image_path=str(image_path))
        if int(debug_image_limit) > 0 and index < int(debug_image_limit):
            overlay_dir = output_dir / "roi_overlay"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_dir / f"frame_{int(frame_id):04d}.png"), _draw_roi_overlay(gray, rois))

        for roi_index, roi in enumerate(rois):
            crop = gray[roi.y0 : roi.y1, roi.x0 : roi.x1]
            rows = build_dense_all_subcell_rows(
                roi=crop,
                frame_id=int(frame_id),
                pose_c2w=pose,
                intrinsics=intrinsics,
                template=template,
                roi_x0=roi.x0,
                roi_y0=roi.y0,
            )
            row_count = int(rows["obs_value"].shape[0])
            start = write_cursor
            stop = start + row_count
            global_unit_id = rows["unit_id"].astype(np.int64) + roi_index * 256
            sensor_key = global_unit_id * 400 + rows["calibration_pixel_id"].astype(np.int64)
            memmaps["frame_id"][start:stop] = int(frame_id)
            memmaps["roi_id"][start:stop] = int(roi_index)
            memmaps["local_unit_id"][start:stop] = rows["local_unit_id"].astype(np.uint8)
            memmaps["subcell_id"][start:stop] = rows["subcell_id"].astype(np.uint8)
            memmaps["is_polar"][start:stop] = rows["is_polar"].astype(bool)
            memmaps["analyzer_id"][start:stop] = rows["analyzer_id"].astype(np.int8)
            memmaps["pixel_offset_id"][start:stop] = rows["pixel_offset_id"].astype(np.uint8)
            memmaps["obs_value"][start:stop] = rows["obs_value"].astype(np.float32)
            memmaps["plane_uv"][start:stop] = rows["plane_uv"].astype(np.float32)
            memmaps["sensor_key"][start:stop] = sensor_key.astype(np.int32)
            memmaps["response_basis_index"][start:stop] = rows["response_basis_index"].astype(np.uint16)
            write_cursor = stop
        if (index + 1) % 50 == 0 or index + 1 == len(frame_ids):
            print(f"[dense-multi-roi] processed {index + 1}/{len(frame_ids)} frames", flush=True)

    if write_cursor != total_count:
        raise RuntimeError(f"dense writer filled {write_cursor} rows, expected {total_count}")
    _flush_memmaps(memmaps)
    if total_count == 0:
        raise ValueError("no dense observations were generated")
    arrays = dict(memmaps)

    arrays["response_basis_table"] = response_basis_table
    arrays["template_local_uv"] = np.asarray(template["local_uv"], dtype=np.float32)
    arrays["template_analyzer"] = np.asarray(template["analyzer"], dtype=np.float32)
    arrays["template_unit_id"] = np.asarray(template["unit_id"], dtype=np.uint8)
    arrays["template_subcell_id"] = np.asarray(template["subcell_id"], dtype=np.uint8)
    arrays["template_pixel_offset"] = np.asarray(template["pixel_offset"], dtype=np.uint8)
    arrays["template_pixel_offset_id"] = np.asarray(template["pixel_offset_id"], dtype=np.uint8)
    arrays["template_calibration_pixel_id"] = np.asarray(template["calibration_pixel_id"], dtype=np.uint16)
    arrays["template_is_polar"] = np.asarray(template["is_polar"], dtype=bool)
    arrays["template_analyzer_id"] = np.asarray(template["analyzer_id"], dtype=np.int8)
    arrays["wavelengths"] = wavelengths.astype(np.float32)
    arrays["spectral_centers_nm"] = spectral_centers
    arrays["fwhm_nm"] = np.asarray(float(fwhm_nm), dtype=np.float32)
    arrays["metasurface_rois"] = np.asarray([[roi.x0, roi.y0, roi.width, roi.height] for roi in rois], dtype=np.int32)
    arrays["metasurface_roi_names"] = np.asarray([roi.name for roi in rois])
    arrays["union_roi"] = np.asarray(union_crop(rois), dtype=np.int32)
    arrays["near"] = np.asarray([0.0], dtype=np.float32)
    arrays["far"] = np.asarray([1.0], dtype=np.float32)
    count = int(total_count)

    observation_path = output_dir / "observations_dense_all16_center3x3_indexed.npz"
    pose_path = output_dir / "poses_bounds_multi_roi.npz"
    np.savez(observation_path, **arrays)
    _save_pose_with_union_crop(pose_arrays, rois, pose_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    report = {
        "source_dataset_dir": str(dataset_dir),
        "source_image_dir": str(source_image_dir),
        "response_mat": str(response_mat),
        "rois_json": str(rois_json),
        "observation_path": str(observation_path),
        "pose_path": str(pose_path),
        "frame_count": int(len(frame_ids)),
        "roi_count": int(len(rois)),
        "rows_per_roi_per_frame": int(template["unit_id"].shape[0]),
        "rows_per_frame": int(template["unit_id"].shape[0] * len(rois)),
        "observation_count": int(count),
        "subcell_ids": list(ALL_SUBCELL_IDS),
        "polar_subcell_ids": list(POLAR_SUBCELL_IDS),
        "center_window": int(center_window),
        "response_storage": "response_basis_table_plus_response_basis_index",
        "mode": "dense_all16_center3x3_multi_roi",
    }
    _write_json(output_dir / "dense_dataset_report.json", report)
    return observation_path, pose_path, report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", "--base-dataset", dest="dataset_dir", required=True)
    parser.add_argument("--source-image-dir", default="G:/galaxydata/pawn")
    parser.add_argument("--response-mat", required=True)
    parser.add_argument("--rois-json", "--multi-roi-json", dest="rois_json", required=True)
    parser.add_argument("--output-dir", default=str(default_output_dir("dense_all16_center3x3_indexed")))
    parser.add_argument("--basis-count", type=int, default=12)
    parser.add_argument("--fwhm-nm", type=float, default=20.0)
    parser.add_argument("--center-window", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--expected-roi-count", type=int, default=4)
    parser.add_argument("--debug-image-limit", type=int, default=20)
    parser.add_argument("--all-subcells", action="store_true", help="Accepted for interface clarity; dense v1 always uses all 16 subcells.")
    parser.add_argument("--include-polar", action="store_true", help="Accepted for interface clarity; dense v1 always includes polar subcells.")
    return parser


def main():
    args = build_parser().parse_args()
    _, _, report = derive_dense_multi_roi_dataset(
        dataset_dir=args.dataset_dir,
        source_image_dir=args.source_image_dir,
        response_mat=args.response_mat,
        rois_json=args.rois_json,
        output_dir=args.output_dir,
        basis_count=args.basis_count,
        fwhm_nm=args.fwhm_nm,
        center_window=args.center_window,
        max_frames=args.max_frames,
        expected_roi_count=args.expected_roi_count,
        debug_image_limit=args.debug_image_limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
