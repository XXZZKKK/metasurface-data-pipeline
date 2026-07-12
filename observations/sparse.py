"""Build a multi-ROI metasurface observation dataset from accepted AprilGrid poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..core.basis import build_gaussian_basis, response_basis_projection

from .spectral_pixel import (
    _load_real_response,
    _resolve_image_path,
    build_independent_spectral_rows,
)
from ..config.roi import (
    concatenate_roi_observations,
    load_rois_json,
    save_rois_json,
    union_crop,
    validate_rois,
)
from ..config.paths import default_output_dir


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("multi-ROI observation construction requires opencv-python") from exc
    return cv2


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def derive_multi_roi_spectral_dataset(
    dataset_dir,
    source_image_dir,
    response_mat,
    rois_json,
    output_dir,
    basis_count=12,
    fwhm_nm=20.0,
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
    if not (len(poses) == len(frame_ids) == len(image_names)):
        raise ValueError("poses, frame_ids, and image_names must have matching lengths")

    rois = load_rois_json(rois_json)
    if expected_roi_count is not None and len(rois) != int(expected_roi_count):
        raise ValueError(f"expected {expected_roi_count} ROIs, got {len(rois)}")
    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        poses = poses[:max_frames]
        frame_ids = frame_ids[:max_frames]
        image_names = image_names[:max_frames]
        for key in ("poses_c2w", "frame_ids", "image_names"):
            pose_arrays[key] = np.asarray(pose_arrays[key])[:max_frames]

    a_matrix, wavelengths = _load_real_response(response_mat)
    spectral_centers = np.linspace(
        float(wavelengths[0]),
        float(wavelengths[-1]),
        int(basis_count),
        dtype=np.float32,
    )
    basis = build_gaussian_basis(wavelengths, spectral_centers, fwhm_nm=fwhm_nm)

    row_chunks = []
    response_basis_template = None
    rows_per_roi = None
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

        per_roi = []
        for roi_index, roi in enumerate(rois):
            crop = gray[roi.y0 : roi.y1, roi.x0 : roi.x1]
            rows = build_independent_spectral_rows(
                roi=crop,
                frame_id=int(frame_id),
                pose_c2w=pose,
                intrinsics=intrinsics,
                a_matrix=a_matrix,
                wavelengths=wavelengths,
                roi_x0=roi.x0,
                roi_y0=roi.y0,
            )
            response = rows.pop("response")
            rows.pop("wavelengths")
            if response_basis_template is None:
                response_basis_template = response_basis_projection(response, basis)
                rows_per_roi = int(response_basis_template.shape[0])
            rows["roi_id"] = np.full((rows["obs_value"].shape[0],), roi_index, dtype=np.int64)
            rows["roi_name"] = np.asarray([roi.name] * rows["obs_value"].shape[0])
            per_roi.append(rows)
        row_chunks.append(concatenate_roi_observations(per_roi, rois))
        if (index + 1) % 50 == 0 or index + 1 == len(frame_ids):
            print(f"[multi-roi] processed {index + 1}/{len(frame_ids)} frames", flush=True)

    if not row_chunks or response_basis_template is None:
        raise ValueError("no multi-ROI observations were generated")
    arrays = {}
    for key in row_chunks[0]:
        if key in ("wavelengths", "metasurface_rois", "metasurface_roi_names", "union_roi"):
            arrays[key] = row_chunks[0][key]
        elif np.asarray(row_chunks[0][key]).shape[:1] == (row_chunks[0]["obs_value"].shape[0],):
            arrays[key] = np.concatenate([chunk[key] for chunk in row_chunks], axis=0)
        else:
            arrays[key] = row_chunks[0][key]

    if rows_per_roi is None:
        raise RuntimeError("rows_per_roi was not initialized")
    response_basis_per_frame = np.tile(response_basis_template, (len(rois), 1)).astype(np.float32)
    arrays["response_basis"] = np.tile(response_basis_per_frame, (len(row_chunks), 1)).astype(np.float32)
    arrays["wavelengths"] = wavelengths.astype(np.float32)
    arrays["near"] = np.asarray([0.0], dtype=np.float32)
    arrays["far"] = np.asarray([1.0], dtype=np.float32)
    n_obs = int(arrays["obs_value"].shape[0])
    indices = np.arange(n_obs, dtype=np.int64)
    arrays["i_train"] = indices
    arrays["i_val"] = indices
    arrays["i_test"] = indices

    observation_path = output_dir / "observations_multi_roi_pixel4_spectral.npz"
    pose_path = output_dir / "poses_bounds_multi_roi.npz"
    response_path = output_dir / "response_multi_roi_basis_metadata.npz"
    np.savez_compressed(observation_path, **arrays)
    _save_pose_with_union_crop(pose_arrays, rois, pose_path)
    np.savez_compressed(
        response_path,
        wavelengths=wavelengths.astype(np.float32),
        spectral_centers_nm=spectral_centers,
        fwhm_nm=np.asarray(float(fwhm_nm), dtype=np.float32),
        basis=basis.astype(np.float32),
    )

    report = {
        "source_dataset_dir": str(dataset_dir),
        "source_image_dir": str(source_image_dir),
        "response_mat": str(response_mat),
        "rois_json": str(rois_json),
        "observation_path": str(observation_path),
        "pose_path": str(pose_path),
        "response_path": str(response_path),
        "frame_count": int(len(frame_ids)),
        "roi_count": int(len(rois)),
        "rows_per_roi_per_frame": int(rows_per_roi),
        "rows_per_frame": int(rows_per_roi * len(rois)),
        "observation_count": int(n_obs),
        "union_roi": list(union_crop(rois)),
        "metasurface_rois": [roi.to_dict() for roi in rois],
        "basis_count": int(basis_count),
        "fwhm_nm": float(fwhm_nm),
        "mode": "spectral_pixel4_multi_roi",
    }
    _write_json(output_dir / "multi_roi_observation_report.json", report)
    return observation_path, pose_path, response_path, report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-image-dir", required=True)
    parser.add_argument("--response-mat", required=True)
    parser.add_argument("--rois-json", required=True)
    parser.add_argument("--output-dir", default=str(default_output_dir("multi_roi_sparse")))
    parser.add_argument("--basis-count", type=int, default=12)
    parser.add_argument("--fwhm-nm", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--expected-roi-count", type=int, default=4)
    parser.add_argument("--debug-image-limit", type=int, default=20)
    return parser


def main():
    args = build_parser().parse_args()
    _, _, _, report = derive_multi_roi_spectral_dataset(
        dataset_dir=args.dataset_dir,
        source_image_dir=args.source_image_dir,
        response_mat=args.response_mat,
        rois_json=args.rois_json,
        output_dir=args.output_dir,
        basis_count=args.basis_count,
        fwhm_nm=args.fwhm_nm,
        max_frames=args.max_frames,
        expected_roi_count=args.expected_roi_count,
        debug_image_limit=args.debug_image_limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
