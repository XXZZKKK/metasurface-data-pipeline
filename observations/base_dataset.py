"""Build a cube ROI metasurface observation-list dataset with AprilGrid poses."""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

from ..calibration.aprilgrid import (
    calibrate_from_detections,
    detect_aprilgrid,
    draw_detection_overlay,
    draw_pose_overlay,
    load_aprilgrid_spec,
    select_best_intrinsic_candidate,
    solve_pose_from_detection,
)
from ..core.real_metasurface import (
    MATLAB_ROI,
    build_real_observation_arrays,
    crop_metasurface_roi,
    load_center2x2_response_from_mat,
)
from ..config.paths import default_output_dir


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_RESPONSE_MAT = PROJECT_ROOT / "metasurface_data" / "shuju" / "A_matrix_normalized.mat"


@dataclass(frozen=True)
class IntrinsicCandidate:
    name: str
    image_dir: pathlib.Path
    aprilgrid_json: pathlib.Path


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cube ROI AprilTag experiment requires opencv-python") from exc
    return cv2


def _write_json(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_image(path, image):
    cv2 = _require_cv2()
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"failed to write image: {path}")


def _list_images(image_dir, max_frames=None, frame_stride=1):
    image_dir = pathlib.Path(image_dir)
    if not image_dir.is_dir():
        raise ValueError(f"image directory does not exist: {image_dir}")
    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames > 0:
        images = images[:max_frames]
    images = images[::frame_stride]
    if not images:
        raise ValueError(f"no image files found in {image_dir}")
    return images


def _even_subset(paths, max_frames):
    paths = list(paths)
    if max_frames is None or max_frames <= 0 or len(paths) <= max_frames:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_frames, dtype=np.int64)
    return [paths[int(i)] for i in idx]


def parse_intrinsic_candidate(text: str) -> IntrinsicCandidate:
    if "=" not in text or "," not in text:
        raise ValueError("intrinsic candidate must be formatted as name=image_dir,aprilgrid_json")
    name, rest = text.split("=", 1)
    image_dir, aprilgrid_json = rest.split(",", 1)
    name = name.strip()
    if not name:
        raise ValueError("intrinsic candidate name cannot be empty")
    return IntrinsicCandidate(name=name, image_dir=pathlib.Path(image_dir), aprilgrid_json=pathlib.Path(aprilgrid_json))


def resolve_cube_frame_stride(frame_stride: int, use_half_frames: bool) -> int:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if not use_half_frames:
        return int(frame_stride)
    if frame_stride != 1:
        raise ValueError("--use-half-frames cannot be combined with --frame-stride other than the default 1")
    return 2


def evaluate_intrinsic_candidate(candidate: IntrinsicCandidate, args, output_dir):
    cv2 = _require_cv2()
    spec = load_aprilgrid_spec(candidate.aprilgrid_json)
    image_paths = _even_subset(
        _list_images(candidate.image_dir, max_frames=args.max_candidate_images, frame_stride=args.candidate_frame_stride),
        args.calibration_max_frames,
    )
    detections = []
    frame_reports = []
    image_size = None
    candidate_dir = pathlib.Path(output_dir) / "intrinsics_candidates" / candidate.name

    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            frame_reports.append({"image": str(image_path), "status": "read_failed"})
            continue
        image_size = (image.shape[1], image.shape[0])
        detection = detect_aprilgrid(image, spec)
        detections.append(detection)
        frame_reports.append(
            {
                "image": str(image_path),
                "status": "ok",
                "detected_markers": int(detection.marker_count),
                "detected_corners": int(detection.corner_count),
            }
        )
        if args.save_debug_images and index < args.debug_image_limit:
            _write_image(candidate_dir / "detection_overlay" / f"{index:04d}.png", draw_detection_overlay(image, detection))

    if image_size is None:
        report = {
            "name": candidate.name,
            "status": "failed",
            "reason": "no_readable_images",
            "image_dir": str(candidate.image_dir),
            "aprilgrid_json": str(candidate.aprilgrid_json),
            "frames": frame_reports,
        }
        _write_json(candidate_dir / "candidate_report.json", report)
        return None, None, report

    camera_matrix, dist_coeffs, report = calibrate_from_detections(
        spec,
        detections,
        image_size=image_size,
        min_markers=args.min_markers,
    )
    report.update(
        {
            "name": candidate.name,
            "image_dir": str(candidate.image_dir),
            "aprilgrid_json": str(candidate.aprilgrid_json),
            "image_count": int(len(image_paths)),
            "frames": frame_reports,
        }
    )
    _write_json(candidate_dir / "candidate_report.json", report)
    return camera_matrix, dist_coeffs, report


def _selected_intrinsics_from_candidates(candidate_results):
    reports = [result["report"] for result in candidate_results]
    selected_report = select_best_intrinsic_candidate(reports)
    for result in candidate_results:
        if result["report"].get("name") == selected_report["name"]:
            return result["camera_matrix"], result["dist_coeffs"], selected_report
    raise RuntimeError("selected intrinsic report did not match any candidate result")


def build_cube_dataset(args):
    cv2 = _require_cv2()
    args.frame_stride = resolve_cube_frame_stride(args.frame_stride, args.use_half_frames)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    response_table, wavelengths = load_center2x2_response_from_mat(args.response_mat)
    np.savez_compressed(
        output_dir / "response_tables_center2x2.npz",
        response_table=np.asarray(response_table, dtype=np.float32),
        wavelengths=np.asarray(wavelengths, dtype=np.float32),
    )

    candidates = [parse_intrinsic_candidate(text) for text in args.intrinsic_candidate]
    if not candidates:
        raise ValueError("at least one --intrinsic-candidate is required")

    candidate_results = []
    for candidate in candidates:
        print(f"[intrinsics] evaluating {candidate.name}: {candidate.image_dir}")
        camera_matrix, dist_coeffs, report = evaluate_intrinsic_candidate(candidate, args, output_dir)
        candidate_results.append(
            {
                "name": candidate.name,
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
                "report": report,
            }
        )

    camera_matrix, dist_coeffs, selected_intrinsics = _selected_intrinsics_from_candidates(candidate_results)
    _write_json(output_dir / "intrinsics_candidates" / "selected_intrinsics.json", selected_intrinsics)

    cube_spec = load_aprilgrid_spec(args.cube_aprilgrid_json)
    cube_paths = _list_images(args.cube_image_dir, max_frames=args.max_frames, frame_stride=args.frame_stride)
    accepted_frame_ids = []
    accepted_image_names = []
    accepted_poses_c2w = []
    accepted_roi_images = []
    cube_frame_reports = []

    for frame_id, image_path in enumerate(cube_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            cube_frame_reports.append({"image": str(image_path), "status": "read_failed"})
            continue
        detection = detect_aprilgrid(image, cube_spec)
        c2w, pose_report = solve_pose_from_detection(
            cube_spec,
            detection,
            camera_matrix,
            dist_coeffs,
            min_markers=args.min_markers,
            reprojection_threshold_px=args.reprojection_threshold_px,
        )
        pose_report.update({"image": str(image_path), "frame_id": int(frame_id)})
        cube_frame_reports.append(pose_report)

        if args.save_debug_images and frame_id < args.debug_image_limit:
            _write_image(output_dir / "apriltag_overlay" / f"frame_{frame_id:04d}.png", draw_detection_overlay(image, detection))
            if pose_report.get("status") == "ok":
                _write_image(
                    output_dir / "pose_overlay" / f"frame_{frame_id:04d}.png",
                    draw_pose_overlay(
                        image,
                        cube_spec,
                        detection,
                        camera_matrix,
                        dist_coeffs,
                        np.asarray(pose_report["rvec"], dtype=np.float64),
                        np.asarray(pose_report["tvec"], dtype=np.float64),
                    ),
                )

        if c2w is None:
            continue

        roi = crop_metasurface_roi(image, MATLAB_ROI)
        if args.save_debug_images and len(accepted_roi_images) < args.debug_image_limit:
            _write_image(output_dir / "roi_gray" / f"frame_{frame_id:04d}.png", roi)
        accepted_frame_ids.append(int(frame_id))
        accepted_image_names.append(str(image_path))
        accepted_poses_c2w.append(c2w)
        accepted_roi_images.append(roi.astype(np.float32))

        if args.progress_every > 0 and len(cube_frame_reports) % args.progress_every == 0:
            print(f"[cube] processed {len(cube_frame_reports)} frames, accepted {len(accepted_frame_ids)} poses")

    if len(accepted_frame_ids) < args.min_accepted_frames:
        raise ValueError(
            f"accepted only {len(accepted_frame_ids)} cube poses, below --min-accepted-frames={args.min_accepted_frames}"
        )

    poses_c2w = np.asarray(accepted_poses_c2w, dtype=np.float32)
    observations = build_real_observation_arrays(
        frame_ids=np.asarray(accepted_frame_ids, dtype=np.int64),
        roi_images=np.asarray(accepted_roi_images, dtype=np.float32),
        poses_c2w=poses_c2w,
        intrinsics=np.asarray(camera_matrix, dtype=np.float32),
        response_table=np.asarray(response_table, dtype=np.float32),
        wavelengths=np.asarray(wavelengths, dtype=np.float32),
        roi_x0=MATLAB_ROI.x0,
        roi_y0=MATLAB_ROI.y0,
        near=float(args.near),
        far=float(args.far),
    )
    np.savez_compressed(output_dir / "observations.npz", **observations)
    np.savez_compressed(
        output_dir / "poses_bounds.npz",
        poses_c2w=poses_c2w,
        frame_ids=np.asarray(accepted_frame_ids, dtype=np.int64),
        image_names=np.asarray(accepted_image_names),
        K=np.asarray(camera_matrix, dtype=np.float32),
        dist_coeffs=np.asarray(dist_coeffs, dtype=np.float32).reshape(-1),
        bounds=np.asarray([args.near, args.far], dtype=np.float32),
        roi=np.asarray([MATLAB_ROI.x0, MATLAB_ROI.y0, MATLAB_ROI.size, MATLAB_ROI.size], dtype=np.int32),
    )

    report = {
        "cube_image_dir": str(args.cube_image_dir),
        "cube_aprilgrid_json": str(args.cube_aprilgrid_json),
        "response_mat": str(args.response_mat),
        "selected_intrinsics": selected_intrinsics,
        "intrinsic_candidates": [result["report"] for result in candidate_results],
        "cube_frames": cube_frame_reports,
        "accepted_pose_count": int(len(accepted_frame_ids)),
        "processed_cube_frame_count": int(len(cube_paths)),
        "cube_frame_stride": int(args.frame_stride),
        "use_half_frames": bool(args.use_half_frames),
        "candidate_frame_stride": int(args.candidate_frame_stride),
        "observation_count": int(len(observations["obs_value"])),
        "roi": {
            "matlab_x": int(MATLAB_ROI.matlab_x),
            "matlab_y": int(MATLAB_ROI.matlab_y),
            "x0": int(MATLAB_ROI.x0),
            "y0": int(MATLAB_ROI.y0),
            "size": int(MATLAB_ROI.size),
            "unit_size": int(MATLAB_ROI.unit_size),
        },
    }
    _write_json(output_dir / "calibration_report.json", report)
    _write_json(
        output_dir / "metadata.json",
        {
            "dataset_type": "real_metasurface_cube_roi_apriltag",
            "image_dir": str(args.cube_image_dir),
            "image_count": int(len(cube_paths)),
            "accepted_pose_count": int(len(accepted_frame_ids)),
            "observation_count": int(len(observations["obs_value"])),
            "response_path": str(output_dir / "response_tables_center2x2.npz"),
            "observation_path": str(output_dir / "observations.npz"),
            "poses_bounds_path": str(output_dir / "poses_bounds.npz"),
            "calibration_report_path": str(output_dir / "calibration_report.json"),
            "selected_intrinsics_name": selected_intrinsics.get("name"),
        },
    )
    print(f"[done] wrote cube ROI AprilTag dataset to {output_dir}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-image-dir", required=True, type=pathlib.Path)
    parser.add_argument("--cube-aprilgrid-json", required=True, type=pathlib.Path)
    parser.add_argument("--intrinsic-candidate", action="append", default=[])
    parser.add_argument("--output-dir", default=default_output_dir("base_aprilgrid_dataset"), type=pathlib.Path)
    parser.add_argument("--response-mat", default=DEFAULT_RESPONSE_MAT, type=pathlib.Path)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--use-half-frames",
        action="store_true",
        help="Use every other cube frame for dataset/training input; equivalent to --frame-stride 2.",
    )
    parser.add_argument("--max-candidate-images", type=int, default=None)
    parser.add_argument("--candidate-frame-stride", type=int, default=1)
    parser.add_argument("--calibration-max-frames", type=int, default=80)
    parser.add_argument("--min-markers", type=int, default=8)
    parser.add_argument("--min-accepted-frames", type=int, default=8)
    parser.add_argument("--reprojection-threshold-px", type=float, default=5.0)
    parser.add_argument("--near", type=float, default=0.0)
    parser.add_argument("--far", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--debug-image-limit", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    build_cube_dataset(args)


if __name__ == "__main__":
    main()
