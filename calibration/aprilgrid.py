"""AprilGrid detection and calibration helpers for real metasurface captures."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AprilGridMarker:
    marker_id: int
    corners_m: np.ndarray


@dataclass(frozen=True)
class AprilGridSpec:
    json_path: pathlib.Path
    dictionary_name: str
    markers: dict[int, AprilGridMarker]


@dataclass(frozen=True)
class AprilGridDetection:
    ids: np.ndarray
    corners: np.ndarray

    @property
    def marker_count(self) -> int:
        return int(len(self.ids))

    @property
    def corner_count(self) -> int:
        return int(len(self.ids) * 4)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("AprilGrid processing requires opencv-python with cv2.aruco") from exc
    if not hasattr(cv2, "aruco"):
        raise ImportError("AprilGrid processing requires cv2.aruco")
    return cv2


def load_aprilgrid_spec(json_path) -> AprilGridSpec:
    path = pathlib.Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [key for key in ("dictionary", "markers") if key not in data]
    if missing:
        raise ValueError(f"AprilGrid JSON {path} is missing required field(s): {missing}")

    markers: dict[int, AprilGridMarker] = {}
    for raw in data["markers"]:
        if "id" not in raw or "corners_m" not in raw:
            raise ValueError(f"marker entry in {path} must contain id and corners_m")
        marker_id = int(raw["id"])
        if marker_id in markers:
            raise ValueError(f"duplicate marker id {marker_id} in {path}")
        corners_m = np.asarray(raw["corners_m"], dtype=np.float32)
        if corners_m.shape != (4, 3):
            raise ValueError(f"marker {marker_id} corners_m must have shape [4,3], got {corners_m.shape}")
        if not np.isfinite(corners_m).all():
            raise ValueError(f"marker {marker_id} corners_m contains non-finite values")
        markers[marker_id] = AprilGridMarker(marker_id=marker_id, corners_m=corners_m)

    if not markers:
        raise ValueError(f"AprilGrid JSON {path} does not contain any markers")
    return AprilGridSpec(json_path=path, dictionary_name=str(data["dictionary"]), markers=markers)


def aruco_dictionary_from_name(dictionary_name: str):
    cv2 = _require_cv2()
    aruco = cv2.aruco
    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"OpenCV aruco has no dictionary named {dictionary_name}")
    return aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))


def _detector_parameters():
    cv2 = _require_cv2()
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()
    if hasattr(aruco, "CORNER_REFINE_APRILTAG"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    elif hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return params


def _as_gray(image):
    cv2 = _require_cv2()
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"image must be 2D grayscale or 3D BGR/RGB, got shape {arr.shape}")


def detect_aprilgrid(image, spec: AprilGridSpec) -> AprilGridDetection:
    """Detect markers from ``spec`` and return sorted ids with 4 image corners each."""
    cv2 = _require_cv2()
    aruco = cv2.aruco
    gray = _as_gray(image)
    dictionary = aruco_dictionary_from_name(spec.dictionary_name)
    params = _detector_parameters()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _rejected = detector.detectMarkers(gray)
    elif hasattr(aruco, "detectMarkers"):
        corners, ids, _rejected = aruco.detectMarkers(gray, dictionary, parameters=params)
    else:
        raise ImportError("cv2.aruco has neither ArucoDetector nor detectMarkers")

    if ids is None or len(ids) == 0:
        return AprilGridDetection(
            ids=np.zeros((0,), dtype=np.int32),
            corners=np.zeros((0, 4, 2), dtype=np.float32),
        )

    ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
    kept_ids = []
    kept_corners = []
    for marker_id, marker_corners in zip(ids_flat, corners):
        marker_id = int(marker_id)
        if marker_id not in spec.markers:
            continue
        kept_ids.append(marker_id)
        kept_corners.append(np.asarray(marker_corners, dtype=np.float32).reshape(4, 2))

    if not kept_ids:
        return AprilGridDetection(
            ids=np.zeros((0,), dtype=np.int32),
            corners=np.zeros((0, 4, 2), dtype=np.float32),
        )

    order = np.argsort(np.asarray(kept_ids, dtype=np.int32))
    ids_arr = np.asarray(kept_ids, dtype=np.int32)[order]
    corners_arr = np.asarray(kept_corners, dtype=np.float32)[order]
    return AprilGridDetection(ids=ids_arr, corners=corners_arr)


def object_image_points_from_detection(spec: AprilGridSpec, detection: AprilGridDetection):
    if detection.marker_count == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    object_points = []
    image_points = []
    for marker_id, corners_px in zip(detection.ids, detection.corners):
        marker_id = int(marker_id)
        if marker_id not in spec.markers:
            raise ValueError(f"detected marker id {marker_id} is not present in {spec.json_path}")
        object_points.append(spec.markers[marker_id].corners_m)
        image_points.append(np.asarray(corners_px, dtype=np.float32).reshape(4, 2))
    return np.concatenate(object_points, axis=0).astype(np.float32), np.concatenate(image_points, axis=0).astype(np.float32)


def rvec_tvec_to_c2w(rvec, tvec):
    cv2 = _require_cv2()
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    r_wc, _ = cv2.Rodrigues(rvec)
    r_cw = r_wc.T
    t_cw = -r_cw @ tvec
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = r_cw.astype(np.float32)
    c2w[:3, 3] = t_cw.reshape(3).astype(np.float32)
    return c2w


def reprojection_rmse(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    cv2 = _require_cv2()
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    projected = projected.reshape(-1, 2)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    err = projected - image_points
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def solve_pose_from_detection(
    spec: AprilGridSpec,
    detection: AprilGridDetection,
    camera_matrix,
    dist_coeffs,
    min_markers: int = 8,
    reprojection_threshold_px: float = 5.0,
):
    cv2 = _require_cv2()
    if detection.marker_count < min_markers:
        return None, {
            "status": "insufficient_markers",
            "marker_count": int(detection.marker_count),
            "corner_count": int(detection.corner_count),
            "min_markers": int(min_markers),
        }

    object_points, image_points = object_image_points_from_detection(spec, detection)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, {
            "status": "solvepnp_failed",
            "marker_count": int(detection.marker_count),
            "corner_count": int(detection.corner_count),
        }

    rmse = reprojection_rmse(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
    report = {
        "status": "ok" if rmse <= reprojection_threshold_px else "reprojection_rejected",
        "marker_count": int(detection.marker_count),
        "corner_count": int(detection.corner_count),
        "reprojection_rmse_px": rmse,
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "rvec": np.asarray(rvec).reshape(3).astype(float).tolist(),
        "tvec": np.asarray(tvec).reshape(3).astype(float).tolist(),
    }
    if report["status"] != "ok":
        return None, report
    return rvec_tvec_to_c2w(rvec, tvec), report


def calibrate_from_detections(
    spec: AprilGridSpec,
    detections: list[AprilGridDetection],
    image_size,
    min_markers: int = 8,
):
    cv2 = _require_cv2()
    obj_points = []
    img_points = []
    marker_counts = []
    for detection in detections:
        if detection.marker_count < min_markers:
            continue
        obj, img = object_image_points_from_detection(spec, detection)
        obj_points.append(obj)
        img_points.append(img)
        marker_counts.append(int(detection.marker_count))

    if not obj_points:
        return None, None, {
            "status": "failed",
            "reason": "no_frames_with_enough_markers",
            "valid_frame_count": 0,
            "min_markers": int(min_markers),
        }

    width, height = int(image_size[0]), int(image_size[1])
    rmse, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        (width, height),
        None,
        None,
    )
    report = {
        "status": "ok",
        "valid_frame_count": int(len(obj_points)),
        "calibration_rmse_px": float(rmse),
        "mean_markers_per_frame": float(np.mean(marker_counts)),
        "min_markers": int(min_markers),
        "image_size": [width, height],
        "camera_matrix": np.asarray(camera_matrix, dtype=float).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
    }
    return camera_matrix.astype(np.float32), dist_coeffs.astype(np.float32), report


def select_best_intrinsic_candidate(reports: list[dict]):
    ok_reports = [report for report in reports if report.get("status") == "ok"]
    if not ok_reports:
        raise ValueError("no valid intrinsic candidates; all calibration attempts failed")
    return sorted(
        ok_reports,
        key=lambda report: (
            float(report.get("calibration_rmse_px", np.inf)),
            -int(report.get("valid_frame_count", 0)),
            -float(report.get("mean_markers_per_frame", 0.0)),
            str(report.get("name", "")),
        ),
    )[0]


def draw_detection_overlay(image, detection: AprilGridDetection):
    cv2 = _require_cv2()
    overlay = np.asarray(image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for marker_id, corners in zip(detection.ids, detection.corners):
        pts = np.asarray(corners, dtype=np.int32).reshape(4, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            overlay,
            str(int(marker_id)),
            tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def draw_pose_overlay(image, spec: AprilGridSpec, detection: AprilGridDetection, camera_matrix, dist_coeffs, rvec, tvec):
    cv2 = _require_cv2()
    overlay = draw_detection_overlay(image, detection)
    object_points, _ = object_image_points_from_detection(spec, detection)
    projected, _ = cv2.projectPoints(
        object_points,
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    for pt in projected.reshape(-1, 2):
        cv2.circle(overlay, tuple(np.round(pt).astype(int)), 2, (255, 0, 0), -1)
    return overlay
