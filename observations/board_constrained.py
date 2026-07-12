"""Prepare ROI observations for analytic-board geometry training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.geometry import ray_box_intersections


REQUIRED_ROW_FIELDS = (
    "obs_value",
    "frame_id",
    "plane_uv",
)


def uniform_frame_split(frame_ids, heldout_fraction=0.1):
    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    if frame_ids.size == 0:
        raise ValueError("frame_ids must not be empty")
    if not 0.0 < float(heldout_fraction) < 1.0:
        raise ValueError("heldout_fraction must be between zero and one")

    unique_frames = np.unique(frame_ids)
    heldout_count = max(1, int(round(len(unique_frames) * float(heldout_fraction))))
    heldout_count = min(heldout_count, len(unique_frames) - 1)
    if heldout_count <= 0:
        raise ValueError("at least two frames are required for a heldout split")
    heldout_positions = np.linspace(
        0, len(unique_frames) - 1, num=heldout_count, dtype=np.int64
    )
    heldout_frames = unique_frames[heldout_positions]
    heldout_mask = np.isin(frame_ids, heldout_frames)
    return (
        np.flatnonzero(~heldout_mask).astype(np.int64),
        np.flatnonzero(heldout_mask).astype(np.int64),
        heldout_frames.astype(np.int64),
    )


def make_sensor_ids(unit_ids, calibration_pixel_ids):
    unit_ids = np.asarray(unit_ids, dtype=np.int64).reshape(-1)
    pixel_ids = np.asarray(calibration_pixel_ids, dtype=np.int64).reshape(-1)
    if unit_ids.shape != pixel_ids.shape:
        raise ValueError("unit and calibration pixel ids must have matching shapes")
    sensor_key = unit_ids * 400 + pixel_ids
    sensor_keys, sensor_ids = np.unique(sensor_key, return_inverse=True)
    return sensor_ids.astype(np.int64), sensor_keys.astype(np.int64)


def sensor_ids_from_arrays(arrays, observation_count):
    if "sensor_key" in arrays:
        sensor_key = np.asarray(arrays["sensor_key"], dtype=np.int64).reshape(-1)
        if sensor_key.shape != (int(observation_count),):
            raise ValueError("sensor_key must have shape [N]")
        sensor_keys, sensor_ids = np.unique(sensor_key, return_inverse=True)
        return sensor_ids.astype(np.int64), sensor_keys.astype(np.int64)
    if "unit_id" in arrays and "calibration_pixel_id" in arrays:
        return make_sensor_ids(arrays["unit_id"], arrays["calibration_pixel_id"])
    raise ValueError("observations must contain sensor_key or unit_id + calibration_pixel_id")


def estimate_object_bbox(
    plane_uv,
    margin_fraction=0.2,
    z_min=-0.045,
    z_max=-0.0002,
):
    plane_uv = np.asarray(plane_uv, dtype=np.float32)
    if plane_uv.ndim != 2 or plane_uv.shape[1] != 2:
        raise ValueError("plane_uv must have shape [N, 2]")
    finite = plane_uv[np.isfinite(plane_uv).all(axis=1)]
    if finite.size == 0:
        raise ValueError("plane_uv does not contain finite coordinates")
    if not float(z_min) < float(z_max) < 0.0:
        raise ValueError("object z bounds must satisfy z_min < z_max < 0")

    xy_min = np.percentile(finite, 1.0, axis=0)
    xy_max = np.percentile(finite, 99.0, axis=0)
    span = np.maximum(xy_max - xy_min, 1e-6)
    xy_min -= float(margin_fraction) * span
    xy_max += float(margin_fraction) * span
    return np.array(
        [[xy_min[0], xy_min[1], z_min], [xy_max[0], xy_max[1], z_max]],
        dtype=np.float32,
    )


def compute_object_intersections(rays, bbox):
    near, far, hit = ray_box_intersections(rays, bbox)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    near[~hit] = np.nan
    far[~hit] = np.nan
    return near, far, hit.astype(bool)


def _has_compact_response_basis(arrays):
    return "response_basis_table" in arrays and "response_basis_index" in arrays


def _validate_response_basis_storage(arrays, observation_count):
    if "response_basis" in arrays:
        response_basis = np.asarray(arrays["response_basis"])
        if response_basis.shape[:1] != (int(observation_count),) or response_basis.ndim != 2:
            raise ValueError("response_basis must have shape [N,K]")
        return int(response_basis.shape[1])
    if _has_compact_response_basis(arrays):
        table = np.asarray(arrays["response_basis_table"])
        index = np.asarray(arrays["response_basis_index"])
        if table.ndim != 2:
            raise ValueError("response_basis_table must have shape [M,K]")
        if index.shape != (int(observation_count),):
            raise ValueError("response_basis_index must have shape [N]")
        if index.size and (int(index.min()) < 0 or int(index.max()) >= table.shape[0]):
            raise ValueError("response_basis_index contains out-of-range values")
        return int(table.shape[1])
    raise ValueError("observations must contain response_basis or response_basis_table + response_basis_index")


def _iter_response_basis_chunks(response_basis, chunk_size=262_144):
    if isinstance(response_basis, tuple):
        table, index = response_basis
        table = np.asarray(table, dtype=np.float32)
        index = np.asarray(index, dtype=np.int64).reshape(-1)
        for start in range(0, index.size, int(chunk_size)):
            idx = index[start : start + int(chunk_size)]
            yield start, table[idx]
        return

    basis = np.asarray(response_basis, dtype=np.float32)
    for start in range(0, basis.shape[0], int(chunk_size)):
        yield start, basis[start : start + int(chunk_size)]


def response_basis_storage(arrays):
    if "response_basis" in arrays:
        return np.asarray(arrays["response_basis"], dtype=np.float32)
    return (
        np.asarray(arrays["response_basis_table"], dtype=np.float32),
        np.asarray(arrays["response_basis_index"], dtype=np.int64),
    )


def board_basis_prediction(
    response_basis,
    board_value,
    black_coeff=0.08,
    white_coeff=0.75,
):
    board_value = np.asarray(board_value, dtype=np.float32).reshape(-1)
    if isinstance(response_basis, tuple):
        row_count = np.asarray(response_basis[1]).shape[0]
    else:
        row_count = np.asarray(response_basis).shape[0]
    if row_count != board_value.shape[0]:
        raise ValueError("response basis storage and board_value must have matching rows")

    pred = np.empty((board_value.shape[0],), dtype=np.float32)
    for start, chunk in _iter_response_basis_chunks(response_basis):
        stop = start + chunk.shape[0]
        coeff = float(black_coeff) + board_value[start:stop, None] * (
            float(white_coeff) - float(black_coeff)
        )
        pred[start:stop] = np.sum(chunk * coeff, axis=1).astype(np.float32)
    return pred


def _fit_frame_affine_board_prediction(
    board_pred,
    obs_value,
    frame_id,
    min_fit_std=1e-6,
):
    board_pred = np.asarray(board_pred, dtype=np.float32).reshape(-1)
    obs_value = np.asarray(obs_value, dtype=np.float32).reshape(-1)
    frame_id = np.asarray(frame_id, dtype=np.int64).reshape(-1)
    if not (board_pred.shape == obs_value.shape == frame_id.shape):
        raise ValueError("board_pred, obs_value, and frame_id must have matching rows")

    aligned = board_pred.astype(np.float32).copy()
    unique_frames = np.unique(frame_id)
    scales = np.ones((unique_frames.size,), dtype=np.float32)
    biases = np.zeros((unique_frames.size,), dtype=np.float32)
    fit_stds = np.zeros((unique_frames.size,), dtype=np.float32)
    for i, current_frame in enumerate(unique_frames):
        rows = frame_id == int(current_frame)
        x = board_pred[rows].astype(np.float64)
        y = obs_value[rows].astype(np.float64)
        fit_stds[i] = float(np.std(x))
        if x.size < 2 or float(np.std(x)) < float(min_fit_std):
            continue

        design = np.column_stack([x, np.ones_like(x)])
        scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = np.abs(y - (scale * x + bias))
        if x.size >= 6:
            cutoff = np.percentile(residual, 75.0)
            inlier = residual <= max(float(cutoff), float(min_fit_std))
            if int(inlier.sum()) >= 2 and float(np.std(x[inlier])) >= float(min_fit_std):
                robust_design = np.column_stack([x[inlier], np.ones(int(inlier.sum()))])
                scale, bias = np.linalg.lstsq(robust_design, y[inlier], rcond=None)[0]

        if np.isfinite(scale) and np.isfinite(bias):
            aligned[rows] = (scale * x + bias).astype(np.float32)
            scales[i] = float(scale)
            biases[i] = float(bias)

    return aligned, {
        "unique_frame_ids": unique_frames.astype(np.int64),
        "board_affine_scale": scales,
        "board_affine_bias": biases,
        "board_affine_fit_std": fit_stds,
    }


def classify_board_residuals(
    obs_value,
    response_basis,
    board_value,
    frame_id,
    residual_threshold,
    min_foreground_fraction=0.02,
    black_coeff=0.08,
    white_coeff=0.75,
    fit_frame_affine=False,
):
    obs_value = np.asarray(obs_value, dtype=np.float32).reshape(-1)
    frame_id = np.asarray(frame_id, dtype=np.int64).reshape(-1)
    if obs_value.shape[0] != frame_id.shape[0]:
        raise ValueError("obs_value and frame_id must have matching rows")
    if float(residual_threshold) <= 0.0:
        raise ValueError("residual_threshold must be positive")
    if not 0.0 <= float(min_foreground_fraction) <= 1.0:
        raise ValueError("min_foreground_fraction must be in [0,1]")

    board_pred = board_basis_prediction(
        response_basis,
        board_value,
        black_coeff=black_coeff,
        white_coeff=white_coeff,
    )
    affine_result = None
    if fit_frame_affine:
        board_pred, affine_result = _fit_frame_affine_board_prediction(
            board_pred, obs_value, frame_id
        )
    residual = np.abs(obs_value - board_pred).astype(np.float32)
    foreground_mask = residual > float(residual_threshold)

    unique_frames = np.unique(frame_id)
    frame_has_object = np.zeros((unique_frames.size,), dtype=bool)
    frame_foreground_fraction = np.zeros((unique_frames.size,), dtype=np.float32)
    row_frame_has_object = np.zeros_like(foreground_mask, dtype=bool)
    for i, current_frame in enumerate(unique_frames):
        rows = frame_id == int(current_frame)
        fraction = float(np.mean(foreground_mask[rows])) if rows.any() else 0.0
        frame_foreground_fraction[i] = fraction
        has_object = fraction >= float(min_foreground_fraction)
        frame_has_object[i] = has_object
        row_frame_has_object[rows] = has_object

    background_alpha_weight = (~foreground_mask).astype(np.float32)
    object_train_weight = (foreground_mask & row_frame_has_object).astype(np.float32)
    result = {
        "board_only_pred": board_pred,
        "board_residual": residual,
        "foreground_mask": foreground_mask,
        "background_alpha_weight": background_alpha_weight,
        "object_train_weight": object_train_weight,
        "unique_frame_ids": unique_frames.astype(np.int64),
        "frame_has_object": frame_has_object,
        "row_frame_has_object": row_frame_has_object,
        "frame_foreground_fraction": frame_foreground_fraction,
    }
    if affine_result is not None:
        result.update(affine_result)
    return result


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def board_canvas_inside_mask(board_prior, plane_uv):
    if not hasattr(board_prior, "world_to_pixel") or not hasattr(board_prior, "image"):
        return np.ones((np.asarray(plane_uv).shape[0],), dtype=bool)
    pixels = board_prior.world_to_pixel(plane_uv)
    height, width = board_prior.image.shape
    return (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1)
    )


def filter_observation_rows(arrays, keep_mask, observation_count):
    keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if keep_mask.shape != (int(observation_count),):
        raise ValueError("keep_mask must match observation count")
    filtered = {}
    for key, value in arrays.items():
        value = np.asarray(value)
        if key in ("i_train", "i_val", "i_test"):
            continue
        if value.shape[:1] == (int(observation_count),):
            filtered[key] = value[keep_mask]
        else:
            filtered[key] = value
    return filtered


def prepare_board_constrained_dataset(
    observation_path,
    board_prior,
    output_dir,
    heldout_fraction=0.1,
    margin_fraction=0.2,
    z_min=-0.045,
    z_max=-0.0002,
    foreground_residual_threshold=None,
    min_foreground_fraction=0.02,
    fit_board_frame_affine=False,
    drop_outside_board=False,
    output_name="observations_board_constrained.npz",
):
    observation_path = Path(observation_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(observation_path, allow_pickle=True) as source:
        missing = [name for name in REQUIRED_ROW_FIELDS if name not in source.files]
        if missing:
            raise ValueError(
                f"{observation_path} missing required fields: {', '.join(missing)}"
            )
        arrays = {name: np.asarray(source[name]) for name in source.files}

    source_observation_count = int(arrays["obs_value"].shape[0])
    observation_count = source_observation_count
    for name in REQUIRED_ROW_FIELDS:
        if arrays[name].shape[0] != observation_count:
            raise ValueError(f"{name} does not match observation count")
    basis_count = _validate_response_basis_storage(arrays, observation_count)
    for name in ("obs_value", "plane_uv"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    if "rays" in arrays and not np.isfinite(arrays["rays"]).all():
        raise ValueError("rays contains non-finite values")
    if "response_basis" in arrays and not np.isfinite(arrays["response_basis"]).all():
        raise ValueError("response_basis contains non-finite values")
    if _has_compact_response_basis(arrays) and not np.isfinite(arrays["response_basis_table"]).all():
        raise ValueError("response_basis_table contains non-finite values")

    dropped_outside_board_count = 0
    if drop_outside_board:
        inside_board = board_canvas_inside_mask(board_prior, arrays["plane_uv"])
        dropped_outside_board_count = int((~inside_board).sum())
        if dropped_outside_board_count > 0:
            arrays = filter_observation_rows(arrays, inside_board, observation_count)
            observation_count = int(arrays["obs_value"].shape[0])
            if observation_count == 0:
                raise ValueError("all observations were outside the board canvas")

    sensor_id, sensor_keys = sensor_ids_from_arrays(arrays, observation_count)
    board_value = np.asarray(
        board_prior.sample(arrays["plane_uv"]), dtype=np.float32
    )
    if board_value.shape != (observation_count,) or not np.isfinite(board_value).all():
        raise ValueError("board prior returned invalid observation values")

    residual_report = None
    if foreground_residual_threshold is not None:
        residual_result = classify_board_residuals(
            arrays["obs_value"],
            response_basis_storage(arrays),
            board_value,
            arrays["frame_id"],
            residual_threshold=float(foreground_residual_threshold),
            min_foreground_fraction=float(min_foreground_fraction),
            fit_frame_affine=bool(fit_board_frame_affine),
        )
        arrays.update(
            board_only_pred=residual_result["board_only_pred"],
            board_residual=residual_result["board_residual"],
            foreground_mask=residual_result["foreground_mask"],
            background_alpha_weight=residual_result["background_alpha_weight"],
            object_train_weight=residual_result["object_train_weight"],
            row_frame_has_object=residual_result["row_frame_has_object"],
            foreground_frame_ids=residual_result["unique_frame_ids"][
                residual_result["frame_has_object"]
            ],
            background_only_frame_ids=residual_result["unique_frame_ids"][
                ~residual_result["frame_has_object"]
            ],
        )
        if "board_affine_scale" in residual_result:
            arrays.update(
                board_affine_frame_ids=residual_result["unique_frame_ids"],
                board_affine_scale=residual_result["board_affine_scale"],
                board_affine_bias=residual_result["board_affine_bias"],
                board_affine_fit_std=residual_result["board_affine_fit_std"],
            )
        bbox_source_mask = residual_result["object_train_weight"] > 0.0
        residual_report = {
            "foreground_residual_threshold": float(foreground_residual_threshold),
            "min_foreground_fraction": float(min_foreground_fraction),
            "fit_board_frame_affine": bool(fit_board_frame_affine),
            "foreground_observation_count": int(residual_result["foreground_mask"].sum()),
            "background_alpha_count": int(
                (residual_result["background_alpha_weight"] > 0.0).sum()
            ),
            "foreground_frame_count": int(residual_result["frame_has_object"].sum()),
            "background_only_frame_count": int((~residual_result["frame_has_object"]).sum()),
            "board_residual_percentiles": np.percentile(
                residual_result["board_residual"], [50.0, 75.0, 90.0, 95.0, 99.0]
            ).astype(float).tolist(),
        }
        if "board_affine_scale" in residual_result:
            residual_report["board_affine_scale_percentiles"] = np.percentile(
                residual_result["board_affine_scale"], [1.0, 50.0, 99.0]
            ).astype(float).tolist()
            residual_report["board_affine_bias_percentiles"] = np.percentile(
                residual_result["board_affine_bias"], [1.0, 50.0, 99.0]
            ).astype(float).tolist()
    else:
        bbox_source_mask = np.isfinite(arrays["plane_uv"]).all(axis=1)
        arrays.update(
            foreground_mask=np.ones((observation_count,), dtype=bool),
            background_alpha_weight=np.zeros((observation_count,), dtype=np.float32),
            object_train_weight=np.ones((observation_count,), dtype=np.float32),
            row_frame_has_object=np.ones((observation_count,), dtype=bool),
            foreground_frame_ids=np.unique(arrays["frame_id"]).astype(np.int64),
            background_only_frame_ids=np.asarray([], dtype=np.int64),
        )

    if not np.any(bbox_source_mask):
        bbox_source_mask = np.isfinite(arrays["plane_uv"]).all(axis=1)
    bbox = estimate_object_bbox(
        arrays["plane_uv"][bbox_source_mask],
        margin_fraction=margin_fraction,
        z_min=z_min,
        z_max=z_max,
    )
    if "rays" in arrays:
        object_near, object_far, object_hit = compute_object_intersections(
            arrays["rays"], bbox
        )
    else:
        object_near = np.full((observation_count,), np.nan, dtype=np.float32)
        object_far = np.full((observation_count,), np.nan, dtype=np.float32)
        object_hit = np.ones((observation_count,), dtype=bool)
    i_train, i_heldout, heldout_frames = uniform_frame_split(
        arrays["frame_id"], heldout_fraction=heldout_fraction
    )

    arrays.update(
        board_value=board_value,
        sensor_id=sensor_id,
        sensor_keys=sensor_keys,
        object_bbox=bbox,
        object_near=object_near,
        object_far=object_far,
        object_hit=object_hit,
        i_train=i_train,
        i_heldout=i_heldout,
        heldout_frame_ids=heldout_frames,
    )
    arrays.pop("i_val", None)
    arrays.pop("i_test", None)

    output_path = output_dir / str(output_name)
    np.savez_compressed(output_path, **arrays)
    hit_near = object_near[object_hit]
    hit_far = object_far[object_hit]
    report = {
        "source_observation_path": str(observation_path),
        "output_observation_path": str(output_path),
        "source_observation_count": int(source_observation_count),
        "observation_count": observation_count,
        "dropped_outside_board_count": int(dropped_outside_board_count),
        "drop_outside_board": bool(drop_outside_board),
        "frame_count": int(np.unique(arrays["frame_id"]).size),
        "train_frame_count": int(
            np.unique(arrays["frame_id"][i_train]).size
        ),
        "heldout_frame_count": int(heldout_frames.size),
        "heldout_frame_ids": heldout_frames.tolist(),
        "sensor_count": int(sensor_keys.size),
        "basis_count": int(basis_count),
        "object_hit_count": int(object_hit.sum()),
        "board_only_count": int((~object_hit).sum()),
        "object_bbox": bbox.astype(float).tolist(),
        "foreground_frame_count": int(np.unique(arrays["foreground_frame_ids"]).size),
        "background_only_frame_count": int(
            np.unique(arrays["background_only_frame_ids"]).size
        ),
        "foreground_observation_count": int(
            np.asarray(arrays["foreground_mask"], dtype=bool).sum()
        ),
        "background_alpha_count": int(
            (np.asarray(arrays["background_alpha_weight"], dtype=np.float32) > 0.0).sum()
        ),
        "residual_classification": residual_report,
        "object_near_range": (
            [float(hit_near.min()), float(hit_near.max())]
            if hit_near.size
            else None
        ),
        "object_far_range": (
            [float(hit_far.min()), float(hit_far.max())] if hit_far.size else None
        ),
    }
    _write_json(output_dir / "board_dataset_report.json", report)
    return output_path, report
