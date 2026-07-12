from __future__ import annotations

import numpy as np


def ray_box_intersections(rays, bbox, eps=1e-8):
    rays = np.asarray(rays, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32)
    if rays.ndim != 3 or rays.shape[1:] != (2, 3):
        raise ValueError("rays must have shape [N,2,3]")
    if bbox.shape != (2, 3):
        raise ValueError("bbox must have shape [2,3]")

    origins = rays[:, 0, :]
    dirs = rays[:, 1, :]
    inv_dirs = np.full_like(dirs, np.inf, dtype=np.float32)
    np.divide(1.0, dirs, out=inv_dirs, where=np.abs(dirs) > eps)
    t0 = (bbox[0][None, :] - origins) * inv_dirs
    t1 = (bbox[1][None, :] - origins) * inv_dirs

    parallel = np.abs(dirs) <= eps
    outside_parallel = parallel & (
        (origins < bbox[0][None, :]) | (origins > bbox[1][None, :])
    )
    t_min_axis = np.minimum(t0, t1)
    t_max_axis = np.maximum(t0, t1)
    t_min_axis = np.where(parallel, -np.inf, t_min_axis)
    t_max_axis = np.where(parallel, np.inf, t_max_axis)

    near = np.max(t_min_axis, axis=1)
    far = np.min(t_max_axis, axis=1)
    hit = (far > np.maximum(near, 0.0)) & ~outside_parallel.any(axis=1)
    return np.maximum(near, 0.0).astype(np.float32), far.astype(np.float32), hit

