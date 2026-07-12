"""Spectral-basis helpers for metasurface scalar-response training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BasisS0QUSpec:
    wavelengths: np.ndarray
    centers_nm: np.ndarray
    fwhm_nm: float
    basis: np.ndarray


def _is_tensor(value):
    return hasattr(value, "dim") and hasattr(value, "to")


def gaussian_sigma_from_fwhm(fwhm_nm):
    if float(fwhm_nm) <= 0.0:
        raise ValueError(f"fwhm_nm must be positive, got {fwhm_nm}")
    return float(fwhm_nm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def build_gaussian_basis(wavelengths, centers_nm, fwhm_nm=20.0):
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    centers_nm = np.asarray(centers_nm, dtype=np.float32).reshape(-1)
    sigma_nm = gaussian_sigma_from_fwhm(float(fwhm_nm))
    basis = np.exp(-0.5 * ((wavelengths[None, :] - centers_nm[:, None]) / sigma_nm) ** 2).astype(np.float32)
    peaks = np.maximum(basis.max(axis=1, keepdims=True), 1e-12)
    return (basis / peaks).astype(np.float32)


def load_basis_spec_from_response_npz(path):
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        required = ("wavelengths", "spectral_centers_nm", "fwhm_nm")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{path} is missing required basis metadata: {', '.join(missing)}")
        wavelengths = np.asarray(data["wavelengths"], dtype=np.float32).reshape(-1)
        centers = np.asarray(data["spectral_centers_nm"], dtype=np.float32).reshape(-1)
        fwhm_nm = float(np.asarray(data["fwhm_nm"]).item())

    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("wavelengths must be a non-empty 1D array")
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("spectral_centers_nm must be a non-empty 1D array")
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be strictly increasing")
    basis = build_gaussian_basis(wavelengths, centers, fwhm_nm=fwhm_nm)
    return BasisS0QUSpec(wavelengths=wavelengths, centers_nm=centers, fwhm_nm=fwhm_nm, basis=basis)


def _basis_like(spec, reference):
    if _is_tensor(reference):
        import torch

        return torch.as_tensor(spec.basis, dtype=reference.dtype, device=reference.device)
    return np.asarray(spec.basis, dtype=np.float32)


def response_basis_projection(response, basis, eps=1e-12):
    """Return normalize(response) @ basis.T for [N,L] response and [K,L] basis."""
    if _is_tensor(response):
        import torch

        basis_t = basis if _is_tensor(basis) else torch.as_tensor(basis, dtype=response.dtype, device=response.device)
        weights = torch.clamp(response, min=0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        return (weights / denom) @ basis_t.transpose(-1, -2)

    response = np.asarray(response, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    weights = np.maximum(response, 0.0)
    denom = np.maximum(weights.sum(axis=-1, keepdims=True), eps)
    return ((weights / denom) @ basis.T).astype(np.float32)


def decode_basis_raw(raw, basis_count):
    """Decode raw basis output into bounded S0 coefficients and q/u ratios."""
    if _is_tensor(raw):
        import torch

        coeffs = torch.sigmoid(raw[..., :basis_count])
        dolp = torch.sigmoid(raw[..., basis_count : basis_count + 1])
        theta = raw[..., basis_count + 1 : basis_count + 2]
        q = dolp * torch.cos(theta)
        u = dolp * torch.sin(theta)
        return {"coeffs": coeffs, "dolp": dolp, "theta": theta, "q": q, "u": u}

    raw = np.asarray(raw, dtype=np.float32)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    coeffs = sigmoid(raw[..., :basis_count]).astype(np.float32)
    dolp = sigmoid(raw[..., basis_count : basis_count + 1]).astype(np.float32)
    theta = raw[..., basis_count + 1 : basis_count + 2]
    q = dolp * np.cos(theta)
    u = dolp * np.sin(theta)
    return {"coeffs": coeffs, "dolp": dolp, "theta": theta, "q": q.astype(np.float32), "u": u.astype(np.float32)}


def basis_values_at_wavelength(wavelength, spec):
    """Evaluate Gaussian basis curves at query wavelengths."""
    sigma_nm = gaussian_sigma_from_fwhm(spec.fwhm_nm)
    if _is_tensor(wavelength):
        import torch

        centers = torch.as_tensor(spec.centers_nm, dtype=wavelength.dtype, device=wavelength.device)
        waves = wavelength.reshape(-1, 1)
        return torch.exp(-0.5 * ((waves - centers.reshape(1, -1)) / sigma_nm) ** 2)

    waves = np.asarray(wavelength, dtype=np.float32).reshape(-1, 1)
    centers = np.asarray(spec.centers_nm, dtype=np.float32).reshape(1, -1)
    return np.exp(-0.5 * ((waves - centers) / sigma_nm) ** 2).astype(np.float32)


def basis_raw2outputs(raw, z_vals, rays_d, wavelength, spec, raw_noise_std=0.0, white_bkgd=False, pytest=False):
    """Volume-render basis coefficients and synthesize Stokes at query wavelengths."""
    import torch
    import torch.nn.functional as F

    basis_count = int(spec.basis.shape[0])
    # ReLU can permanently freeze a basis model when all initial density logits
    # are slightly negative: opacity becomes exactly zero and density receives no
    # gradient. Softplus keeps density nonnegative while preserving gradients.
    raw2alpha = lambda raw_alpha, dists: 1.0 - torch.exp(-F.softplus(raw_alpha) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    if dists.shape[-1] == 0:
        terminal_dist = torch.ones_like(z_vals[..., :1])
    else:
        terminal_dist = dists[..., -1:]
    dists = torch.cat([dists, terminal_dist], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    decoded = decode_basis_raw(raw, basis_count)
    density = raw[..., basis_count + 2]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn_like(density) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = torch.as_tensor(
                np.random.rand(*list(density.shape)) * raw_noise_std,
                dtype=density.dtype,
                device=density.device,
            )

    alpha = raw2alpha(density + noise, dists)
    weights = alpha * torch.cumprod(
        torch.cat(
            [
                torch.ones((alpha.shape[0], 1), dtype=alpha.dtype, device=alpha.device),
                1.0 - alpha + 1e-10,
            ],
            -1,
        ),
        -1,
    )[:, :-1]

    coeffs = decoded["coeffs"]
    q = decoded["q"]
    u = decoded["u"]
    basis_C = torch.sum(weights[..., None] * coeffs, dim=-2)
    basis_QC = torch.sum(weights[..., None] * q * coeffs, dim=-2)
    basis_UC = torch.sum(weights[..., None] * u * coeffs, dim=-2)

    basis_lambda = basis_values_at_wavelength(wavelength, spec).to(raw.device)
    s0 = torch.sum(basis_C * basis_lambda, dim=-1, keepdim=True)
    s1 = torch.sum(basis_QC * basis_lambda, dim=-1, keepdim=True)
    s2 = torch.sum(basis_UC * basis_lambda, dim=-1, keepdim=True)
    s3 = torch.zeros_like(s0)
    stoke_map = torch.cat([s0, s1, s2, s3], dim=-1)

    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

    if white_bkgd:
        stoke_map_tmp = torch.zeros_like(stoke_map)
        stoke_map_tmp[..., 0:1] = stoke_map[..., 0:1] + (1.0 - acc_map[..., None])
        stoke_map_tmp[..., 1:] = stoke_map[..., 1:] * acc_map[..., None]
        stoke_map = stoke_map_tmp

    basis_maps = {"C": basis_C, "QC": basis_QC, "UC": basis_UC}
    return stoke_map, disp_map, acc_map, weights, depth_map, basis_maps


def project_basis_observations(coeff_maps, response, analyzer, obs_type, spec):
    basis = _basis_like(spec, response)
    rB = response_basis_projection(response, basis)
    return project_basis_observations_from_projection(
        coeff_maps,
        rB,
        analyzer,
        obs_type,
    )


def project_basis_observations_from_projection(coeff_maps, response_basis, analyzer, obs_type):
    """Project rendered basis maps using a precomputed normalize(response) @ B.T."""
    rB = response_basis
    C = coeff_maps["C"]
    QC = coeff_maps["QC"]
    UC = coeff_maps["UC"]
    s0 = (rB * C).sum(axis=-1, keepdims=True) if not _is_tensor(rB) else (rB * C).sum(dim=-1, keepdim=True)
    s1 = (rB * QC).sum(axis=-1, keepdims=True) if not _is_tensor(rB) else (rB * QC).sum(dim=-1, keepdim=True)
    s2 = (rB * UC).sum(axis=-1, keepdims=True) if not _is_tensor(rB) else (rB * UC).sum(dim=-1, keepdim=True)

    obs_strings = [item.decode() if hasattr(item, "decode") else str(item) for item in obs_type]
    if _is_tensor(rB):
        import torch

        pred = torch.zeros_like(s0)
        spectral_idx = [idx for idx, value in enumerate(obs_strings) if value == "spectral"]
        polar_idx = [idx for idx, value in enumerate(obs_strings) if value == "polarization"]
        if spectral_idx:
            idx = torch.tensor(spectral_idx, dtype=torch.long, device=rB.device)
            pred[idx] = s0[idx]
        if polar_idx:
            idx = torch.tensor(polar_idx, dtype=torch.long, device=rB.device)
            pred[idx] = (
                analyzer[idx, 0:1] * s0[idx]
                + analyzer[idx, 1:2] * s1[idx]
                + analyzer[idx, 2:3] * s2[idx]
            )
        return pred

    pred = np.zeros_like(s0, dtype=np.float32)
    for idx, value in enumerate(obs_strings):
        if value == "spectral":
            pred[idx, 0] = s0[idx, 0]
        elif value == "polarization":
            pred[idx, 0] = analyzer[idx, 0] * s0[idx, 0] + analyzer[idx, 1] * s1[idx, 0] + analyzer[idx, 2] * s2[idx, 0]
        else:
            raise ValueError(f"unknown obs_type: {value}")
    return pred.astype(np.float32)
