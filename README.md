# Metasurface Data Pipeline

Model-agnostic real metasurface data preparation for AprilGrid-calibrated
multi-view captures. This package produces `.npz`/`.json` observation datasets
that can be consumed by 3DGS, NeSpoF, or later reconstruction models without
importing model training code.

Generated data defaults to:

```text
code/metasurface_data_pipeline/outputs/
```

`outputs/` is intentionally ignored by git.

## Package Layout

```text
metasurface_data_pipeline/
  calibration/     AprilGrid detection, camera calibration, board texture prior
  config/          ROI definitions, output path helpers
  core/            Spectral basis, response extraction, OpenCV ray geometry
  observations/    Base pose dataset, sparse/dense observations, board constraint
  cli/             Command-line entrypoints
  tests/           Unit and package-layout tests
```

Top-level modules such as `metasurface_data_pipeline.roi` and
`metasurface_data_pipeline.dense_observations` remain as compatibility wrappers.
New code should import from the categorized packages.

## Customizable Parameters

Use `--help` on any CLI to see the full set of parameters:

```powershell
python -m metasurface_data_pipeline.cli.build_base_dataset --help
python -m metasurface_data_pipeline.cli.select_rois --help
python -m metasurface_data_pipeline.cli.build_multi_roi_sparse --help
python -m metasurface_data_pipeline.cli.build_multi_roi_dense_board_constrained --help
```

Common parameters:

| Purpose | Parameter |
| --- | --- |
| Raw image folder | `--cube-image-dir`, `--source-image-dir` |
| AprilGrid geometry | `--cube-aprilgrid-json`, `--aprilgrid-json`, `--aprilgrid-png` |
| Intrinsic calibration source | `--intrinsic-candidate name=image_dir,aprilgrid_json` |
| Frame selection | `--max-frames`, `--frame-stride`, `--use-half-frames`, `--calibration-max-frames` |
| Pose quality | `--min-markers`, `--reprojection-threshold-px`, `--min-accepted-frames` |
| Metasurface response | `--response-mat`, `--basis-count`, `--fwhm-nm` |
| ROI selection | `--rois-json`, `--multi-roi-json`, `--expected-roi-count` |
| Dense subcell sampling | `--center-window`; currently must be an odd value within each `5x5` subcell |
| Board/object split | `--foreground-residual-threshold`, `--min-foreground-fraction`, `--fit-board-frame-affine` |
| Output location | `--output-dir` |

Design parameters such as ROI size, unit size, subcells per side, units per side,
and polar subcell IDs are currently code-level constants in:

```text
config/roi.py
core/real_metasurface.py
observations/dense.py
observations/spectral_pixel.py
```

If the physical metasurface design changes, update those modules together and
add tests for the new expected row counts and analyzer layout.

## Main Commands

Build a base AprilGrid pose dataset from raw images:

```powershell
cd G:\projects\metasurface_3d_reconstruction\code

G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m metasurface_data_pipeline.cli.build_base_dataset `
  --cube-image-dir G:\galaxydata\cube2 `
  --cube-aprilgrid-json G:\projects\metasurface_3d_reconstruction\code\Aprilgrid\outputs\aprilgrid_10x10_5mm.json `
  --intrinsic-candidate cube2=G:\galaxydata\cube2,G:\projects\metasurface_3d_reconstruction\code\Aprilgrid\outputs\aprilgrid_10x10_5mm.json `
  --output-dir metasurface_data_pipeline\outputs\cube2_base
```

Select four `160x160` metasurface ROIs:

```powershell
G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m metasurface_data_pipeline.cli.select_rois `
  --dataset-dir metasurface_data_pipeline\outputs\cube2_base `
  --output metasurface_data_pipeline\outputs\cube2_base\metasurface_rois.json `
  --overlay metasurface_data_pipeline\outputs\cube2_base\metasurface_rois_overlay.png
```

Build dense all-subcell board-constrained observations:

```powershell
G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m metasurface_data_pipeline.cli.build_multi_roi_dense_board_constrained `
  --base-dataset metasurface_data_pipeline\outputs\cube2_base `
  --source-image-dir G:\galaxydata\cube2 `
  --response-mat ..\metasurface_data\shuju\A_matrix_normalized.mat `
  --multi-roi-json metasurface_data_pipeline\outputs\cube2_base\metasurface_rois.json `
  --aprilgrid-json G:\projects\metasurface_3d_reconstruction\code\Aprilgrid\outputs\aprilgrid_10x10_5mm.json `
  --aprilgrid-png G:\projects\metasurface_3d_reconstruction\code\Aprilgrid\outputs\aprilgrid_10x10_5mm.png `
  --output-dir metasurface_data_pipeline\outputs\cube2_dense_all16_center3x3 `
  --center-window 3 `
  --include-polar
```

Key outputs for current 3DGS training:

```text
observations_dense_all16_center3x3_board_constrained.npz
poses_bounds_multi_roi.npz
dense_dataset_report.json
board_dataset_report.json
metasurface_rois.json
```

## GitHub Repository Setup

Recommended local setup:

```powershell
cd G:\projects\metasurface_3d_reconstruction\code\metasurface_data_pipeline

git init
git add .gitignore README.md *.py calibration config core observations cli tests
git commit -m "feat: initialize metasurface data pipeline"
git branch -M main
git remote add origin https://github.com/<your-name>/metasurface-data-pipeline.git
git push -u origin main
```

Do not commit generated data:

```text
outputs/
__pycache__/
*.pyc
*.npz
*.npy
*.png
*.jpg
*.ply
*.pt
```

If you later want this package installable with `pip install -e .`, add a
minimal `pyproject.toml` and move the package under a standard `src/` layout.
For the current local research workflow, direct `python -m ...` usage from the
`code/` directory is sufficient.

## Scope

This package should not import NeSpoF or 3DGS training code. It may emit data
that those models read, but model-specific optimization and rendering stay in
their own repositories.
