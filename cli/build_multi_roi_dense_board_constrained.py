from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..observations.board_constrained import prepare_board_constrained_dataset
from ..calibration.board_prior import BoardPrior
from ..observations.dense import derive_dense_multi_roi_dataset
from ..config.paths import default_output_dir
from ..config.text_config import add_config_argument, expand_config_argv


DEFAULT_APRILGRID_JSON = (
    "G:/projects/metasurface_3d_reconstruction/code/Aprilgrid/outputs/"
    "aprilgrid_10x10_tag6mm_border80mm/aprilgrid_10x10_tag6mm_border80mm.json"
)
DEFAULT_APRILGRID_PNG = (
    "G:/projects/metasurface_3d_reconstruction/code/Aprilgrid/outputs/"
    "aprilgrid_10x10_tag6mm_border80mm/aprilgrid_10x10_tag6mm_border80mm.png"
)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--dataset-dir", "--base-dataset", dest="dataset_dir", required=True)
    parser.add_argument("--source-image-dir", default="G:/galaxydata/pawn")
    parser.add_argument("--response-mat", required=True)
    parser.add_argument("--rois-json", "--multi-roi-json", dest="rois_json", required=True)
    parser.add_argument("--aprilgrid-json", default=DEFAULT_APRILGRID_JSON)
    parser.add_argument("--aprilgrid-png", default=DEFAULT_APRILGRID_PNG)
    parser.add_argument("--output-dir", default=str(default_output_dir("dense_all16_center3x3")))
    parser.add_argument("--basis-count", type=int, default=12)
    parser.add_argument("--fwhm-nm", type=float, default=20.0)
    parser.add_argument("--center-window", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--expected-roi-count", type=int, default=4)
    parser.add_argument("--foreground-residual-threshold", type=float, default=0.08)
    parser.add_argument("--min-foreground-fraction", type=float, default=0.02)
    parser.add_argument("--fit-board-frame-affine", action="store_true")
    parser.add_argument("--strict-board-canvas", action="store_true")
    parser.add_argument("--debug-image-limit", type=int, default=20)
    parser.add_argument("--all-subcells", action="store_true")
    parser.add_argument("--include-polar", action="store_true")
    return parser


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observation_path, pose_path, dense_report = derive_dense_multi_roi_dataset(
        dataset_dir=args.dataset_dir,
        source_image_dir=args.source_image_dir,
        response_mat=args.response_mat,
        rois_json=args.rois_json,
        output_dir=output_dir,
        basis_count=args.basis_count,
        fwhm_nm=args.fwhm_nm,
        center_window=args.center_window,
        max_frames=args.max_frames,
        expected_roi_count=args.expected_roi_count,
        debug_image_limit=args.debug_image_limit,
    )

    board_prior = BoardPrior.from_files(args.aprilgrid_json, args.aprilgrid_png)
    board_dataset_path, board_report = prepare_board_constrained_dataset(
        observation_path,
        board_prior,
        output_dir,
        foreground_residual_threshold=args.foreground_residual_threshold,
        min_foreground_fraction=args.min_foreground_fraction,
        fit_board_frame_affine=bool(args.fit_board_frame_affine),
        drop_outside_board=not bool(args.strict_board_canvas),
        output_name="observations_dense_all16_center3x3_board_constrained.npz",
    )
    report = {
        "dense_dataset": dense_report,
        "board_constrained_dataset": board_report,
        "board_observation_path": str(board_dataset_path),
        "pose_path": str(pose_path),
    }
    _write_json(output_dir / "dense_board_constrained_report.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    run(build_parser().parse_args(expand_config_argv()))


if __name__ == "__main__":
    main()
