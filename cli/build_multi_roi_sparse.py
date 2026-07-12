from __future__ import annotations

import json

from ..observations.sparse import build_parser, derive_multi_roi_spectral_dataset
from ..config.text_config import expand_config_argv


def main():
    args = build_parser().parse_args(expand_config_argv())
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
