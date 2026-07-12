from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class MetasurfaceDataPipelineCoreTests(unittest.TestCase):
    def test_roi_validation_and_union_crop_keep_fixed_160_square(self):
        from metasurface_data_pipeline.roi import (
            MetasurfaceRoi,
            fixed_roi_from_center,
            union_crop,
            validate_rois,
        )

        base = MetasurfaceRoi(name="roi0", x0=1746, y0=1019)
        extra = fixed_roi_from_center(
            "roi1",
            center_x=2100,
            center_y=1100,
            image_width=3840,
            image_height=2160,
        )
        rois = validate_rois([base, extra], image_width=3840, image_height=2160)

        self.assertEqual([(roi.width, roi.height) for roi in rois], [(160, 160), (160, 160)])
        self.assertEqual((extra.x0, extra.y0), (2020, 1020))
        self.assertEqual(union_crop(rois), (1746, 1019, 434, 161))

    def test_dense_template_has_expected_center3x3_all16_rows(self):
        from metasurface_data_pipeline.dense_observations import dense_observation_template

        template = dense_observation_template(center_window=3)

        self.assertEqual(template["local_uv"].shape, (9216, 2))
        self.assertEqual(int((template["subcell_id"] == 0).sum()), 576)
        np.testing.assert_array_equal(np.unique(template["subcell_id"]), np.arange(16))
        np.testing.assert_array_equal(np.unique(template["pixel_offset_id"]), np.arange(9))
        np.testing.assert_array_equal(np.unique(template["analyzer_id"][template["is_polar"]]), np.arange(4))

    def test_compact_response_basis_projection_matches_direct_rows(self):
        from metasurface_data_pipeline.basis import build_gaussian_basis, response_basis_projection
        from metasurface_data_pipeline.response import extract_pixel_response_vectors

        wavelengths = np.linspace(460.0, 650.0, 8, dtype=np.float32)
        centers = np.linspace(460.0, 650.0, 3, dtype=np.float32)
        basis = build_gaussian_basis(wavelengths, centers, fwhm_nm=20.0)
        a_matrix = np.arange(400 * 8 * 64, dtype=np.float32).reshape(400, 8, 64) + 1.0

        unit_ids = np.asarray([0, 7, 63], dtype=np.int64)
        rows = np.asarray([2, 8, 18], dtype=np.int64)
        cols = np.asarray([2, 9, 19], dtype=np.int64)
        response = extract_pixel_response_vectors(a_matrix, unit_ids, rows, cols)
        projected = response_basis_projection(response, basis)

        direct = []
        for unit_id, row, col in zip(unit_ids, rows, cols):
            vector = a_matrix[row * 20 + col, :, unit_id]
            weights = np.maximum(vector, 0.0)
            weights = weights / max(float(weights.sum()), 1e-12)
            direct.append(weights @ basis.T)
        np.testing.assert_allclose(projected, np.asarray(direct, dtype=np.float32), rtol=1e-6, atol=1e-6)

    def test_aprilgrid_json_loader_validates_marker_corners(self):
        from metasurface_data_pipeline.aprilgrid import load_aprilgrid_spec

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.json"
            path.write_text(
                """
                {
                  "dictionary": "DICT_APRILTAG_36h11",
                  "markers": [
                    {"id": 5, "corners_m": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]}
                  ]
                }
                """,
                encoding="utf-8",
            )
            spec = load_aprilgrid_spec(path)

        self.assertEqual(spec.dictionary_name, "DICT_APRILTAG_36h11")
        self.assertEqual(list(spec.markers.keys()), [5])
        self.assertEqual(spec.markers[5].corners_m.shape, (4, 3))

    def test_text_config_expands_to_cli_arguments(self):
        from metasurface_data_pipeline.config.text_config import config_file_to_argv, expand_config_argv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense_config.txt"
            path.write_text(
                """
                # comments and blank lines are ignored
                dataset_dir = G:/data/base
                source_image_dir = "G:/data folder/pawn"
                center-window = 3
                include-polar = true
                strict-board-canvas = false
                intrinsic-candidate = pawn=G:/images,G:/grid.json
                base-roi = 1746 1019
                """,
                encoding="utf-8",
            )

            expanded = config_file_to_argv(path)
            self.assertIn("--dataset-dir", expanded)
            self.assertIn("G:/data/base", expanded)
            self.assertIn("--source-image-dir", expanded)
            self.assertIn("G:/data folder/pawn", expanded)
            self.assertIn("--include-polar", expanded)
            self.assertNotIn("--strict-board-canvas", expanded)
            self.assertEqual(expanded[expanded.index("--intrinsic-candidate") + 1], "pawn=G:/images,G:/grid.json")
            self.assertEqual(expanded[expanded.index("--base-roi") + 1 : expanded.index("--base-roi") + 3], ["1746", "1019"])

            combined = expand_config_argv(["--config", str(path), "--center-window", "1"])
            self.assertEqual(combined[-2:], ["--center-window", "1"])

    def test_example_text_configs_are_parseable(self):
        from metasurface_data_pipeline.config.text_config import config_file_to_argv

        package_root = Path(__file__).resolve().parents[1]
        for path in sorted((package_root / "examples").glob("*.txt")):
            with self.subTest(config=path.name):
                argv = config_file_to_argv(path)
                self.assertGreater(len(argv), 0)


if __name__ == "__main__":
    unittest.main()
