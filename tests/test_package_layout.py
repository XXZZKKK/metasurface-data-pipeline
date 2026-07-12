from __future__ import annotations

import importlib
import unittest


class MetasurfaceDataPipelineLayoutTests(unittest.TestCase):
    def test_categorized_modules_are_importable(self):
        modules = [
            "metasurface_data_pipeline.core.basis",
            "metasurface_data_pipeline.core.geometry",
            "metasurface_data_pipeline.core.real_metasurface",
            "metasurface_data_pipeline.core.response",
            "metasurface_data_pipeline.calibration.aprilgrid",
            "metasurface_data_pipeline.calibration.board_prior",
            "metasurface_data_pipeline.config.paths",
            "metasurface_data_pipeline.config.roi",
            "metasurface_data_pipeline.config.text_config",
            "metasurface_data_pipeline.compat.roi",
            "metasurface_data_pipeline.compat.dense_observations",
            "metasurface_data_pipeline.observations.base_dataset",
            "metasurface_data_pipeline.observations.board_constrained",
            "metasurface_data_pipeline.observations.dense",
            "metasurface_data_pipeline.observations.sparse",
            "metasurface_data_pipeline.observations.spectral_pixel",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_legacy_top_level_imports_still_work(self):
        from metasurface_data_pipeline import basis, dense_observations, roi
        from metasurface_data_pipeline.core import basis as categorized_basis
        from metasurface_data_pipeline.config import roi as categorized_roi
        from metasurface_data_pipeline.observations import dense as categorized_dense

        self.assertIs(basis.build_gaussian_basis, categorized_basis.build_gaussian_basis)
        self.assertIs(roi.MetasurfaceRoi, categorized_roi.MetasurfaceRoi)
        self.assertIs(
            dense_observations.dense_observation_template,
            categorized_dense.dense_observation_template,
        )

    def test_legacy_wrapper_files_are_grouped_under_compat(self):
        from pathlib import Path

        package_root = Path(__file__).resolve().parents[1]
        legacy_names = [
            "aprilgrid.py",
            "base_dataset.py",
            "basis.py",
            "board_constrained_dataset.py",
            "board_prior.py",
            "dense_observations.py",
            "geometry.py",
            "paths.py",
            "real_metasurface.py",
            "response.py",
            "roi.py",
            "sparse_observations.py",
            "spectral_pixel_observations.py",
        ]
        for name in legacy_names:
            with self.subTest(wrapper=name):
                self.assertFalse((package_root / name).exists())
                self.assertTrue((package_root / "compat" / name).is_file())


if __name__ == "__main__":
    unittest.main()
