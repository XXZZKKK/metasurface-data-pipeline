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


if __name__ == "__main__":
    unittest.main()
