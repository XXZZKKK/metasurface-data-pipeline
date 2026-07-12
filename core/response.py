"""Response-table helpers for metasurface scalar observations."""

import numpy as np


def _shape3(array_like):
    first = array_like[0]
    second = first[0]
    return len(array_like), len(first), len(second)


def aggregate_a_matrix_to_subcells(a_matrix, pixels_per_unit_side=20, subcells_per_side=4):
    """Aggregate A_matrix [400,L,U] into subcell responses [16,L,U].

    MATLAB's A_matrix indexes a 20x20 large unit. The center-ray MVP uses a
    4x4 subcell grid, so each subcell receives the mean of its 5x5 pixel block.
    """
    pixel_count, wavelength_count, unit_count = _shape3(a_matrix)
    expected_pixels = pixels_per_unit_side * pixels_per_unit_side
    if pixel_count != expected_pixels:
        raise ValueError(f"expected {expected_pixels} flattened pixels, got {pixel_count}")
    if pixels_per_unit_side % subcells_per_side != 0:
        raise ValueError("pixels_per_unit_side must be divisible by subcells_per_side")

    block_size = pixels_per_unit_side // subcells_per_side
    response = []
    for sub_row in range(subcells_per_side):
        for sub_col in range(subcells_per_side):
            sums = [[0.0 for _ in range(unit_count)] for _ in range(wavelength_count)]
            count = 0

            for row in range(sub_row * block_size, (sub_row + 1) * block_size):
                for col in range(sub_col * block_size, (sub_col + 1) * block_size):
                    pixel_id = row * pixels_per_unit_side + col
                    for wave_id in range(wavelength_count):
                        for unit_id in range(unit_count):
                            sums[wave_id][unit_id] += float(a_matrix[pixel_id][wave_id][unit_id])
                    count += 1

            response.append(
                [
                    [value / count for value in wave_values]
                    for wave_values in sums
                ]
            )

    return response


def extract_center2x2_response_table(a_matrix, pixels_per_unit_side=20, subcells_per_side=4):
    """Extract center-right-down 2x2 effective responses from A_matrix.

    Each 20x20 large unit is split into a 4x4 subcell grid. For every 5x5
    subcell block, this function averages local rows/cols [2,3] x [2,3].
    That matches the first simulation MVP observation value, which also uses
    the same center 2x2 CMOS pixels.
    """
    pixel_count, wavelength_count, unit_count = _shape3(a_matrix)
    expected_pixels = pixels_per_unit_side * pixels_per_unit_side
    if pixel_count != expected_pixels:
        raise ValueError(f"expected {expected_pixels} flattened pixels, got {pixel_count}")
    if pixels_per_unit_side % subcells_per_side != 0:
        raise ValueError("pixels_per_unit_side must be divisible by subcells_per_side")

    block_size = pixels_per_unit_side // subcells_per_side
    if block_size < 4:
        raise ValueError("center2x2 extraction requires subcell blocks at least 4x4")

    response = []
    for sub_row in range(subcells_per_side):
        for sub_col in range(subcells_per_side):
            sums = [[0.0 for _ in range(unit_count)] for _ in range(wavelength_count)]
            count = 0
            row0 = sub_row * block_size
            col0 = sub_col * block_size

            for row in (row0 + 2, row0 + 3):
                for col in (col0 + 2, col0 + 3):
                    pixel_id = row * pixels_per_unit_side + col
                    for wave_id in range(wavelength_count):
                        for unit_id in range(unit_count):
                            sums[wave_id][unit_id] += float(a_matrix[pixel_id][wave_id][unit_id])
                    count += 1

            response.append(
                [
                    [value / count for value in wave_values]
                    for wave_values in sums
                ]
            )

    return response


def extract_pixel_response_vectors(
    a_matrix,
    unit_ids,
    unit_rows,
    unit_cols,
    pixels_per_unit_side=20,
):
    """Extract calibrated response vectors for individual pixels in a unit."""
    array = np.asarray(a_matrix, dtype=np.float32)
    unit_ids = np.asarray(unit_ids, dtype=np.int64)
    rows = np.asarray(unit_rows, dtype=np.int64)
    cols = np.asarray(unit_cols, dtype=np.int64)

    expected_pixels = int(pixels_per_unit_side) ** 2
    if array.ndim != 3 or array.shape[0] != expected_pixels:
        raise ValueError(f"a_matrix must have shape [{expected_pixels},L,U]")
    if not (unit_ids.shape == rows.shape == cols.shape):
        raise ValueError("unit_ids, unit_rows, and unit_cols must have matching shapes")
    if np.any(rows < 0) or np.any(rows >= pixels_per_unit_side):
        raise ValueError("unit_rows are outside the calibrated unit")
    if np.any(cols < 0) or np.any(cols >= pixels_per_unit_side):
        raise ValueError("unit_cols are outside the calibrated unit")
    if np.any(unit_ids < 0) or np.any(unit_ids >= array.shape[2]):
        raise ValueError("unit_ids are outside the calibration table")

    pixel_ids = rows * int(pixels_per_unit_side) + cols
    return array[pixel_ids, :, unit_ids].astype(np.float32)


def normalize_response_table(response, eps=1e-12):
    """Normalize each subcell/unit response over wavelength for stable projection."""
    subcell_count, wavelength_count, unit_count = _shape3(response)
    normalized = [
        [[0.0 for _ in range(unit_count)] for _ in range(wavelength_count)]
        for _ in range(subcell_count)
    ]

    for subcell_id in range(subcell_count):
        for unit_id in range(unit_count):
            total = sum(max(float(response[subcell_id][wave_id][unit_id]), 0.0) for wave_id in range(wavelength_count))
            denom = total if total > eps else 1.0
            for wave_id in range(wavelength_count):
                value = max(float(response[subcell_id][wave_id][unit_id]), 0.0)
                normalized[subcell_id][wave_id][unit_id] = value / denom

    return normalized


def response_vectors_for_observations(response_table, unit_ids, subcell_ids):
    """Gather [L] response vectors from [16,L,U] table for each observation."""
    vectors = []
    for unit_id, subcell_id in zip(unit_ids, subcell_ids):
        subcell_response = response_table[int(subcell_id)]
        vectors.append(
            [
                wave_values[int(unit_id)]
                for wave_values in subcell_response
            ]
        )
    return vectors
