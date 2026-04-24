from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brunoise"))

from scanning_patterns import (  # noqa: E402
    n_total,
    reconstruct_image_pattern,
    simple_scanning_pattern,
)


def test_n_total_matches_pattern_length():
    for pause in (False, True):
        for n_x, n_y, n_turn, n_extra in (
            (400, 400, 10, 100),
            (512, 256, 10, 100),
            (128, 512, 20, 50),
            (5, 3, 2, 4),
            (5, 1, 2, 4),
        ):
            x, y = simple_scanning_pattern(n_x, n_y, n_turn, n_extra, pause)
            assert len(x) == n_total(n_x, n_y, n_turn, n_extra, pause)
            assert len(y) == len(x)


def test_reconstruct_image_pattern_uses_y_x_shape():
    scan_x = np.array([0, 1, 2, 0, 1, 2])
    scan_y = np.array([0, 0, 0, 1, 1, 1])
    signal = np.array([1, 2, 3, 4, 5, 6])

    image = reconstruct_image_pattern(signal, scan_x, scan_y, (2, 3), 1)

    np.testing.assert_array_equal(image, np.array([[1, 2, 3], [4, 5, 6]]))
