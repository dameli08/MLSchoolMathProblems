import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from p06_linear_transformations import (
    UNIT_SQUARE,
    plot_four_transformations,
    plot_transformation,
    signed_area,
)


class LinearTransformationTests(unittest.TestCase):
    def test_signed_area_matches_determinant_including_orientation(self):
        matrices = [
            np.eye(2),
            np.array([[1, 2], [2, 4]], dtype=float),
            np.array([[3, 1], [1, 2]], dtype=float),
            np.array([[0, 1], [1, 0]], dtype=float),
        ]
        for matrix in matrices:
            with self.subTest(matrix=matrix):
                image = UNIT_SQUARE @ matrix.T
                self.assertAlmostEqual(signed_area(image), np.linalg.det(matrix))

    def test_plot_helpers_and_matrix_count(self):
        figure, axes = plot_transformation([[1, 2], [2, 4]])
        self.assertEqual(axes.shape, (2,))
        plt.close(figure)

        figure, axes = plot_four_transformations([np.eye(2)] * 4)
        self.assertEqual(axes.shape, (2, 4))
        plt.close(figure)

        with self.assertRaises(ValueError):
            plot_four_transformations([np.eye(2)])
        with self.assertRaises(ValueError):
            plot_transformation(np.eye(3))

    def test_degenerate_polygons_are_rejected(self):
        with self.assertRaises(ValueError):
            signed_area([[0, 0], [1, 1]])


if __name__ == "__main__":
    unittest.main()
