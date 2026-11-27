import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

from p12_dual_bases_interpolation import (
    dual_basis,
    interpolate,
    lagrange_coefficients,
    vandermonde,
)


class DualBasisInterpolationTests(unittest.TestCase):
    def test_dual_rows_reconstruct_coordinates(self):
        rng = np.random.default_rng(12)
        basis_matrix = rng.normal(size=(5, 5))
        dual_rows = dual_basis(basis_matrix)
        np.testing.assert_allclose(
            dual_rows @ basis_matrix, np.eye(5), atol=1e-14
        )

        vector = rng.normal(size=5)
        rebuilt = basis_matrix @ (dual_rows @ vector)
        np.testing.assert_allclose(rebuilt, vector, atol=1e-14)

    def test_lagrange_columns_are_basis_polynomial_coefficients(self):
        nodes = np.array([-1.0, 0.0, 1.0])
        coefficients = lagrange_coefficients(nodes)
        np.testing.assert_allclose(vandermonde(nodes) @ coefficients, np.eye(3))

    def test_interpolation_is_exact_for_low_degree_polynomials(self):
        nodes = np.linspace(-1, 1, 5)
        values = 2 - 3 * nodes + nodes**2 + 0.5 * nodes**3
        query = np.linspace(-1, 1, 31)
        expected = 2 - 3 * query + query**2 + 0.5 * query**3
        np.testing.assert_allclose(interpolate(nodes, values, query), expected)

    def test_invalid_nodes_and_shapes(self):
        with self.assertRaises(ValueError):
            vandermonde([1, 1])
        with self.assertRaises(ValueError):
            interpolate([0, 1], [1], [0.5])
        with self.assertRaises(np.linalg.LinAlgError):
            dual_basis([[1, 2], [2, 4]])


if __name__ == "__main__":
    unittest.main()
