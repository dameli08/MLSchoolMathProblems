import unittest

import numpy as np

from p03_row_reduction import col_space_basis, null_space_basis, rank, rref


class RowReductionTests(unittest.TestCase):
    def test_textbook_example(self):
        matrix = np.array([[1, 2, -1], [3, -1, 4]], dtype=float)
        reduced, pivots = rref(matrix)
        np.testing.assert_allclose(reduced, [[1, 0, 1], [0, 1, -1]])
        self.assertEqual(pivots, [0, 1])

    def test_null_and_column_bases(self):
        matrix = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
        basis = null_space_basis(matrix)
        self.assertEqual(basis.shape, (2, 3))
        np.testing.assert_allclose(matrix @ basis.T, 0, atol=1e-10)
        self.assertEqual(col_space_basis(matrix).shape, (2, 1))

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(rank(np.zeros((2, 0))), 0)
        with self.assertRaises(ValueError):
            rref([1, 2])
        with self.assertRaises(ValueError):
            rref([[1]], tol=-1)

    def test_random_rank_nullity(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            matrix = rng.normal(size=(5, 7))
            basis = null_space_basis(matrix)
            self.assertEqual(rank(matrix) + len(basis), 7)
            self.assertEqual(rank(matrix), np.linalg.matrix_rank(matrix))
            np.testing.assert_allclose(matrix @ basis.T, 0, atol=1e-9)

    def test_constructed_rank_four(self):
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
        self.assertEqual(rank(matrix), 4)
        self.assertEqual(null_space_basis(matrix).shape, (4, 8))


if __name__ == "__main__":
    unittest.main()
