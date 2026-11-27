import unittest

import numpy as np

from p16_lu_rank_one_updates import (
    inverse_gram_rank_one_updates,
    lu,
    solve_lu,
    solve_no_pivot,
)


class LURankOneTests(unittest.TestCase):
    def test_lu_reconstructs_random_matrices_and_solves(self):
        rng = np.random.default_rng(16)
        for size in (2, 4, 6):
            matrix = rng.normal(size=(size, size))
            lower, upper = lu(matrix)
            np.testing.assert_allclose(lower @ upper, matrix, atol=1e-12)
            rhs = rng.normal(size=(size, 3))
            np.testing.assert_allclose(
                solve_lu(lower, upper, rhs), np.linalg.solve(matrix, rhs), atol=1e-10
            )

    def test_no_pivoting_has_expected_ill_conditioned_failure(self):
        matrix = np.array([[1e-17, 1.0], [1.0, 1.0]])
        rhs = np.array([1.0, 2.0])
        no_pivot = solve_no_pivot(matrix, rhs)
        pivoted = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(pivoted, [1.0, 1.0], atol=1e-12)
        self.assertGreater(np.linalg.norm(matrix @ no_pivot - rhs), 0.1)

    def test_zero_pivot_is_reported(self):
        with self.assertRaises(np.linalg.LinAlgError):
            lu([[0, 1], [1, 1]])

    def test_sequential_inverse_updates_match_recomputation(self):
        rng = np.random.default_rng(17)
        initial = rng.normal(size=(100, 50))
        additions = rng.normal(size=(100, 50))
        updated = inverse_gram_rank_one_updates(initial, additions)
        all_rows = np.vstack((initial, additions))
        expected = np.linalg.inv(all_rows.T @ all_rows)
        np.testing.assert_allclose(updated, expected, rtol=1e-8, atol=1e-10)

    def test_incompatible_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            lu(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            inverse_gram_rank_one_updates(np.ones((3, 2)), np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
