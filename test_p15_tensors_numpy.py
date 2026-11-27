import unittest

import numpy as np

from p15_tensors_numpy import (
    batch_trace_and_products,
    batch_trace_and_products_loop,
    einsum_outer,
    einsum_trace,
    kron_bilinear_identity,
    kron_storage_requirements,
    rank_one,
    tensor_indexing_check,
    tensor_unfoldings,
    tensor_unfolding_ranks,
)


class TensorNumpyTests(unittest.TestCase):
    def test_indexing_and_unfolding_ranks(self):
        tensor = np.arange(24).reshape(2, 3, 4)
        self.assertTrue(tensor_indexing_check())
        first, second = tensor_unfoldings(tensor)
        self.assertEqual(first.shape, (2, 12))
        self.assertEqual(second.shape, (3, 8))
        self.assertEqual(
            tensor_unfolding_ranks(tensor),
            (np.linalg.matrix_rank(first), np.linalg.matrix_rank(second)),
        )

    def test_einsum_trace_outer_and_batches(self):
        matrix = np.arange(16).reshape(4, 4)
        self.assertEqual(einsum_trace(matrix), np.trace(matrix))
        np.testing.assert_allclose(einsum_outer([1, 2], [3, 4]), np.outer([1, 2], [3, 4]))

        rng = np.random.default_rng(15)
        batch = rng.normal(size=(100, 4, 4))
        for einsum_result, loop_result in zip(
            batch_trace_and_products(batch),
            batch_trace_and_products_loop(batch),
        ):
            np.testing.assert_allclose(einsum_result, loop_result)

    def test_kron_identity_and_storage_cost(self):
        rng = np.random.default_rng(16)
        A = rng.normal(size=(3, 4))
        B = rng.normal(size=(2, 5))
        X = rng.normal(size=(4, 5))
        direct, kron = kron_bilinear_identity(A, B, X)
        np.testing.assert_allclose(direct, kron)
        self.assertEqual(kron_storage_requirements((100, 100), (100, 100)), (100_000_000, 800_000_000))

    def test_rank_one_slices(self):
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([-1.0, 4.0])
        w = np.array([0.5, 2.0, -3.0, 1.0])
        tensor = rank_one(u, v, w)
        self.assertEqual(tensor.shape, (3, 2, 4))
        for k in range(w.size):
            np.testing.assert_allclose(tensor[:, :, k], w[k] * np.outer(u, v))

    def test_bad_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            tensor_unfoldings(np.zeros((2, 2, 2)))
        with self.assertRaises(ValueError):
            batch_trace_and_products(np.zeros((3, 2, 3)))
        with self.assertRaises(ValueError):
            kron_bilinear_identity(np.zeros((3, 4)), np.zeros((2, 5)), np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
