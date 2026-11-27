"""Tensor indexing, contractions and rank-one tensors (homework P15)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def tensor_indexing_check() -> bool:
    """Check the reshape/index formula for the specified 2 by 3 by 4 tensor."""
    tensor = np.arange(24).reshape(2, 3, 4)
    i, j, k = np.indices(tensor.shape)
    flattened_index = 12 * i + 4 * j + k
    return bool(np.array_equal(tensor[i, j, k], tensor.reshape(-1)[flattened_index]))


def tensor_unfoldings(
    tensor: ArrayLike,
) -> tuple[NDArray, NDArray]:
    """Return the two unfoldings requested for a tensor of shape (2, 3, 4)."""
    array = np.asarray(tensor)
    if array.shape != (2, 3, 4):
        raise ValueError("tensor must have shape (2, 3, 4)")
    return array.reshape(2, 12), array.transpose(1, 0, 2).reshape(3, 8)


def tensor_unfolding_ranks(tensor: ArrayLike) -> tuple[int, int]:
    first, second = tensor_unfoldings(tensor)
    return int(np.linalg.matrix_rank(first)), int(np.linalg.matrix_rank(second))


def einsum_trace(matrix: ArrayLike) -> float:
    """Compute a square matrix's trace with einsum."""
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return float(np.einsum("ii->", array))


def einsum_outer(u: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """Compute the outer product using einsum."""
    left = np.asarray(u, dtype=float)
    right = np.asarray(v, dtype=float)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("u and v must be one-dimensional vectors")
    return np.einsum("i,j->ij", left, right)


def batch_trace_and_products(
    matrices: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return traces and squares of a batch of square matrices via einsum."""
    batch = np.asarray(matrices, dtype=float)
    if batch.ndim != 3 or batch.shape[1] != batch.shape[2]:
        raise ValueError("matrices must have shape (batch, n, n)")
    traces = np.einsum("bii->b", batch)
    products = np.einsum("bij,bjk->bik", batch, batch)
    return traces, products


def batch_trace_and_products_loop(
    matrices: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reference Python-loop implementation for comparing batch operations."""
    batch = np.asarray(matrices, dtype=float)
    if batch.ndim != 3 or batch.shape[1] != batch.shape[2]:
        raise ValueError("matrices must have shape (batch, n, n)")
    traces = np.array([np.trace(matrix) for matrix in batch])
    products = np.array([matrix @ matrix for matrix in batch])
    return traces, products


def kron_bilinear_identity(
    A: ArrayLike, B: ArrayLike, X: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compare vec(A X B.T) with kron(A, B) vec(X), using row-major vec."""
    left_factor = np.asarray(A, dtype=float)
    right_factor = np.asarray(B, dtype=float)
    middle = np.asarray(X, dtype=float)
    if left_factor.ndim != 2 or right_factor.ndim != 2 or middle.ndim != 2:
        raise ValueError("A, B and X must be matrices")
    if middle.shape != (left_factor.shape[1], right_factor.shape[1]):
        raise ValueError("X must have shape (A.shape[1], B.shape[1])")
    direct = left_factor @ middle @ right_factor.T
    kron_result = np.kron(left_factor, right_factor) @ middle.reshape(-1)
    return direct.reshape(-1), kron_result


def kron_storage_requirements(
    shape_a: tuple[int, int], shape_b: tuple[int, int], itemsize: int = 8
) -> tuple[int, int]:
    """Return (number of entries, bytes) for np.kron of two matrix shapes."""
    if len(shape_a) != 2 or len(shape_b) != 2:
        raise ValueError("both shapes must describe matrices")
    if any(size < 0 for size in (*shape_a, *shape_b)):
        raise ValueError("matrix dimensions must be nonnegative")
    if itemsize <= 0:
        raise ValueError("itemsize must be positive")
    entries = shape_a[0] * shape_b[0] * shape_a[1] * shape_b[1]
    return entries, entries * itemsize


def rank_one(u: ArrayLike, v: ArrayLike, w: ArrayLike) -> NDArray[np.float64]:
    """Return the rank-one tensor u_i v_j w_k using einsum."""
    first = np.asarray(u, dtype=float)
    second = np.asarray(v, dtype=float)
    third = np.asarray(w, dtype=float)
    if any(vector.ndim != 1 for vector in (first, second, third)):
        raise ValueError("u, v and w must be one-dimensional vectors")
    return np.einsum("i,j,k->ijk", first, second, third)


if __name__ == "__main__":
    tensor = np.arange(24).reshape(2, 3, 4)
    print("Index formula verified:", tensor_indexing_check())
    print("Unfolding ranks:", tensor_unfolding_ranks(tensor))

    rng = np.random.default_rng(15)
    batch = rng.normal(size=(100, 4, 4))
    einsum_results = batch_trace_and_products(batch)
    loop_results = batch_trace_and_products_loop(batch)
    assert np.allclose(einsum_results[0], loop_results[0])
    assert np.allclose(einsum_results[1], loop_results[1])

    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(2, 5))
    X = rng.normal(size=(4, 5))
    direct, kron = kron_bilinear_identity(A, B, X)
    assert np.allclose(direct, kron)
    entries, byte_count = kron_storage_requirements((100, 100), (100, 100))
    print(f"np.kron(100x100, 100x100): {entries:,} float64 entries ({byte_count:,} bytes)")

    u, v, w = (rng.normal(size=n) for n in (3, 4, 5))
    result = rank_one(u, v, w)
    for k in range(len(w)):
        assert np.allclose(result[:, :, k], w[k] * np.outer(u, v))
