"""LU factorization without row swaps and rank-one inverse updates (P16)."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lu(
    A: ArrayLike, tol: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return L, U with A = L @ U, without row swaps."""
    matrix = np.array(A, dtype=float, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("A must contain only finite values")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite nonnegative number")

    size = matrix.shape[0]
    lower = np.eye(size)
    upper = matrix.copy()
    for column in range(size):
        pivot = upper[column, column]
        if abs(pivot) <= tol:
            raise np.linalg.LinAlgError(
                f"zero or too-small pivot at column {column}; partial pivoting is needed"
            )
        if column + 1 < size:
            multipliers = upper[column + 1 :, column] / pivot
            lower[column + 1 :, column] = multipliers
            upper[column + 1 :, column:] -= (
                multipliers[:, np.newaxis] * upper[column, column:]
            )
            upper[column + 1 :, column] = 0.0
    return lower, upper


def solve_lu(
    L: ArrayLike, U: ArrayLike, b: ArrayLike
) -> NDArray[np.float64]:
    """Solve L U x = b by forward and back substitution."""
    lower = np.asarray(L, dtype=float)
    upper = np.asarray(U, dtype=float)
    right_hand_side = np.asarray(b, dtype=float)
    if (
        lower.ndim != 2
        or upper.ndim != 2
        or lower.shape[0] != lower.shape[1]
        or upper.shape != lower.shape
    ):
        raise ValueError("L and U must be square matrices of the same shape")
    size = lower.shape[0]
    if right_hand_side.ndim not in (1, 2) or right_hand_side.shape[0] != size:
        raise ValueError("b must have shape (n,) or (n, number_of_rhs)")
    if not (
        np.isfinite(lower).all()
        and np.isfinite(upper).all()
        and np.isfinite(right_hand_side).all()
    ):
        raise ValueError("L, U and b must contain only finite values")

    solution_shape = right_hand_side.shape
    intermediate = np.empty(solution_shape, dtype=float)
    solution = np.empty(solution_shape, dtype=float)
    for row in range(size):
        intermediate[row] = right_hand_side[row] - lower[row, :row] @ intermediate[:row]
    for row in range(size - 1, -1, -1):
        diagonal = upper[row, row]
        if diagonal == 0:
            raise np.linalg.LinAlgError(f"zero diagonal in U at row {row}")
        solution[row] = (
            intermediate[row] - upper[row, row + 1 :] @ solution[row + 1 :]
        ) / diagonal
    return solution


def solve_no_pivot(A: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Solve a system using the no-row-swap LU factorization."""
    lower, upper = lu(A)
    return solve_lu(lower, upper, b)


def inverse_gram_rank_one_updates(
    initial_rows: ArrayLike, added_rows: ArrayLike
) -> NDArray[np.float64]:
    """Update (X.T @ X)^-1 as new rows are appended to X."""
    initial = np.asarray(initial_rows, dtype=float)
    additions = np.asarray(added_rows, dtype=float)
    if initial.ndim != 2 or additions.ndim != 2:
        raise ValueError("initial_rows and added_rows must be matrices")
    if initial.shape[1] == 0 or initial.shape[0] < initial.shape[1]:
        raise ValueError("initial_rows must contain at least as many rows as columns")
    if additions.shape[1] != initial.shape[1]:
        raise ValueError("initial_rows and added_rows must have the same column count")
    if not np.isfinite(initial).all() or not np.isfinite(additions).all():
        raise ValueError("input rows must contain only finite values")

    inverse = np.linalg.inv(initial.T @ initial)
    for row in additions:
        projected = inverse @ row
        denominator = 1.0 + row @ projected
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise np.linalg.LinAlgError("rank-one update produced an invalid denominator")
        inverse -= np.outer(projected, projected) / denominator
    return inverse


def benchmark_gram_inverse_updates(
    initial_rows: ArrayLike, added_rows: ArrayLike
) -> tuple[NDArray[np.float64], float, float]:
    """Return the updated inverse and elapsed times for incremental and fresh solves."""
    initial = np.asarray(initial_rows, dtype=float)
    additions = np.asarray(added_rows, dtype=float)
    start = perf_counter()
    updated = inverse_gram_rank_one_updates(initial, additions)
    incremental_seconds = perf_counter() - start

    all_rows = np.vstack((initial, additions))
    start = perf_counter()
    from_scratch = np.linalg.inv(all_rows.T @ all_rows)
    from_scratch_seconds = perf_counter() - start
    if not np.allclose(updated, from_scratch, rtol=1e-8, atol=1e-10):
        raise ArithmeticError("incremental and recomputed inverses disagree")
    return updated, incremental_seconds, from_scratch_seconds


if __name__ == "__main__":
    rng = np.random.default_rng(16)
    matrix = rng.normal(size=(6, 6))
    lower, upper = lu(matrix)
    assert np.allclose(lower @ upper, matrix)
    right_hand_side = rng.normal(size=6)
    assert np.allclose(solve_lu(lower, upper, right_hand_side), np.linalg.solve(matrix, right_hand_side))

    ill_conditioned = np.array([[1e-17, 1.0], [1.0, 1.0]])
    rhs = np.array([1.0, 2.0])
    print("No-pivot solution:", solve_no_pivot(ill_conditioned, rhs))
    print("NumPy solution:   ", np.linalg.solve(ill_conditioned, rhs))

    initial = rng.normal(size=(100, 50))
    additions = rng.normal(size=(100, 50))
    _, updated_time, fresh_time = benchmark_gram_inverse_updates(initial, additions)
    print(f"100 sequential updates: {updated_time:.6f}s")
    print(f"Recompute at final size: {fresh_time:.6f}s")
