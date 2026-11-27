"""Reduced row-echelon form and fundamental subspaces (homework P3)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rref(
    A: ArrayLike, tol: float = 1e-10
) -> tuple[NDArray[np.float64], list[int]]:
    """Return the reduced row-echelon form and pivot-column indices."""
    matrix = np.array(A, dtype=float, copy=True)
    if matrix.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("A must contain only finite values")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite nonnegative number")

    rows, columns = matrix.shape
    pivot_row = 0
    pivots: list[int] = []
    for column in range(columns):
        if pivot_row == rows:
            break
        candidate = pivot_row + int(np.argmax(np.abs(matrix[pivot_row:, column])))
        pivot = matrix[candidate, column]
        if abs(pivot) <= tol:
            continue

        matrix[[pivot_row, candidate]] = matrix[[candidate, pivot_row]]
        matrix[pivot_row] /= matrix[pivot_row, column]
        for row in range(rows):
            if row != pivot_row:
                matrix[row] -= matrix[row, column] * matrix[pivot_row]
        matrix[np.abs(matrix) <= tol] = 0.0
        pivots.append(column)
        pivot_row += 1

    return matrix, pivots


def rank(A: ArrayLike, tol: float = 1e-10) -> int:
    """Return the numerical rank of A using the pivot tolerance."""
    return len(rref(A, tol)[1])


def null_space_basis(A: ArrayLike, tol: float = 1e-10) -> NDArray[np.float64]:
    """Return one row-vector basis element for each free variable."""
    matrix = np.asarray(A, dtype=float)
    reduced, pivots = rref(matrix, tol)
    free_columns = [column for column in range(matrix.shape[1]) if column not in pivots]
    basis = np.zeros((len(free_columns), matrix.shape[1]), dtype=float)
    for basis_row, free_column in enumerate(free_columns):
        basis[basis_row, free_column] = 1.0
        for pivot_row, pivot_column in enumerate(pivots):
            basis[basis_row, pivot_column] = -reduced[pivot_row, free_column]
    return basis


def col_space_basis(A: ArrayLike, tol: float = 1e-10) -> NDArray[np.float64]:
    """Return the pivot columns of the original matrix as a column-space basis."""
    matrix = np.asarray(A, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    pivots = rref(matrix, tol)[1]
    return matrix[:, pivots].copy()


if __name__ == "__main__":
    example = np.array([[1, 2, -1], [3, -1, 4]], dtype=float)
    reduced, pivot_columns = rref(example)
    print("RREF:\n", reduced)
    print("Pivot columns:", pivot_columns)

    rng = np.random.default_rng(42)
    for sample in range(20):
        matrix = rng.normal(size=(5, 7))
        null_basis = null_space_basis(matrix)
        assert rank(matrix) + len(null_basis) == 7
        assert np.allclose(matrix @ null_basis.T, 0, atol=1e-9)
        assert rank(matrix) == np.linalg.matrix_rank(matrix)

    left = rng.normal(size=(6, 4))
    right = rng.normal(size=(4, 8))
    rank_four = left @ right
    assert rank(rank_four) == 4
    assert null_space_basis(rank_four).shape == (4, 8)
