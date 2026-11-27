"""Dual bases and polynomial interpolation (homework P12)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray


def dual_basis(P: ArrayLike) -> NDArray[np.float64]:
    """Return the dual basis vectors as rows for the columns of invertible P."""
    matrix = np.asarray(P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("P must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("P must contain only finite values")
    return np.linalg.inv(matrix)


def vandermonde(nodes: ArrayLike) -> NDArray[np.float64]:
    """Build V[k, j] = nodes[k]**j for monomial coefficient vectors."""
    points = np.asarray(nodes, dtype=float)
    if points.ndim != 1 or points.size == 0:
        raise ValueError("nodes must be a nonempty one-dimensional array")
    if not np.isfinite(points).all():
        raise ValueError("nodes must contain only finite values")
    if np.unique(points).size != points.size:
        raise ValueError("interpolation nodes must be distinct")
    return np.vander(points, N=points.size, increasing=True)


def lagrange_coefficients(nodes: ArrayLike) -> NDArray[np.float64]:
    """Return one monomial coefficient vector per Lagrange basis polynomial."""
    matrix = vandermonde(nodes)
    # V maps monomial coefficients to node values, so V^{-1}'s columns
    # are the coefficients of the polynomials that interpolate the unit vectors.
    return np.linalg.inv(matrix)


def interpolate(
    nodes: ArrayLike, values: ArrayLike, points: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate the unique degree-at-most-n-1 interpolating polynomial."""
    nodes_array = np.asarray(nodes, dtype=float)
    values_array = np.asarray(values, dtype=float)
    query = np.asarray(points, dtype=float)
    matrix = vandermonde(nodes_array)
    if values_array.shape != nodes_array.shape:
        raise ValueError("values must have the same shape as nodes")
    if not np.isfinite(values_array).all() or not np.isfinite(query).all():
        raise ValueError("values and evaluation points must be finite")
    coefficients = np.linalg.solve(matrix, values_array)
    return np.polynomial.polynomial.polyval(query, coefficients)


def runge_function(points: ArrayLike) -> NDArray[np.float64]:
    points_array = np.asarray(points, dtype=float)
    return 1.0 / (1.0 + points_array**2)


def plot_interpolation_examples() -> tuple[Figure, NDArray[np.object_]]:
    """Plot the two equally spaced-node examples, including Runge oscillations."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    nodes = np.linspace(-1.0, 1.0, 9)
    values = np.sin(3.0 * nodes)
    dense = np.linspace(-1.0, 1.0, 600)
    axes[0].plot(dense, np.sin(3.0 * dense), label="sin(3t)")
    axes[0].plot(dense, interpolate(nodes, values, dense), "--", label="interpolant")
    axes[0].plot(nodes, values, "o", label="nodes")
    axes[0].set_title("9 equally spaced nodes on [-1, 1]")
    axes[0].legend()

    nodes = np.linspace(-5.0, 5.0, 21)
    values = runge_function(nodes)
    dense = np.linspace(-5.0, 5.0, 1000)
    axes[1].plot(dense, runge_function(dense), label="1 / (1 + t²)")
    axes[1].plot(dense, interpolate(nodes, values, dense), "--", label="interpolant")
    axes[1].plot(nodes, values, "o", label="nodes")
    axes[1].set_title("21 equally spaced nodes on [-5, 5]")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure, axes


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(5, 5))
    while abs(np.linalg.det(matrix)) < 1e-8:
        matrix = rng.normal(size=(5, 5))
    dual_rows = dual_basis(matrix)
    assert np.allclose(dual_rows @ matrix, np.eye(5))

    nodes = np.linspace(-1.0, 1.0, 4)
    lagrange = lagrange_coefficients(nodes)
    assert np.allclose(vandermonde(nodes) @ lagrange, np.eye(4))
    plot_interpolation_examples()
    plt.show()
