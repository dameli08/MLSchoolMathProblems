"""Plot planar linear transformations and verify signed polygon area (P6)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray


F_VERTICES = np.array(
    [
        [0.2, 0.1],
        [0.4, 0.1],
        [0.4, 0.45],
        [0.65, 0.45],
        [0.65, 0.6],
        [0.4, 0.6],
        [0.4, 0.75],
        [0.8, 0.75],
        [0.8, 0.9],
        [0.2, 0.9],
    ],
    dtype=float,
)
UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def _validated_matrix(A: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(A, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("A must be a 2 by 2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("A must contain only finite values")
    return matrix


def signed_area(polygon: ArrayLike) -> float:
    """Return a polygon's signed area using the shoelace formula."""
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("polygon must have shape (number_of_vertices >= 3, 2)")
    if not np.isfinite(points).all():
        raise ValueError("polygon must contain only finite coordinates")
    following = np.roll(points, -1, axis=0)
    return float(
        0.5
        * np.sum(
            points[:, 0] * following[:, 1]
            - following[:, 0] * points[:, 1]
        )
    )


def _draw_grid(ax: Axes, transform: NDArray[np.float64]) -> None:
    for coordinate in np.linspace(0.0, 1.0, 6):
        vertical = np.array([[coordinate, 0.0], [coordinate, 1.0]]) @ transform.T
        horizontal = np.array([[0.0, coordinate], [1.0, coordinate]]) @ transform.T
        ax.plot(*vertical.T, color="0.78", linewidth=0.7, zorder=0)
        ax.plot(*horizontal.T, color="0.78", linewidth=0.7, zorder=0)
    square = UNIT_SQUARE @ transform.T
    square = np.vstack((square, square[0]))
    ax.plot(*square.T, color="black", linewidth=1.2, zorder=1)


def _draw_letter(ax: Axes, transform: NDArray[np.float64]) -> None:
    letter = F_VERTICES @ transform.T
    closed_letter = np.vstack((letter, letter[0]))
    ax.fill(*letter.T, color="#6baed6", alpha=0.6, zorder=2)
    ax.plot(*closed_letter.T, color="#08519c", linewidth=1.8, zorder=3)


def _set_bounds(ax: Axes, transform: NDArray[np.float64], padding: float = 0.12) -> None:
    corners = np.vstack((UNIT_SQUARE, F_VERTICES)) @ transform.T
    low = corners.min(axis=0)
    high = corners.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    ax.set_xlim(low[0] - padding * span[0], high[0] + padding * span[0])
    ax.set_ylim(low[1] - padding * span[1], high[1] + padding * span[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def plot_transformation(
    A: ArrayLike, axes: Sequence[Axes] | None = None
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot the unit square, grid and F before and after applying A."""
    matrix = _validated_matrix(A)
    if axes is None:
        figure, axis_array = plt.subplots(1, 2, figsize=(9, 4))
    else:
        axis_array = np.asarray(axes, dtype=object)
        if axis_array.shape != (2,):
            raise ValueError("axes must contain exactly two Matplotlib axes")
        figure = axis_array[0].figure

    for ax, transform, title in zip(
        axis_array, (np.eye(2), matrix), ("Original", "Transformed")
    ):
        _draw_grid(ax, transform)
        _draw_letter(ax, transform)
        _set_bounds(ax, transform)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    figure.tight_layout()
    return figure, axis_array


def plot_four_transformations(
    matrices: Sequence[ArrayLike],
) -> tuple[Figure, NDArray[np.object_]]:
    """Display four transformations in a two-row original/image panel."""
    if len(matrices) != 4:
        raise ValueError("exactly four matrices are required")
    validated = [_validated_matrix(matrix) for matrix in matrices]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for column, matrix in enumerate(validated):
        plot_transformation(matrix, axes[:, column])
        axes[0, column].set_title(f"Original {column + 1}")
        axes[1, column].set_title(f"Image {column + 1}")
    figure.tight_layout()
    return figure, axes


if __name__ == "__main__":
    examples = [
        np.eye(2),
        np.array([[1.0, 0.5], [0.0, 1.0]]),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[1.0, 2.0], [2.0, 4.0]]),
    ]
    plot_four_transformations(examples)
    plt.show()

    area_test_matrices = examples + [np.array([[3.0, 1.0], [1.0, 2.0]])]
    square_area = signed_area(UNIT_SQUARE)
    for matrix in area_test_matrices:
        image_area = signed_area(UNIT_SQUARE @ matrix.T)
        assert np.isclose(image_area, np.linalg.det(matrix))
        assert np.isclose(image_area, square_area * np.linalg.det(matrix))
