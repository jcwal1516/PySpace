"""Typed results and the deterministic one-dimensional SOM implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SOMResult:
    """Observable state from a fitted one-dimensional SOM."""

    weights: np.ndarray
    node_assignments: np.ndarray
    quantization_errors: np.ndarray
    training_history: dict[str, list[float]]
    grid_size: int
    input_dimensions: int


@dataclass(frozen=True)
class PatternResult:
    """Python-friendly pattern result layered on the parity table."""

    som_result: SOMResult
    covariation_data: pd.DataFrame
    enrichment_scores: np.ndarray
    variable_names: list[str]
    significance_threshold: float = 0.95


class SelfOrganizingMap:
    """Small 1D Gaussian-neighborhood Kohonen map with local RNG ownership."""

    def __init__(
        self,
        grid_size: int = 1000,
        initial_learning_rate: float = 0.05,
        final_learning_rate: float = 0.01,
        max_iterations: int = 50,
        topology: str = "linear",
        random_seed: int | None = None,
    ) -> None:
        if grid_size <= 0 or max_iterations <= 0:
            raise ValueError("grid_size and max_iterations must be positive")
        if initial_learning_rate <= 0 or final_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if topology not in {"linear", "toroidal"}:
            raise ValueError("topology must be 'linear' or 'toroidal'")
        self.grid_size = int(grid_size)
        self.input_dim: int | None = None
        self.initial_learning_rate = float(initial_learning_rate)
        self.final_learning_rate = float(final_learning_rate)
        self.max_iterations = int(max_iterations)
        self.topology = topology
        self._rng = np.random.default_rng(random_seed)
        self.weights: np.ndarray | None = None
        self.training_history: dict[str, list[float]] = {}

    @staticmethod
    def _data(value: np.ndarray) -> np.ndarray:
        data = np.asarray(value, dtype=float)
        if data.ndim != 2 or len(data) == 0:
            raise ValueError("Input data must be a non-empty two-dimensional array")
        if np.any(~np.isfinite(data)):
            raise ValueError("Input data must contain only finite values")
        return data

    def _assign(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.weights is None:
            raise RuntimeError("SOM has not been fitted")
        distances = np.linalg.norm(data[:, None, :] - self.weights[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        errors = distances[np.arange(len(data)), assignments]
        return assignments, errors

    def fit(
        self,
        input_data: np.ndarray,
        verbose: bool = False,
        *,
        initial_weights: np.ndarray | None = None,
        epoch_orders: Sequence[np.ndarray] | None = None,
    ) -> SelfOrganizingMap:
        """Fit the map; explicit weights/orders enable shared random plans."""
        data = self._data(input_data)
        self.grid_size = min(self.grid_size, len(data))
        self.input_dim = data.shape[1]
        if initial_weights is None:
            spread = np.std(data, axis=0)
            spread[spread == 0] = 1
            self.weights = self._rng.normal(
                np.mean(data, axis=0),
                0.1 * spread,
                size=(self.grid_size, self.input_dim),
            )
        else:
            weights = np.asarray(initial_weights, dtype=float)
            if weights.shape != (self.grid_size, self.input_dim) or np.any(~np.isfinite(weights)):
                raise ValueError("initial_weights must have shape (grid_size, input_dimensions)")
            self.weights = weights.copy()
        if epoch_orders is not None and len(epoch_orders) != self.max_iterations:
            raise ValueError("epoch_orders must contain one permutation per iteration")

        history: dict[str, list[float]] = {
            "quantization_error": [],
            "learning_rate": [],
            "neighborhood_radius": [],
        }
        node_indices = np.arange(self.grid_size)
        initial_radius = max(self.grid_size / 4, 1.0)
        for iteration in range(self.max_iterations):
            progress = iteration / self.max_iterations
            learning_rate = self.initial_learning_rate * math.exp(
                -progress * math.log(self.initial_learning_rate / self.final_learning_rate)
            )
            radius = max(initial_radius * (1 - progress), 1.0)
            order = self._rng.permutation(len(data)) if epoch_orders is None else np.asarray(epoch_orders[iteration])
            if sorted(order.tolist()) != list(range(len(data))):
                raise ValueError("Each epoch order must be a zero-based permutation of the observations")
            for observation in data[order]:
                assignments, _ = self._assign(observation[None, :])
                best = int(assignments[0])
                offsets = np.abs(node_indices - best)
                if self.topology == "toroidal":
                    offsets = np.minimum(offsets, self.grid_size - offsets)
                influence = np.exp(-(offsets**2) / (2 * radius**2))
                self.weights += learning_rate * influence[:, None] * (observation - self.weights)
            _, errors = self._assign(data)
            history["quantization_error"].append(float(np.mean(errors)))
            history["learning_rate"].append(float(learning_rate))
            history["neighborhood_radius"].append(float(radius))
            if verbose:
                print(f"SOM iteration {iteration + 1}/{self.max_iterations}: error={np.mean(errors):.6g}")
        self.training_history = history
        return self

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        data = self._data(input_data)
        if self.input_dim != data.shape[1]:
            raise ValueError("Input dimensions do not match the fitted SOM")
        return self._assign(data)[0]

    def get_quantization_errors(self, input_data: np.ndarray) -> np.ndarray:
        data = self._data(input_data)
        if self.input_dim != data.shape[1]:
            raise ValueError("Input dimensions do not match the fitted SOM")
        return self._assign(data)[1]


__all__ = ["PatternResult", "SOMResult", "SelfOrganizingMap"]
