"""Manifold navigator: CMA-ES covariance learning and PC-based proposals."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from autotune.types import ManifoldState

logger = logging.getLogger(__name__)

_MIN_POINTS: int = 10
_VARIANCE_THRESHOLD: float = 0.95
_COVARIANCE_LR: float = 0.3


class ManifoldNavigator:
    """CMA-ES-inspired covariance learning for hyperparameter manifold navigation.

    Learns the correlation structure of high-performing hyperparameter points and
    proposes new candidates by moving along the learned principal components.
    Does NOT make strategy decisions — that is the responsibility of StrategyNavigator.

    Attributes:
        _param_names: Ordered parameter names.
        _param_bounds: Per-parameter (lo, hi) bounds used for normalization.
        _covariance: Current covariance matrix in normalized [0,1]^d space.
        _eigenvalues: Eigenvalues of the covariance, sorted descending.
        _eigenvectors: Corresponding eigenvectors (columns), sorted by eigenvalue.
        _mean: Current distribution mean in normalized space.
        _step_sizes: Per-parameter proposal step size.
        _effective_dim: Number of PCs capturing >= 95% of variance.
        _generation: Number of completed update steps.
    """

    def __init__(
        self,
        param_names: tuple[str, ...] | list[str],
        param_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Initialize a ManifoldNavigator with identity covariance.

        Args:
            param_names: Ordered sequence of hyperparameter names.
            param_bounds: Mapping from name to (lo, hi) float bounds.
        """
        self._param_names: tuple[str, ...] = tuple(param_names)
        self._param_bounds: dict[str, tuple[float, float]] = dict(param_bounds)
        d = len(self._param_names)
        self._covariance: np.ndarray = np.eye(d)
        self._eigenvalues: np.ndarray = np.ones(d)
        self._eigenvectors: np.ndarray = np.eye(d)
        self._mean: np.ndarray = np.full(d, 0.5)
        self._step_sizes: np.ndarray = np.full(d, 0.1)
        self._effective_dim: int = d
        self._generation: int = 0

    # ── normalization ──────────────────────────────────────────────────────────

    def _normalize(self, theta: dict[str, float]) -> np.ndarray:
        """Map theta from raw parameter space into normalized [0, 1]^d.

        Args:
            theta: Raw hyperparameter dict.

        Returns:
            1-D numpy array in [0, 1]^d.
        """
        result = np.zeros(len(self._param_names))
        for i, name in enumerate(self._param_names):
            lo, hi = self._param_bounds[name]
            val = theta.get(name, (lo + hi) / 2.0)
            result[i] = (val - lo) / (hi - lo) if hi > lo else 0.5
        return np.clip(result, 0.0, 1.0)

    def _denormalize(self, vec: np.ndarray) -> dict[str, float]:
        """Map a normalized vector back into raw parameter space.

        Args:
            vec: 1-D array in [0, 1]^d (clipped internally).

        Returns:
            Dict mapping each parameter name to its raw value.
        """
        result: dict[str, float] = {}
        for i, name in enumerate(self._param_names):
            lo, hi = self._param_bounds[name]
            result[name] = lo + float(np.clip(vec[i], 0.0, 1.0)) * (hi - lo)
        return result

    # ── learning ───────────────────────────────────────────────────────────────

    def update(
        self,
        points: list[dict[str, float]],
        objectives: list[tuple[float, ...]],
    ) -> ManifoldState:
        """Incorporate new evaluated points into the covariance estimate.

        Uses a CMA-ES-inspired weighted covariance update: the top mu = len/2
        points (ranked by first objective, higher is better) are combined with
        log-decay weights, and the running covariance is updated via an
        exponential moving average.

        If fewer than _MIN_POINTS are provided, the state is left unchanged
        (generation remains 0 until sufficient data is available).

        Args:
            points: List of hyperparameter dicts for evaluated experiments.
            objectives: Corresponding objective tuples; objectives[i][0] is the
                primary scalar used for ranking.

        Returns:
            The updated ManifoldState snapshot.
        """
        if len(points) < _MIN_POINTS:
            logger.debug("update skipped: %d points < minimum %d", len(points), _MIN_POINTS)
            return self.state()

        normed = np.array([self._normalize(p) for p in points])
        scores = np.array([obj[0] for obj in objectives])

        # Rank descending (higher score = better)
        ranked_idx = np.argsort(scores)[::-1]
        mu = max(len(ranked_idx) // 2, 2)
        top_idx = ranked_idx[:mu]

        # CMA-ES log-decay weights
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = raw_weights / raw_weights.sum()

        top_points = normed[top_idx]
        new_mean = weights @ top_points

        # Weighted covariance from deviations relative to the *old* mean
        diffs = top_points - self._mean
        weighted_outer: np.ndarray = sum(  # type: ignore[assignment]
            w * np.outer(d, d) for w, d in zip(weights, diffs)
        )

        if self._generation == 0:
            # First update: use data covariance directly instead of blending
            # with the identity prior, so the learned structure is not diluted.
            self._covariance = weighted_outer
        else:
            self._covariance = (
                1.0 - _COVARIANCE_LR
            ) * self._covariance + _COVARIANCE_LR * weighted_outer
        self._mean = new_mean

        # Eigendecomposition (eigh guarantees real, sorted ascending)
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance)
        sort_idx = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[sort_idx]
        self._eigenvectors = eigenvectors[:, sort_idx]

        # Effective dimensionality: smallest k s.t. sum(λ[:k]) / sum(λ) >= threshold
        total_var = float(self._eigenvalues.sum())
        if total_var > 0:
            cumulative = np.cumsum(self._eigenvalues) / total_var
            self._effective_dim = int(np.searchsorted(cumulative, _VARIANCE_THRESHOLD)) + 1
        else:
            self._effective_dim = len(self._param_names)

        self._generation += 1
        logger.debug(
            "manifold updated: generation=%d effective_dim=%d",
            self._generation,
            self._effective_dim,
        )
        return self.state()

    # ── proposals ─────────────────────────────────────────────────────────────

    def propose_direction(
        self,
        from_theta: dict[str, float],
        pc_index: int | None = None,
    ) -> dict[str, float]:
        """Propose a single new theta by stepping along a principal component.

        Before any update (generation == 0) a random unit-direction step is used.
        After training, a PC is selected proportionally to its eigenvalue and a
        fixed step is taken in that direction from from_theta.

        Args:
            from_theta: Starting hyperparameter point.
            pc_index: If provided, use this PC index instead of sampling.

        Returns:
            New hyperparameter dict within bounds.
        """
        from_norm = self._normalize(from_theta)

        if self._generation == 0:
            direction = np.random.randn(len(self._param_names))
            direction /= np.linalg.norm(direction)
            step = 0.1
        else:
            if pc_index is None:
                probs = self._eigenvalues[: self._effective_dim]
                probs = probs / probs.sum()
                pc_index = int(np.random.choice(self._effective_dim, p=probs))
            direction = self._eigenvectors[:, pc_index]
            step = float(self._step_sizes[pc_index])

        new_norm = np.clip(from_norm + step * direction, 0.0, 1.0)
        return self._denormalize(new_norm)

    def propose_candidates(
        self,
        from_theta: dict[str, float],
        n: int,
    ) -> list[dict[str, float]]:
        """Sample n candidate hyperparameter dicts from the learned distribution.

        Before any update, candidates are drawn uniformly. After training,
        candidates are sampled from a multivariate Gaussian centred at from_theta
        with covariance scaled to 1% of the learned covariance.

        Args:
            from_theta: Centre of the proposal distribution.
            n: Number of candidates to return.

        Returns:
            List of n hyperparameter dicts, all within bounds.
        """
        results: list[dict[str, float]] = []
        from_norm = self._normalize(from_theta)

        for _ in range(n):
            if self._generation == 0:
                sample = np.random.uniform(0.0, 1.0, len(self._param_names))
            else:
                sample = np.random.multivariate_normal(
                    from_norm,
                    self._covariance * 0.01,
                )
            sample = np.clip(sample, 0.0, 1.0)
            results.append(self._denormalize(sample))

        return results

    # ── queries ────────────────────────────────────────────────────────────────

    def project(self, theta: dict[str, float]) -> tuple[float, ...]:
        """Project theta onto the effective principal component subspace.

        Args:
            theta: Hyperparameter dict to project.

        Returns:
            Coordinates in the effective PC subspace (length == effective_dim).
        """
        normed = self._normalize(theta)
        centered = normed - self._mean
        coords = self._eigenvectors[:, : self._effective_dim].T @ centered
        return tuple(float(c) for c in coords)

    def effective_dimensionality(self) -> int:
        """Return the number of PCs capturing >= 95% of variance.

        Returns:
            Effective dimensionality as an integer.
        """
        return self._effective_dim

    def parameter_importance(self) -> dict[str, float]:
        """Compute each parameter's contribution to total variance.

        Importance is the eigenvalue-weighted sum of absolute PC loadings,
        normalised to sum to 1 across the effective subspace.

        Returns:
            Dict mapping parameter name to importance score in [0, 1].
        """
        total = float(self._eigenvalues.sum())
        if total == 0.0:
            uniform = 1.0 / len(self._param_names)
            return {name: uniform for name in self._param_names}

        result: dict[str, float] = {}
        for i, name in enumerate(self._param_names):
            importance = sum(
                float(self._eigenvalues[j]) * abs(float(self._eigenvectors[i, j]))
                for j in range(self._effective_dim)
            )
            result[name] = importance / total
        return result

    def principal_components(self) -> list[dict[str, float]]:
        """Return the effective principal components as parameter-keyed dicts.

        Returns:
            List of dicts, one per effective PC, mapping param name to loading.
        """
        pcs: list[dict[str, float]] = []
        for j in range(self._effective_dim):
            pc: dict[str, float] = {
                name: float(self._eigenvectors[i, j]) for i, name in enumerate(self._param_names)
            }
            pcs.append(pc)
        return pcs

    # ── state / persistence ────────────────────────────────────────────────────

    def state(self) -> ManifoldState:
        """Return a frozen snapshot of the current navigator state.

        Returns:
            ManifoldState dataclass with all current fields.
        """
        return ManifoldState(
            param_names=self._param_names,
            covariance=tuple(tuple(float(v) for v in row) for row in self._covariance),
            principal_components=tuple(
                tuple(float(v) for v in self._eigenvectors[:, j])
                for j in range(self._effective_dim)
            ),
            eigenvalues=tuple(float(v) for v in self._eigenvalues),
            effective_dim=self._effective_dim,
            mean=tuple(float(v) for v in self._mean),
            step_sizes=tuple(float(v) for v in self._step_sizes),
            generation=self._generation,
        )

    def save(self, path: Path) -> None:
        """Atomically write navigator state to a JSON file.

        Uses a write-then-rename pattern to prevent partial writes.

        Args:
            path: Destination path for the JSON state file.

        Raises:
            ValueError: If path is None.
            OSError: If the write or rename fails.
        """
        state = self.state()
        data = {
            "param_names": list(state.param_names),
            "param_bounds": {k: list(v) for k, v in self._param_bounds.items()},
            "covariance": [list(row) for row in state.covariance],
            "principal_components": [list(pc) for pc in state.principal_components],
            "eigenvalues": list(state.eigenvalues),
            "effective_dim": state.effective_dim,
            "mean": list(state.mean),
            "step_sizes": list(state.step_sizes),
            "generation": state.generation,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp, str(path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        logger.debug("manifold state saved to %s", path)

    @staticmethod
    def load(path: Path) -> "ManifoldNavigator":
        """Load a ManifoldNavigator from a previously saved JSON file.

        Args:
            path: Path to the JSON state file produced by save().

        Returns:
            A fully restored ManifoldNavigator instance.

        Raises:
            FileNotFoundError: If path does not exist.
            KeyError: If required fields are missing from the JSON.
        """
        with open(path) as f:
            data = json.load(f)

        nav = ManifoldNavigator(
            param_names=data["param_names"],
            param_bounds={k: tuple(v) for k, v in data["param_bounds"].items()},
        )
        nav._covariance = np.array(data["covariance"])
        nav._eigenvalues = np.array(data["eigenvalues"])

        d = len(data["param_names"])
        pcs = data["principal_components"]
        nav._eigenvectors = np.eye(d)
        for j, pc in enumerate(pcs):
            nav._eigenvectors[:, j] = pc

        nav._mean = np.array(data["mean"])
        nav._step_sizes = np.array(data["step_sizes"])
        nav._effective_dim = data["effective_dim"]
        nav._generation = data["generation"]

        logger.debug("manifold state loaded from %s (generation=%d)", path, nav._generation)
        return nav
