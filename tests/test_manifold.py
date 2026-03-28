import numpy as np

from autotune.manifold import ManifoldNavigator


class TestManifoldUpdate:
    def test_learns_covariance_from_points(self):
        nav = ManifoldNavigator(
            param_names=("lr", "rank"),
            param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
        )
        points = [
            {"lr": 1e-4, "rank": 8},
            {"lr": 2e-4, "rank": 16},
            {"lr": 3e-4, "rank": 24},
            {"lr": 4e-4, "rank": 32},
            {"lr": 5e-4, "rank": 40},
            {"lr": 6e-4, "rank": 48},
            {"lr": 7e-4, "rank": 56},
            {"lr": 8e-4, "rank": 60},
            {"lr": 9e-4, "rank": 62},
            {"lr": 1e-3, "rank": 64},
        ]
        objectives = [(0.6 + i * 0.03,) for i in range(10)]
        nav.update(points, objectives)
        state = nav.state()
        assert state.effective_dim >= 1
        assert state.eigenvalues[0] > state.eigenvalues[1] * 2

    def test_not_enough_points_stays_generation_0(self):
        nav = ManifoldNavigator(
            param_names=("lr", "rank"),
            param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
        )
        nav.update(
            [{"lr": 2e-4, "rank": 16}],
            [(0.7,)],
        )
        assert nav.state().generation == 0


class TestManifoldProposals:
    def _make_trained_nav(self) -> ManifoldNavigator:
        nav = ManifoldNavigator(
            param_names=("lr", "rank"),
            param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
        )
        points = [{"lr": i * 1e-4, "rank": 8 + i * 6} for i in range(1, 11)]
        objectives = [(0.5 + i * 0.04,) for i in range(10)]
        nav.update(points, objectives)
        return nav

    def test_propose_direction_within_bounds(self):
        nav = self._make_trained_nav()
        from_theta = {"lr": 5e-4, "rank": 32}
        result = nav.propose_direction(from_theta)
        assert 1e-5 <= result["lr"] <= 1e-3
        assert 4 <= result["rank"] <= 64

    def test_propose_candidates_returns_n(self):
        nav = self._make_trained_nav()
        candidates = nav.propose_candidates({"lr": 5e-4, "rank": 32}, n=5)
        assert len(candidates) == 5
        for c in candidates:
            assert "lr" in c
            assert "rank" in c


class TestManifoldPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        nav = ManifoldNavigator(
            param_names=("lr", "rank"),
            param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
        )
        points = [{"lr": i * 1e-4, "rank": 8 + i * 6} for i in range(1, 11)]
        objectives = [(0.5 + i * 0.04,) for i in range(10)]
        nav.update(points, objectives)
        path = tmp_path / "manifold_state.json"
        nav.save(path)
        loaded = ManifoldNavigator.load(path)
        assert loaded.state().generation == nav.state().generation
        assert loaded.state().effective_dim == nav.state().effective_dim


class TestStepSizeAdaptation:
    """Regression: step_sizes must adapt after manifold updates."""

    def test_step_sizes_adapt_after_update(self):
        nav = ManifoldNavigator(
            param_names=("a", "b"),
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        initial_steps = nav._step_sizes.copy()

        # First update: strong positive correlation
        points1 = [{"a": 0.1 * i, "b": 0.1 * i} for i in range(10)]
        objectives1 = [(0.5 + 0.05 * i,) for i in range(10)]
        nav.update(points1, objectives1)
        after_first = nav._step_sizes.copy()

        # Second update: shift the correlation pattern
        points2 = [{"a": 0.1 * i, "b": 0.9 - 0.05 * i} for i in range(10)]
        objectives2 = [(0.8 + 0.02 * i,) for i in range(10)]
        nav.update(points2, objectives2)
        after_second = nav._step_sizes.copy()

        # Step sizes must have changed from initial
        assert not np.array_equal(after_second, initial_steps), (
            "Step sizes were never adapted from initial values"
        )
        # All step sizes within valid range
        assert np.all(after_second >= 0.01)
        assert np.all(after_second <= 1.0)


class TestEigenvectorPersistence:
    """Regression: save/load must preserve ALL eigenvector columns."""

    def test_save_load_preserves_all_eigenvectors(self, tmp_path):
        nav = ManifoldNavigator(
            param_names=("a", "b", "c"),
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)},
        )
        # Create correlated data so effective_dim < 3
        points = [
            {"a": 0.1 * i, "b": 0.1 * i + 0.01 * (i % 3), "c": 0.5}
            for i in range(10)
        ]
        objectives = [(0.5 + 0.05 * i,) for i in range(10)]
        nav.update(points, objectives)

        original_eigenvectors = nav._eigenvectors.copy()
        assert nav._effective_dim < 3, "Test setup: need effective_dim < d for this test"

        path = tmp_path / "manifold.json"
        nav.save(path)
        loaded = ManifoldNavigator.load(path)

        np.testing.assert_array_almost_equal(
            loaded._eigenvectors,
            original_eigenvectors,
            decimal=10,
            err_msg="Eigenvectors not fully preserved after save/load round-trip",
        )
