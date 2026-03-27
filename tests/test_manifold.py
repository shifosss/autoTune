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
