from autotune.dag import ExperimentDAG
from autotune.forgetting import ForgettingDetector
from autotune.manifold import ManifoldNavigator
from autotune.navigator import StrategyNavigator
from autotune.types import ScoringConfig


def _setup(tmp_path):
    scoring = ScoringConfig(0.005, 0.01, -0.10)
    dag = ExperimentDAG.create(tmp_path / "dag.json", scoring, objective_count=3)
    manifold = ManifoldNavigator(
        param_names=("lr", "rank"),
        param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
    )
    forgetting = ForgettingDetector(cliff_threshold=-0.10)
    return dag, manifold, forgetting


class TestSuggestMoves:
    def test_suggests_branch_from_kept_nodes(self, tmp_path):
        dag, manifold, forgetting = _setup(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4, "rank": 16},
            description="base",
            techniques=(),
            composite=0.72,
            task_score=0.72,
            retention_score=0.95,
            cost=1.0,
        )
        dag.recalculate_potential()
        nav = StrategyNavigator(dag, manifold, forgetting)
        moves = nav.suggest_moves(n=3)
        assert len(moves) >= 1
        assert moves[0].move_type == "branch"
        assert moves[0].parent_ids == ("node_0001",)

    def test_suggests_merge_when_two_kept_nodes(self, tmp_path):
        dag, manifold, forgetting = _setup(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4, "rank": 16},
            description="A",
            techniques=(),
            composite=0.8,
            task_score=0.85,
            retention_score=0.7,
            cost=1.0,
        )
        dag.add_seed(
            theta={"lr": 5e-4, "rank": 32},
            description="B",
            techniques=(),
            composite=0.75,
            task_score=0.7,
            retention_score=0.95,
            cost=1.5,
        )
        dag.recalculate_potential()
        nav = StrategyNavigator(dag, manifold, forgetting)
        moves = nav.suggest_moves(n=5)
        move_types = [m.move_type for m in moves]
        assert "merge" in move_types


class TestExplainMethods:
    def test_explain_manifold_returns_string(self, tmp_path):
        dag, manifold, forgetting = _setup(tmp_path)
        nav = StrategyNavigator(dag, manifold, forgetting)
        result = nav.explain_manifold()
        assert isinstance(result, str)

    def test_explain_frontier_returns_string(self, tmp_path):
        dag, manifold, forgetting = _setup(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4, "rank": 16},
            description="base",
            techniques=(),
            composite=0.72,
            task_score=0.72,
            retention_score=0.95,
            cost=1.0,
        )
        dag.rerank_pareto()
        nav = StrategyNavigator(dag, manifold, forgetting)
        result = nav.explain_frontier()
        assert "node_0001" in result
