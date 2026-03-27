import pytest
from autotune.types import (
    CliffReport,
    DAGMeta,
    DAGNode,
    DangerZone,
    ManifoldState,
    MoveProposal,
    ScoringConfig,
    TechniqueStats,
)


class TestDAGNode:
    def test_create_seed_node(self):
        node = DAGNode(
            id="node_0001",
            parent_ids=(),
            move_type="seed",
            merge_strategy=None,
            merge_weights=None,
            theta={"lr": 2e-4, "rank": 16},
            manifold_coords=None,
            strategy_desc="baseline",
            technique_tags=(),
            task_score=0.72,
            retention_score=0.98,
            cost=1.5,
            composite=0.72,
            checkpoint_id="node_0001",
            depth=0,
            potential=0.0,
            pareto_rank=0,
            node_type="executed",
            status="keep",
            created_at="2026-03-27T00:00:00",
        )
        assert node.id == "node_0001"
        assert node.parent_ids == ()
        assert node.move_type == "seed"

    def test_create_merge_node(self):
        node = DAGNode(
            id="node_0010",
            parent_ids=("node_0005", "node_0008"),
            move_type="merge",
            merge_strategy="weighted_avg",
            merge_weights=(0.6, 0.4),
            theta={"lr": 3e-4, "rank": 24},
            manifold_coords=(0.5, -0.2),
            strategy_desc="merge task-strong with retention-strong",
            technique_tags=("merge_weighted",),
            task_score=None,
            retention_score=None,
            cost=None,
            composite=None,
            checkpoint_id=None,
            depth=3,
            potential=0.0,
            pareto_rank=None,
            node_type="proposed",
            status="pending",
            created_at="2026-03-27T01:00:00",
        )
        assert node.parent_ids == ("node_0005", "node_0008")
        assert node.merge_weights == (0.6, 0.4)

    def test_node_is_frozen(self):
        node = DAGNode(
            id="node_0001",
            parent_ids=(),
            move_type="seed",
            merge_strategy=None,
            merge_weights=None,
            theta={"lr": 2e-4},
            manifold_coords=None,
            strategy_desc="baseline",
            technique_tags=(),
            task_score=0.5,
            retention_score=None,
            cost=1.0,
            composite=0.5,
            checkpoint_id=None,
            depth=0,
            potential=0.0,
            pareto_rank=0,
            node_type="executed",
            status="keep",
            created_at="2026-03-27T00:00:00",
        )
        with pytest.raises(AttributeError):
            node.status = "discard"


class TestScoringConfig:
    def test_defaults(self):
        sc = ScoringConfig(
            exploration_weight=0.005,
            diversity_weight=0.01,
            retention_cliff_threshold=-0.10,
        )
        assert sc.exploration_weight == 0.005


class TestDangerZone:
    def test_create(self):
        dz = DangerZone(
            center=(0.5, -0.3),
            radius=0.15,
            cause_technique="lr_increase",
            severity=0.12,
        )
        assert dz.cause_technique == "lr_increase"


class TestMoveProposal:
    def test_branch_proposal(self):
        mp = MoveProposal(
            move_type="branch",
            parent_ids=("node_0005",),
            suggested_theta={"lr": 4.2e-4, "rank": 28},
            merge_strategy=None,
            merge_weights=None,
            direction_label="PC1 (+capacity)",
            score=0.847,
            reasoning="high retention budget, PC1 consistently improves task",
        )
        assert mp.move_type == "branch"
        assert mp.score == 0.847


class TestDAGMeta:
    def test_create(self):
        sc = ScoringConfig(0.005, 0.01, -0.10)
        meta = DAGMeta(
            next_id=5,
            total_executed=3,
            total_proposed=1,
            best_composite=0.85,
            best_node_id="node_0002",
            objective_count=3,
            scoring=sc,
        )
        assert meta.next_id == 5
        assert meta.scoring.exploration_weight == 0.005

    def test_is_frozen(self):
        sc = ScoringConfig(0.005, 0.01, -0.10)
        meta = DAGMeta(1, 0, 0, None, None, 3, sc)
        with pytest.raises(AttributeError):
            meta.next_id = 10


class TestTechniqueStats:
    def test_create(self):
        ts = TechniqueStats(
            times_tried=5,
            avg_task_delta=0.03,
            avg_retention_delta=-0.02,
            best_parent_delta=0.08,
        )
        assert ts.times_tried == 5
        assert ts.avg_retention_delta == -0.02

    def test_is_frozen(self):
        ts = TechniqueStats(1, 0.0, 0.0, 0.0)
        with pytest.raises(AttributeError):
            ts.times_tried = 10


class TestManifoldState:
    def test_create(self):
        ms = ManifoldState(
            param_names=("lr", "rank"),
            covariance=((1.0, 0.0), (0.0, 1.0)),
            principal_components=((1.0, 0.0), (0.0, 1.0)),
            eigenvalues=(1.0, 0.5),
            effective_dim=2,
            mean=(0.5, 0.5),
            step_sizes=(0.1, 0.1),
            generation=1,
        )
        assert ms.param_names == ("lr", "rank")
        assert ms.effective_dim == 2

    def test_is_frozen(self):
        ms = ManifoldState(("lr",), ((1.0,),), ((1.0,),), (1.0,), 1, (0.5,), (0.1,), 0)
        with pytest.raises(AttributeError):
            ms.generation = 5


class TestCliffReport:
    def test_create(self):
        dz = DangerZone(center=(0.5,), radius=0.1, cause_technique="lr_increase", severity=0.15)
        cr = CliffReport(
            node_id="node_0005",
            parent_ids=("node_0003",),
            delta_retention=-0.15,
            cause_techniques=("lr_increase",),
            severity=0.15,
            danger_zone=dz,
        )
        assert cr.node_id == "node_0005"
        assert cr.delta_retention == -0.15
        assert cr.danger_zone.cause_technique == "lr_increase"

    def test_is_frozen(self):
        dz = DangerZone(center=(0.5,), radius=0.1, cause_technique="x", severity=0.1)
        cr = CliffReport("n", (), -0.1, (), 0.1, dz)
        with pytest.raises(AttributeError):
            cr.severity = 0.5
