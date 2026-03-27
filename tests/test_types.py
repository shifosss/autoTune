import pytest
from autotune.types import (
    DAGNode,
    ScoringConfig,
    DangerZone,
    MoveProposal,
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
