import json
from pathlib import Path

import pytest
from autotune.dag import ExperimentDAG
from autotune.types import ScoringConfig


def _make_dag(tmp_path: Path) -> ExperimentDAG:
    path = tmp_path / "dag.json"
    scoring = ScoringConfig(
        exploration_weight=0.005,
        diversity_weight=0.01,
        retention_cliff_threshold=-0.10,
    )
    return ExperimentDAG.create(path, scoring, objective_count=3)


class TestNodeCRUD:
    def test_add_seed(self, tmp_path):
        dag = _make_dag(tmp_path)
        node = dag.add_seed(
            theta={"lr": 2e-4, "rank": 16},
            description="baseline",
            techniques=("baseline",),
            composite=0.72,
            task_score=0.72,
            retention_score=0.98,
            cost=1.5,
        )
        assert node.id == "node_0001"
        assert node.move_type == "seed"
        assert node.status == "keep"
        assert node.parent_ids == ()

    def test_add_proposed_branch(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        node = dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="increase lr",
            techniques=("lr_increase",),
            theta={"lr": 5e-4},
        )
        assert node.id == "node_0002"
        assert node.node_type == "proposed"
        assert node.status == "pending"
        assert node.parent_ids == ("node_0001",)

    def test_add_proposed_merge(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4},
            description="a",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_seed(
            theta={"lr": 5e-4},
            description="b",
            techniques=(),
            composite=0.8,
            task_score=0.8,
            retention_score=0.8,
            cost=1.2,
        )
        node = dag.add_proposed(
            parent_ids=("node_0001", "node_0002"),
            move_type="merge",
            description="merge a+b",
            techniques=("merge_weighted",),
            theta={"lr": 3.5e-4},
            merge_strategy="weighted_avg",
            merge_weights=(0.5, 0.5),
        )
        assert node.move_type == "merge"
        assert node.parent_ids == ("node_0001", "node_0002")
        assert node.merge_strategy == "weighted_avg"

    def test_get_node(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        assert dag.get_node("node_0001") is not None
        assert dag.get_node("nonexistent") is None

    def test_parents_and_children(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="child",
            techniques=(),
            theta={"lr": 3e-4},
        )
        parents = dag.parents("node_0002")
        assert len(parents) == 1
        assert parents[0].id == "node_0001"
        children = dag.children("node_0001")
        assert len(children) == 1
        assert children[0].id == "node_0002"

    def test_mark_running(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="test",
            techniques=(),
            theta={},
        )
        node = dag.mark_running("node_0002")
        assert node.status == "running"

    def test_promote(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="test",
            techniques=(),
            theta={},
        )
        dag.mark_running("node_0002")
        node = dag.promote(
            "node_0002",
            {
                "task_score": 0.85,
                "retention_score": 0.88,
                "cost": 1.2,
                "composite": 0.85,
            },
        )
        assert node.node_type == "executed"
        assert node.composite == 0.85

    def test_mark_failed(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="test",
            techniques=(),
            theta={},
        )
        dag.mark_running("node_0002")
        node = dag.mark_failed("node_0002", "oom")
        assert node.status == "oom"

    def test_cancel(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="test",
            techniques=(),
            theta={},
        )
        node = dag.cancel("node_0002")
        assert node.status == "cancelled"

    def test_ancestors(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="root",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="child",
            techniques=(),
            theta={},
        )
        dag.mark_running("node_0002")
        dag.promote("node_0002", {"task_score": 0.8, "composite": 0.8, "cost": 1.0})
        dag.add_proposed(
            parent_ids=("node_0002",),
            move_type="branch",
            description="grandchild",
            techniques=(),
            theta={},
        )
        ancestors = dag.ancestors("node_0003")
        ancestor_ids = {a.id for a in ancestors}
        assert "node_0001" in ancestor_ids
        assert "node_0002" in ancestor_ids

    def test_technique_stats(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.update_technique_stats("lr_increase", task_delta=0.05, retention_delta=-0.02)
        dag.update_technique_stats("lr_increase", task_delta=0.03, retention_delta=-0.01)
        ts = dag.technique_stats("lr_increase")
        assert ts.times_tried == 2
        assert ts.avg_task_delta == pytest.approx(0.04)
        assert ts.best_parent_delta == pytest.approx(0.05)

    def test_technique_stats_unknown(self, tmp_path):
        dag = _make_dag(tmp_path)
        ts = dag.technique_stats("nonexistent")
        assert ts.times_tried == 0


class TestParetoRanking:
    def test_non_dominated_sorting_3obj(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="A",
            techniques=(),
            composite=0.9,
            task_score=0.9,
            retention_score=0.7,
            cost=2.0,
        )
        dag.add_seed(
            theta={},
            description="B",
            techniques=(),
            composite=0.8,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_seed(
            theta={},
            description="C",
            techniques=(),
            composite=0.6,
            task_score=0.6,
            retention_score=0.6,
            cost=3.0,
        )
        dag.rerank_pareto()
        a = dag.get_node("node_0001")
        b = dag.get_node("node_0002")
        c = dag.get_node("node_0003")
        assert a.pareto_rank == 0
        assert b.pareto_rank == 0
        assert c.pareto_rank == 1
        assert a.status == "keep"
        assert b.status == "keep"
        assert c.status == "keep"  # rank 1 is still "keep"

    def test_rank_2_is_discard(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="best",
            techniques=(),
            composite=0.9,
            task_score=0.9,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_seed(
            theta={},
            description="mid",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.7,
            cost=2.0,
        )
        dag.add_seed(
            theta={},
            description="worst",
            techniques=(),
            composite=0.5,
            task_score=0.5,
            retention_score=0.5,
            cost=3.0,
        )
        dag.rerank_pareto()
        assert dag.get_node("node_0001").pareto_rank == 0
        assert dag.get_node("node_0002").pareto_rank == 1
        assert dag.get_node("node_0003").pareto_rank == 2
        assert dag.get_node("node_0003").status == "discard"

    def test_2obj_mode_no_retention(self, tmp_path):
        path = tmp_path / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(path, scoring, objective_count=2)
        dag.add_seed(
            theta={}, description="A", techniques=(), composite=0.9, task_score=0.9, cost=2.0
        )
        dag.add_seed(
            theta={}, description="B", techniques=(), composite=0.7, task_score=0.7, cost=1.0
        )
        dag.rerank_pareto()
        assert dag.get_node("node_0001").pareto_rank == 0
        assert dag.get_node("node_0002").pareto_rank == 0


class TestUCBScoring:
    def test_potential_increases_with_fewer_children(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4},
            description="A",
            techniques=(),
            composite=0.8,
            task_score=0.8,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_seed(
            theta={"lr": 3e-4},
            description="B",
            techniques=(),
            composite=0.8,
            task_score=0.8,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",), move_type="branch", description="c1", techniques=(), theta={}
        )
        dag.add_proposed(
            parent_ids=("node_0001",), move_type="branch", description="c2", techniques=(), theta={}
        )
        dag.recalculate_potential()
        a = dag.get_node("node_0001")
        b = dag.get_node("node_0002")
        assert b.potential > a.potential

    def test_retention_health_penalty(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="healthy",
            techniques=(),
            composite=0.8,
            task_score=0.8,
            retention_score=0.95,
            cost=1.0,
        )
        dag.add_seed(
            theta={},
            description="forgetting",
            techniques=(),
            composite=0.8,
            task_score=0.8,
            retention_score=0.05,
            cost=1.0,
        )
        dag.recalculate_potential()
        healthy = dag.get_node("node_0001")
        forgetting = dag.get_node("node_0002")
        assert healthy.potential > forgetting.potential


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4, "rank": 16},
            description="base",
            techniques=("baseline",),
            composite=0.72,
            task_score=0.72,
            retention_score=0.98,
            cost=1.5,
        )
        dag.save()
        loaded = ExperimentDAG.load(tmp_path / "dag.json")
        node = loaded.get_node("node_0001")
        assert node is not None
        assert node.theta == {"lr": 2e-4, "rank": 16}
        assert node.composite == 0.72

    def test_save_is_atomic(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="base",
            techniques=(),
            composite=0.5,
            task_score=0.5,
            retention_score=0.9,
            cost=1.0,
        )
        dag.save()
        path = tmp_path / "dag.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "nodes" in data
        assert "node_0001" in data["nodes"]


class TestReconciliation:
    def test_promote_from_completed_dir(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="test",
            techniques=("lr_increase",),
            theta={"lr": 5e-4},
        )
        dag.mark_running("node_0002")
        queue = tmp_path / "queue"
        running = tmp_path / "running"
        completed = tmp_path / "completed"
        archive = tmp_path / "archive"
        for d in (queue, running, completed, archive):
            d.mkdir()
        completion = {
            "node_id": "node_0002",
            "status": "completed",
            "metrics": {
                "task_score": 0.85,
                "retention_score": 0.88,
                "cost": 1.2,
                "composite": 0.85,
            },
        }
        (completed / "node_0002.json").write_text(json.dumps(completion))
        dag.reconcile(queue, running, completed, archive)
        node = dag.get_node("node_0002")
        assert node.node_type == "executed"
        assert node.task_score == 0.85
        assert node.composite == 0.85

    def test_stale_running_reset_to_pending(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="test",
            techniques=(),
            theta={},
        )
        dag.mark_running("node_0002")
        queue = tmp_path / "queue"
        running = tmp_path / "running"
        completed = tmp_path / "completed"
        archive = tmp_path / "archive"
        for d in (queue, running, completed, archive):
            d.mkdir()
        dag.reconcile(queue, running, completed, archive)
        node = dag.get_node("node_0002")
        assert node.status == "pending"

    def test_reconcile_archives_completion_files(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="base",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="test",
            techniques=("lr_increase",),
            theta={"lr": 5e-4},
        )
        dag.mark_running("node_0002")
        queue = tmp_path / "queue"
        running = tmp_path / "running"
        completed = tmp_path / "completed"
        archive = tmp_path / "archive"
        for d in (queue, running, completed, archive):
            d.mkdir()
        completion = {
            "node_id": "node_0002",
            "status": "completed",
            "metrics": {"task_score": 0.85, "composite": 0.85, "cost": 1.2},
        }
        (completed / "node_0002.json").write_text(json.dumps(completion))
        dag.reconcile(queue, running, completed, archive)
        # Completion file should be moved to archive
        assert not (completed / "node_0002.json").exists()
        assert (archive / "node_0002.json").exists()


class TestNewQueryMethods:
    def test_pareto_front_returns_rank_0(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="A",
            techniques=(),
            composite=0.9,
            task_score=0.9,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_seed(
            theta={},
            description="B",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.7,
            cost=2.0,
        )
        dag.add_seed(
            theta={},
            description="C",
            techniques=(),
            composite=0.5,
            task_score=0.5,
            retention_score=0.5,
            cost=3.0,
        )
        dag.rerank_pareto()
        front = dag.pareto_front(layer=0)
        assert len(front) == 1
        assert front[0].id == "node_0001"
        layer1 = dag.pareto_front(layer=1)
        assert len(layer1) == 1
        assert layer1[0].id == "node_0002"

    def test_lineage_single_path(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="root",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="child",
            techniques=(),
            theta={},
        )
        dag.mark_running("node_0002")
        dag.promote("node_0002", {"task_score": 0.8, "composite": 0.8, "cost": 1.0})
        dag.add_proposed(
            parent_ids=("node_0002",),
            move_type="branch",
            description="grandchild",
            techniques=(),
            theta={},
        )
        paths = dag.lineage("node_0003")
        assert len(paths) == 1
        assert [n.id for n in paths[0]] == ["node_0001", "node_0002", "node_0003"]

    def test_lineage_merge_multiple_paths(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="A",
            techniques=(),
            composite=0.7,
            task_score=0.7,
            retention_score=0.9,
            cost=1.0,
        )
        dag.add_seed(
            theta={},
            description="B",
            techniques=(),
            composite=0.8,
            task_score=0.8,
            retention_score=0.8,
            cost=1.2,
        )
        dag.add_proposed(
            parent_ids=("node_0001", "node_0002"),
            move_type="merge",
            description="merge",
            techniques=(),
            theta={},
        )
        paths = dag.lineage("node_0003")
        assert len(paths) == 2  # one path through each parent

    def test_rank_proposals(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={},
            description="base",
            techniques=(),
            composite=0.8,
            task_score=0.8,
            retention_score=0.9,
            cost=1.0,
        )
        dag.recalculate_potential()
        dag.add_proposed(
            parent_ids=("node_0001",), move_type="branch", description="p1", techniques=(), theta={}
        )
        dag.add_proposed(
            parent_ids=("node_0001",), move_type="branch", description="p2", techniques=(), theta={}
        )
        ranked = dag.rank_proposals(n=5)
        assert len(ranked) == 2
        assert ranked[0].status == "pending"


class TestMarkFailedCounter:
    """Regression: mark_failed must not inflate total_executed (UCB denominator)."""

    def test_mark_failed_does_not_inflate_total_executed(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4, "rank": 16},
            description="baseline",
            techniques=("baseline",),
            composite=0.72,
            task_score=0.72,
            retention_score=0.98,
            cost=1.5,
        )
        assert dag.meta().total_executed == 1

        dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="test",
            techniques=("lr_increase",),
            theta={"lr": 5e-4, "rank": 16},
        )
        dag.mark_running("node_0002")
        dag.mark_failed("node_0002", "oom")

        # Failed node should NOT inflate total_executed
        assert dag.meta().total_executed == 1
