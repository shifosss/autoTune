"""End-to-end integration tests for the init → seed → suggest → reconcile flow."""

import json
import subprocess

import pytest
from click.testing import CliRunner

from autotune.cli import main
from autotune.dag import ExperimentDAG
from autotune.types import ScoringConfig


@pytest.fixture
def full_project(tmp_path):
    """Create a minimal git repo suitable for autoTune overlay initialization."""
    repo = tmp_path / "project"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    (repo / "train.py").write_text("print('train')")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return repo


class TestEndToEnd:
    def test_init_seed_suggest_flow(self, full_project, monkeypatch):
        """Full flow: init overlay, seed DAG, run status/suggest/explain/frontier."""
        monkeypatch.chdir(full_project)
        runner = CliRunner()

        # Init
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0

        # Create DAG with seed
        dag_path = full_project / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=3)
        dag.add_seed(
            theta={"lr": 2e-4, "rank": 16, "alpha": 32, "dropout": 0.1, "warmup_ratio": 0.05},
            description="baseline LoRA",
            techniques=("baseline",),
            composite=0.72,
            task_score=0.72,
            retention_score=0.95,
            cost=1.5,
        )
        dag.recalculate_potential()
        dag.rerank_pareto()
        dag.save()

        # Status
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "0.7200" in result.output

        # Suggest
        result = runner.invoke(main, ["suggest", "--n", "3"])
        assert result.exit_code == 0

        # Explain
        result = runner.invoke(main, ["explain"])
        assert result.exit_code == 0

        # Frontier
        result = runner.invoke(main, ["frontier"])
        assert result.exit_code == 0
        assert "node_0001" in result.output

    def test_reconcile_promotes_completed(self, full_project, monkeypatch):
        """Reconcile command promotes a completed node and updates DAG metrics."""
        monkeypatch.chdir(full_project)
        runner = CliRunner()
        runner.invoke(main, ["init"])

        dag_path = full_project / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=3)
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
        dag.save()

        # Simulate completion
        completed_dir = full_project / "autotune" / "orchestrator" / "completed"
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
        (completed_dir / "node_0002.json").write_text(json.dumps(completion))

        result = runner.invoke(main, ["reconcile"])
        assert result.exit_code == 0
        assert "0.85" in result.output

        dag = ExperimentDAG.load(dag_path)
        node = dag.get_node("node_0002")
        assert node.node_type == "executed"
        assert node.composite == 0.85
