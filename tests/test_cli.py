"""Tests for autoTune CLI commands."""

import json
import subprocess
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from autotune.cli import main


@pytest.fixture
def project_dir(tmp_path):
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


class TestInit:
    def test_creates_overlay(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output
        assert (project_dir / "autotune" / "config.yaml").exists()
        assert (project_dir / "autotune" / "program.md").exists()
        assert (project_dir / "autotune" / "learnings.md").exists()
        assert (project_dir / "autotune" / ".gitignore").exists()
        assert (project_dir / "autotune" / "orchestrator" / "queue").is_dir()
        assert (project_dir / "autotune" / "orchestrator" / "checkpoints").is_dir()

    def test_requires_git(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code != 0

    def test_no_double_init(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["init"])
        assert result.exit_code != 0


class TestStatus:
    def test_status_after_init(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "queue" in result.output.lower() or "no experiments" in result.output.lower()


class TestSuggest:
    def test_suggest_after_seed(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])

        from autotune.dag import ExperimentDAG
        from autotune.types import ScoringConfig

        dag_path = project_dir / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=3)
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
        dag.save()

        result = runner.invoke(main, ["suggest", "--n", "3"])
        assert result.exit_code == 0, result.output
        assert "BRANCH" in result.output or "branch" in result.output.lower()


class TestExplain:
    def test_explain_no_manifold(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["explain"])
        assert result.exit_code == 0
        assert (
            "not yet learned" in result.output.lower() or "no experiments" in result.output.lower()
        )


class TestCheck:
    def test_check_passes(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["check"])
        assert result.exit_code == 0
        assert "all checks passed" in result.output.lower()

    def test_check_not_in_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["check"])
        assert result.exit_code != 0

    def test_check_valid_dag(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        from autotune.dag import ExperimentDAG
        from autotune.types import ScoringConfig

        dag_path = project_dir / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=3)
        dag.add_seed(
            theta={"lr": 2e-4, "rank": 16},
            description="base",
            techniques=(),
            composite=0.72,
            task_score=0.72,
            retention_score=0.95,
            cost=1.0,
        )
        dag.save()
        result = runner.invoke(main, ["check"])
        assert result.exit_code == 0
        assert "all checks passed" in result.output.lower()

    def test_check_invalid_dag_json(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        (project_dir / "autotune" / "dag.json").write_text("{bad json")
        result = runner.invoke(main, ["check"])
        assert result.exit_code == 0  # reports issues, doesn't crash
        assert "dag.json" in result.output.lower()

    def test_check_orphan_parent(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        dag_data = {
            "schema_version": 1,
            "meta": {
                "next_id": 2,
                "total_executed": 1,
                "total_proposed": 0,
                "best_composite": 0.72,
                "best_node_id": "node_0001",
                "objective_count": 3,
                "scoring": {
                    "exploration_weight": 0.005,
                    "diversity_weight": 0.01,
                    "retention_cliff_threshold": -0.10,
                },
            },
            "nodes": {
                "node_0001": {
                    "id": "node_0001",
                    "parent_ids": ["node_9999"],
                    "move_type": "branch",
                    "node_type": "executed",
                    "status": "keep",
                    "theta": {"lr": 2e-4},
                    "task_score": 0.72,
                    "retention_score": 0.95,
                    "cost": 1.0,
                    "composite": 0.72,
                    "depth": 0,
                    "potential": 0.0,
                    "pareto_rank": 0,
                    "created_at": "2026-03-28T00:00:00",
                    "merge_strategy": None,
                    "merge_weights": None,
                    "manifold_coords": None,
                    "strategy_desc": "test",
                    "technique_tags": [],
                    "checkpoint_id": None,
                },
            },
            "technique_stats": {},
        }
        (project_dir / "autotune" / "dag.json").write_text(json.dumps(dag_data))
        result = runner.invoke(main, ["check"])
        assert result.exit_code == 0
        assert "orphan" in result.output.lower()


class TestSubmit:
    def test_submit_seed_creates_dag(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "baseline LoRA",
                "--theta",
                '{"lr": 2e-4, "rank": 16}',
                "--task-score",
                "0.72",
                "--cost",
                "1.5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "node_0001" in result.output
        assert (project_dir / "autotune" / "dag.json").exists()

    def test_submit_seed_with_retention(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "baseline",
                "--theta",
                '{"lr": 2e-4, "rank": 16}',
                "--task-score",
                "0.72",
                "--cost",
                "1.5",
                "--retention-score",
                "0.95",
            ],
        )
        assert result.exit_code == 0, result.output
        from autotune.dag import ExperimentDAG

        dag = ExperimentDAG.load(project_dir / "autotune" / "dag.json")
        node = dag.get_node("node_0001")
        assert node.retention_score == 0.95

    def test_submit_seed_appends(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "first",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        result = runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "second",
                "--theta",
                '{"lr": 5e-4}',
                "--task-score",
                "0.80",
                "--cost",
                "1.2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "node_0002" in result.output

    def test_submit_seed_composite_defaults(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "test",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        from autotune.dag import ExperimentDAG

        dag = ExperimentDAG.load(project_dir / "autotune" / "dag.json")
        node = dag.get_node("node_0001")
        assert node.composite == 0.72  # defaults to task_score

    def test_submit_requires_theta(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(
            main,
            ["submit", "--desc", "test", "--task-score", "0.72", "--cost", "1.0"],
        )
        assert result.exit_code != 0

    def test_submit_queue_writes_spec(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        # Create seed first
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "base",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        # Propose a branch
        runner.invoke(
            main,
            [
                "propose",
                "--parents",
                "node_0001",
                "--move",
                "branch",
                "--desc",
                "increase lr",
                "--theta",
                '{"lr": 5e-4}',
            ],
        )
        # Submit the proposed node to queue
        result = runner.invoke(main, ["submit", "--node", "node_0002"])
        assert result.exit_code == 0, result.output
        queue_file = project_dir / "autotune" / "orchestrator" / "queue" / "node_0002.json"
        assert queue_file.exists()
        spec = json.loads(queue_file.read_text())
        assert spec["node_id"] == "node_0002"

    def test_submit_queue_rejects_nonexistent_node(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "base",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        result = runner.invoke(main, ["submit", "--node", "node_9999"])
        assert result.exit_code != 0


class TestPropose:
    def test_propose_branch(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "base",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        result = runner.invoke(
            main,
            [
                "propose",
                "--parents",
                "node_0001",
                "--move",
                "branch",
                "--desc",
                "increase lr",
                "--theta",
                '{"lr": 5e-4}',
            ],
        )
        assert result.exit_code == 0, result.output
        assert "node_0002" in result.output

    def test_propose_merge(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "A",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "B",
                "--theta",
                '{"lr": 5e-4}',
                "--task-score",
                "0.80",
                "--cost",
                "1.5",
            ],
        )
        result = runner.invoke(
            main,
            [
                "propose",
                "--parents",
                "node_0001",
                "--parents",
                "node_0002",
                "--move",
                "merge",
                "--desc",
                "merge A+B",
                "--theta",
                '{"lr": 3.5e-4}',
                "--merge-strategy",
                "weighted_avg",
                "--merge-weights",
                "0.6,0.4",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "node_0003" in result.output

    def test_propose_invalid_move(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "base",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        result = runner.invoke(
            main,
            [
                "propose",
                "--parents",
                "node_0001",
                "--move",
                "invalid",
                "--desc",
                "bad",
                "--theta",
                '{"lr": 5e-4}',
            ],
        )
        assert result.exit_code != 0

    def test_propose_no_dag(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(
            main,
            [
                "propose",
                "--parents",
                "node_0001",
                "--move",
                "branch",
                "--desc",
                "test",
                "--theta",
                '{"lr": 5e-4}',
            ],
        )
        assert result.exit_code != 0

    def test_propose_nonexistent_parent(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "base",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        result = runner.invoke(
            main,
            [
                "propose",
                "--parents",
                "node_9999",
                "--move",
                "branch",
                "--desc",
                "test",
                "--theta",
                '{"lr": 5e-4}',
            ],
        )
        assert result.exit_code != 0


class TestRank:
    def test_rank_shows_proposals(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "base",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        runner.invoke(
            main,
            [
                "propose",
                "--parents",
                "node_0001",
                "--move",
                "branch",
                "--desc",
                "A",
                "--theta",
                '{"lr": 3e-4}',
            ],
        )
        runner.invoke(
            main,
            [
                "propose",
                "--parents",
                "node_0001",
                "--move",
                "branch",
                "--desc",
                "B",
                "--theta",
                '{"lr": 5e-4}',
            ],
        )
        result = runner.invoke(main, ["rank"])
        assert result.exit_code == 0, result.output
        assert "node_0002" in result.output
        assert "node_0003" in result.output

    def test_rank_no_dag(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["rank"])
        assert result.exit_code != 0

    def test_rank_n_limit(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        runner.invoke(
            main,
            [
                "submit",
                "--desc",
                "base",
                "--theta",
                '{"lr": 2e-4}',
                "--task-score",
                "0.72",
                "--cost",
                "1.0",
            ],
        )
        for i in range(3):
            runner.invoke(
                main,
                [
                    "propose",
                    "--parents",
                    "node_0001",
                    "--move",
                    "branch",
                    "--desc",
                    f"trial {i}",
                    "--theta",
                    f'{{"lr": {(i + 3) * 1e-4}}}',
                ],
            )
        result = runner.invoke(main, ["rank", "--n", "1"])
        assert result.exit_code == 0, result.output
        # Output should have exactly 1 ranked entry (line with "1.")
        ranked_lines = [
            line for line in result.output.strip().split("\n") if line.strip().startswith("1.")
        ]
        assert len(ranked_lines) == 1


class TestOrchestratorCLI:
    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_orchestrator_status(self, mock_gpus, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["orchestrator", "status"])
        assert result.exit_code == 0
        assert "not running" in result.output.lower()


class TestCheckpoints:
    def test_checkpoints_empty(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["checkpoints"])
        assert result.exit_code == 0
        assert "no checkpoints" in result.output.lower() or result.output.strip() != ""

    def test_checkpoints_prune(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        # Create a checkpoint dir and a DAG with a rank-2 node
        ckpt_dir = project_dir / "autotune" / "orchestrator" / "checkpoints" / "node_0001"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model.bin").write_text("weights")

        from autotune.dag import ExperimentDAG
        from autotune.types import ScoringConfig
        dag_path = project_dir / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=2)
        dag.add_seed(theta={"lr": 1e-4}, description="a", techniques=(),
                     composite=0.9, task_score=0.9, cost=1.0)
        dag._data["nodes"]["node_0001"]["pareto_rank"] = 3
        dag._data["nodes"]["node_0001"]["status"] = "discard"
        dag.save()

        result = runner.invoke(main, ["checkpoints", "--prune"])
        assert result.exit_code == 0
        assert not ckpt_dir.exists()
