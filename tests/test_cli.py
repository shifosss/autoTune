"""Tests for autoTune CLI commands."""

import subprocess

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
        import json

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
