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
