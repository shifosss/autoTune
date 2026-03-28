import json
import subprocess

import pytest
from autotune.runner import ExperimentRunner


@pytest.fixture
def project_repo(tmp_path):
    """Create a real git repo for testing."""
    repo = tmp_path / "project"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    (repo / "train.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return repo


class TestWorktreeLifecycle:
    def test_create_and_cleanup(self, project_repo):
        runner = ExperimentRunner(project_repo)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        wt = runner.create_worktree(commit, "node_0001")
        assert wt.exists()
        assert (wt / "train.py").exists()
        runner.cleanup_worktree(wt)
        assert not wt.exists()

    def test_apply_overlay(self, project_repo):
        runner = ExperimentRunner(project_repo)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        wt = runner.create_worktree(commit, "node_0001")
        overlay = project_repo / "overlay"
        overlay.mkdir()
        (overlay / "train.py").write_text("print('modified')")
        runner.apply_overlay(wt, overlay, deletions=[])
        assert (wt / "train.py").read_text() == "print('modified')"
        runner.cleanup_worktree(wt)


class TestResultCollection:
    def test_collect_result(self, project_repo):
        runner = ExperimentRunner(project_repo)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        wt = runner.create_worktree(commit, "node_0001")
        result = {"status": "completed", "composite": 0.85}
        (wt / "result.json").write_text(json.dumps(result))
        archive = project_repo / "archive" / "node_0001"
        archive.mkdir(parents=True)
        collected = runner.collect_result(wt, archive)
        assert collected is not None
        assert collected["composite"] == 0.85
        assert (archive / "result.json").exists()
        runner.cleanup_worktree(wt)

    def test_collect_result_missing(self, project_repo):
        runner = ExperimentRunner(project_repo)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        wt = runner.create_worktree(commit, "node_0001")
        archive = project_repo / "archive" / "node_0001"
        archive.mkdir(parents=True)
        assert runner.collect_result(wt, archive) is None
        runner.cleanup_worktree(wt)


class TestCheckpointCollection:
    def test_collect_checkpoint(self, project_repo):
        runner = ExperimentRunner(project_repo)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        wt = runner.create_worktree(commit, "node_0001")
        ckpt_dir = wt / "checkpoint_output"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.safetensors").write_text("fake weights")
        store = project_repo / "checkpoints"
        store.mkdir()
        ckpt_id = runner.collect_checkpoint(ckpt_dir, store, "node_0001")
        assert ckpt_id == "node_0001"
        assert (store / "node_0001" / "model.safetensors").exists()
        runner.cleanup_worktree(wt)
