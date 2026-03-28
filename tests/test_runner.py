import json
import subprocess
from pathlib import Path

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


class TestCheckpointIO:
    """Tests for _load_state_dict and _save_state_dict helpers."""

    def test_load_state_dict_torch_file(self, project_repo, monkeypatch):
        """Load a .pt file via torch.load."""
        import types

        fake_state = {"layer.weight": [1.0, 2.0], "layer.bias": [0.1]}
        mock_torch = types.ModuleType("torch")
        mock_torch.load = lambda path, **kw: fake_state
        monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)

        runner = ExperimentRunner(project_repo)
        pt_file = project_repo / "model.pt"
        pt_file.write_text("fake")
        result = runner._load_state_dict(pt_file)
        assert result == fake_state

    def test_load_state_dict_directory_finds_bin(self, project_repo, monkeypatch):
        """Load from a HuggingFace-style directory."""
        import types

        fake_state = {"w": [1.0]}
        mock_torch = types.ModuleType("torch")
        mock_torch.load = lambda path, **kw: fake_state
        monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)
        # Remove safetensors if present
        monkeypatch.setitem(__import__("sys").modules, "safetensors", None)

        runner = ExperimentRunner(project_repo)
        ckpt_dir = project_repo / "checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "pytorch_model.bin").write_text("fake")
        result = runner._load_state_dict(ckpt_dir)
        assert result == fake_state

    def test_load_state_dict_no_torch_raises(self, project_repo, monkeypatch):
        """ImportError if torch not installed."""
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        monkeypatch.setitem(__import__("sys").modules, "safetensors", None)

        runner = ExperimentRunner(project_repo)
        with pytest.raises(ImportError, match="torch"):
            runner._load_state_dict(project_repo / "model.pt")

    def test_save_state_dict_pytorch(self, project_repo, monkeypatch):
        """Save via torch.save."""
        import types

        saved = {}
        mock_torch = types.ModuleType("torch")
        mock_torch.save = lambda obj, path: saved.update({"obj": obj, "path": path})
        monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)

        runner = ExperimentRunner(project_repo)
        out = project_repo / "out.pt"
        runner._save_state_dict({"w": [1.0]}, out, "pytorch")
        assert saved["obj"] == {"w": [1.0]}


class TestMergeCheckpoints:
    """Tests for merge_checkpoints with various strategies."""

    def _setup_parents(self, project_repo, monkeypatch):
        """Create fake checkpoints and mock torch."""
        import types

        store = project_repo / "checkpoints"
        store.mkdir()
        # Parent A: {"w": [1.0, 2.0]}
        p_a = store / "node_0001"
        p_a.mkdir()
        (p_a / "pytorch_model.bin").write_text("fake_a")
        # Parent B: {"w": [3.0, 4.0]}
        p_b = store / "node_0002"
        p_b.mkdir()
        (p_b / "pytorch_model.bin").write_text("fake_b")

        import numpy as np

        state_a = {"w": np.array([1.0, 2.0])}
        state_b = {"w": np.array([3.0, 4.0])}

        mock_torch = types.ModuleType("torch")
        load_map = {str(p_a / "pytorch_model.bin"): state_a, str(p_b / "pytorch_model.bin"): state_b}
        mock_torch.load = lambda path, **kw: load_map[str(path)]
        saved_state = {}
        mock_torch.save = lambda obj, path: saved_state.update({"obj": obj, "path": str(path)})
        monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)
        monkeypatch.setitem(__import__("sys").modules, "safetensors", None)
        return store, saved_state

    def test_weighted_avg(self, project_repo, monkeypatch):
        import numpy as np

        store, saved_state = self._setup_parents(project_repo, monkeypatch)
        dest = project_repo / "merged"
        runner = ExperimentRunner(project_repo)
        runner.merge_checkpoints(
            parent_ids=("node_0001", "node_0002"),
            weights=(0.5, 0.5),
            strategy="weighted_avg",
            checkpoint_store=store,
            dest_path=dest,
        )
        np.testing.assert_array_almost_equal(saved_state["obj"]["w"], [2.0, 3.0])

    def test_compose(self, project_repo, monkeypatch):
        import numpy as np

        store, saved_state = self._setup_parents(project_repo, monkeypatch)
        dest = project_repo / "merged"
        runner = ExperimentRunner(project_repo)
        runner.merge_checkpoints(
            parent_ids=("node_0001", "node_0002"),
            weights=(1.0, 1.0),
            strategy="compose",
            checkpoint_store=store,
            dest_path=dest,
        )
        # compose: base + sum(deltas from base)
        # base = A = [1,2], delta_B = B - A = [2,2], result = A + [2,2] = [3,4]
        np.testing.assert_array_almost_equal(saved_state["obj"]["w"], [3.0, 4.0])

    def test_selective_raises(self, project_repo, monkeypatch):
        store, _ = self._setup_parents(project_repo, monkeypatch)
        runner = ExperimentRunner(project_repo)
        with pytest.raises(NotImplementedError, match="selective"):
            runner.merge_checkpoints(
                parent_ids=("node_0001", "node_0002"),
                weights=(0.5, 0.5),
                strategy="selective",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )

    def test_unknown_strategy_raises(self, project_repo, monkeypatch):
        store, _ = self._setup_parents(project_repo, monkeypatch)
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="Unknown"):
            runner.merge_checkpoints(
                parent_ids=("node_0001",),
                weights=(1.0,),
                strategy="magic",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )

    def test_no_torch_raises(self, project_repo, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        monkeypatch.setitem(__import__("sys").modules, "safetensors", None)
        store = project_repo / "checkpoints"
        store.mkdir()
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ImportError, match="torch"):
            runner.merge_checkpoints(
                parent_ids=("node_0001",),
                weights=(1.0,),
                strategy="weighted_avg",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )


class TestGraftCheckpoint:
    """Tests for graft_checkpoint delta-transfer operation."""

    def test_graft_applies_delta(self, project_repo, monkeypatch):
        import types

        import numpy as np

        store = project_repo / "checkpoints"
        store.mkdir()
        for name in ("source", "source_parent", "target_parent"):
            d = store / name
            d.mkdir()
            (d / "pytorch_model.bin").write_text("fake")

        states = {
            "source": {"w": np.array([5.0, 6.0])},
            "source_parent": {"w": np.array([1.0, 2.0])},
            "target_parent": {"w": np.array([10.0, 20.0])},
        }
        mock_torch = types.ModuleType("torch")
        mock_torch.load = lambda path, **kw: states[Path(path).parent.name]
        saved = {}
        mock_torch.save = lambda obj, path: saved.update({"obj": obj})
        monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)
        monkeypatch.setitem(__import__("sys").modules, "safetensors", None)

        runner = ExperimentRunner(project_repo)
        runner.graft_checkpoint(
            source_id="source",
            source_parent_id="source_parent",
            target_parent_id="target_parent",
            checkpoint_store=store,
            dest_path=project_repo / "grafted",
        )
        # delta = source - source_parent = [4, 4]
        # grafted = target_parent + delta = [14, 24]
        np.testing.assert_array_almost_equal(saved["obj"]["w"], [14.0, 24.0])

    def test_graft_no_torch_raises(self, project_repo, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        monkeypatch.setitem(__import__("sys").modules, "safetensors", None)
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ImportError, match="torch"):
            runner.graft_checkpoint(
                source_id="s", source_parent_id="sp", target_parent_id="tp",
                checkpoint_store=project_repo, dest_path=project_repo / "out",
            )


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
