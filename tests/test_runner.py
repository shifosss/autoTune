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


def adapter_config(**overrides):
    config = {
        "base_model_name_or_path": "meta/llama-base",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "v_proj"],
    }
    config.update(overrides)
    return config


def setup_mock_adapter_store(project_repo, monkeypatch, specs):
    import types

    store = project_repo / "checkpoints"
    store.mkdir()
    load_map = {}
    for spec in specs:
        checkpoint_dir = store / spec["name"]
        checkpoint_dir.mkdir()
        (checkpoint_dir / "pytorch_model.bin").write_text("fake")
        if spec.get("write_config", True):
            config = spec.get("config", adapter_config())
            (checkpoint_dir / "adapter_config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True)
            )
        load_map[str(checkpoint_dir / "pytorch_model.bin")] = spec["state"]

    mock_torch = types.ModuleType("torch")
    mock_torch.load = lambda path, **kw: load_map[str(path)]
    saved_state = {}
    mock_torch.save = lambda obj, path: saved_state.update({"obj": obj, "path": str(path)})
    monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)
    monkeypatch.setitem(__import__("sys").modules, "safetensors", None)
    return store, saved_state


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

    def _setup_parents(self, project_repo, monkeypatch, specs=None):
        import numpy as np

        if specs is None:
            specs = [
                {"name": "node_0001", "state": {"w": np.array([1.0, 2.0], dtype=np.float32)}},
                {"name": "node_0002", "state": {"w": np.array([3.0, 4.0], dtype=np.float32)}},
            ]
        return setup_mock_adapter_store(project_repo, monkeypatch, specs)

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
        assert json.loads((dest / "adapter_config.json").read_text()) == adapter_config()

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
        assert json.loads((dest / "adapter_config.json").read_text()) == adapter_config()

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

    def test_missing_adapter_config_raises(self, project_repo, monkeypatch):
        store, _ = self._setup_parents(
            project_repo,
            monkeypatch,
            specs=[
                {"name": "node_0001", "state": {"w": [1.0]}, "write_config": False},
                {"name": "node_0002", "state": {"w": [2.0]}},
            ],
        )
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="missing adapter_config.json"):
            runner.merge_checkpoints(
                parent_ids=("node_0001", "node_0002"),
                weights=(0.5, 0.5),
                strategy="weighted_avg",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )

    def test_mismatched_base_model_raises(self, project_repo, monkeypatch):
        store, _ = self._setup_parents(
            project_repo,
            monkeypatch,
            specs=[
                {"name": "node_0001", "state": {"w": [1.0]}},
                {
                    "name": "node_0002",
                    "state": {"w": [2.0]},
                    "config": adapter_config(base_model_name_or_path="meta/other-base"),
                },
            ],
        )
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="base_model_name_or_path"):
            runner.merge_checkpoints(
                parent_ids=("node_0001", "node_0002"),
                weights=(0.5, 0.5),
                strategy="weighted_avg",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )

    def test_mismatched_peft_type_raises(self, project_repo, monkeypatch):
        store, _ = self._setup_parents(
            project_repo,
            monkeypatch,
            specs=[
                {"name": "node_0001", "state": {"w": [1.0]}},
                {
                    "name": "node_0002",
                    "state": {"w": [2.0]},
                    "config": adapter_config(peft_type="IA3"),
                },
            ],
        )
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="peft_type"):
            runner.merge_checkpoints(
                parent_ids=("node_0001", "node_0002"),
                weights=(0.5, 0.5),
                strategy="weighted_avg",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )

    def test_weighted_avg_key_mismatch_raises(self, project_repo, monkeypatch):
        store, _ = self._setup_parents(
            project_repo,
            monkeypatch,
            specs=[
                {"name": "node_0001", "state": {"w": [1.0], "b": [0.1]}},
                {"name": "node_0002", "state": {"w": [2.0]}},
            ],
        )
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="tensor keys"):
            runner.merge_checkpoints(
                parent_ids=("node_0001", "node_0002"),
                weights=(0.5, 0.5),
                strategy="weighted_avg",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )

    def test_compose_key_mismatch_raises(self, project_repo, monkeypatch):
        store, _ = self._setup_parents(
            project_repo,
            monkeypatch,
            specs=[
                {"name": "node_0001", "state": {"w": [1.0], "b": [0.1]}},
                {"name": "node_0002", "state": {"w": [2.0]}},
            ],
        )
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="tensor keys"):
            runner.merge_checkpoints(
                parent_ids=("node_0001", "node_0002"),
                weights=(1.0, 1.0),
                strategy="compose",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )

    def test_shape_mismatch_raises(self, project_repo, monkeypatch):
        import numpy as np

        store, _ = self._setup_parents(
            project_repo,
            monkeypatch,
            specs=[
                {"name": "node_0001", "state": {"w": np.array([1.0, 2.0], dtype=np.float32)}},
                {"name": "node_0002", "state": {"w": np.array([[3.0, 4.0]], dtype=np.float32)}},
            ],
        )
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="tensor shape"):
            runner.merge_checkpoints(
                parent_ids=("node_0001", "node_0002"),
                weights=(0.5, 0.5),
                strategy="weighted_avg",
                checkpoint_store=store,
                dest_path=project_repo / "merged",
            )

    def test_dtype_mismatch_raises(self, project_repo, monkeypatch):
        import numpy as np

        store, _ = self._setup_parents(
            project_repo,
            monkeypatch,
            specs=[
                {"name": "node_0001", "state": {"w": np.array([1.0, 2.0], dtype=np.float32)}},
                {"name": "node_0002", "state": {"w": np.array([3.0, 4.0], dtype=np.float64)}},
            ],
        )
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="tensor dtype"):
            runner.merge_checkpoints(
                parent_ids=("node_0001", "node_0002"),
                weights=(0.5, 0.5),
                strategy="weighted_avg",
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
        import numpy as np

        store, saved = setup_mock_adapter_store(
            project_repo,
            monkeypatch,
            [
                {"name": "source", "state": {"w": np.array([5.0, 6.0], dtype=np.float32)}},
                {
                    "name": "source_parent",
                    "state": {"w": np.array([1.0, 2.0], dtype=np.float32)},
                },
                {
                    "name": "target_parent",
                    "state": {"w": np.array([10.0, 20.0], dtype=np.float32)},
                },
            ],
        )

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
        assert json.loads((project_repo / "grafted" / "adapter_config.json").read_text()) == (
            adapter_config()
        )

    def test_graft_target_missing_delta_key_raises(self, project_repo, monkeypatch):
        import numpy as np

        store, _ = setup_mock_adapter_store(
            project_repo,
            monkeypatch,
            [
                {"name": "source", "state": {"w": np.array([5.0], dtype=np.float32)}},
                {"name": "source_parent", "state": {"w": np.array([1.0], dtype=np.float32)}},
                {"name": "target_parent", "state": {"other": np.array([10.0], dtype=np.float32)}},
            ],
        )
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ValueError, match="tensor keys"):
            runner.graft_checkpoint(
                source_id="source",
                source_parent_id="source_parent",
                target_parent_id="target_parent",
                checkpoint_store=store,
                dest_path=project_repo / "grafted",
            )

    def test_graft_no_torch_raises(self, project_repo, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        monkeypatch.setitem(__import__("sys").modules, "safetensors", None)
        runner = ExperimentRunner(project_repo)
        with pytest.raises(ImportError, match="torch"):
            runner.graft_checkpoint(
                source_id="s",
                source_parent_id="sp",
                target_parent_id="tp",
                checkpoint_store=project_repo,
                dest_path=project_repo / "out",
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
