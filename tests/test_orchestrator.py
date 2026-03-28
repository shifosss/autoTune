"""Tests for ExperimentOrchestrator."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autotune.orchestrator import (
    DEFAULT_TIMEOUT_HOURS,
    GPUInfo,
    RunningExperiment,
    query_gpus,
)


class TestGPUInfo:
    def test_free_gb(self):
        gpu = GPUInfo(index=0, total_mb=24576, free_mb=12288, utilization=50)
        assert gpu.free_gb == 12.0


class TestQueryGPUs:
    def test_parses_nvidia_smi(self, monkeypatch):
        output = "0, 24576, 20000, 30\n1, 16384, 10000, 60\n"
        mock_run = MagicMock(return_value=MagicMock(stdout=output, returncode=0))
        monkeypatch.setattr(subprocess, "run", mock_run)
        gpus = query_gpus()
        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[0].total_mb == 24576
        assert gpus[0].free_mb == 20000
        assert gpus[1].index == 1

    def test_returns_empty_on_failure(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", MagicMock(side_effect=FileNotFoundError)
        )
        assert query_gpus() == []


from autotune.orchestrator import ExperimentOrchestrator


@pytest.fixture
def orch_project(tmp_path):
    """Create a minimal autoTune project for orchestrator tests."""
    repo = tmp_path / "project"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True, check=True)
    (repo / "train.py").write_text("print('train')")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True)

    # Create autotune overlay
    at = repo / "autotune"
    at.mkdir()
    config = {
        "run": {"script": "train.py", "checkpoint_format": "safetensors"},
        "retention": {"enabled": False},
        "orchestrator": {"poll_interval": 10, "max_per_gpu": 1, "safety_margin_gb": 1.0,
                         "default_timeout_hours": 4, "checkpoint_retention_layers": 2},
        "merge": {"default_strategy": "weighted_avg"},
        "scoring": {"exploration_weight": 0.005, "diversity_weight": 0.01,
                     "retention_cliff_threshold": -0.10},
    }
    import yaml
    (at / "config.yaml").write_text(yaml.dump(config))
    for d in ("orchestrator/queue", "orchestrator/running", "orchestrator/completed",
              "orchestrator/archive", "orchestrator/checkpoints"):
        (at / d).mkdir(parents=True)
    return repo


class TestOrchestratorInit:
    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_loads_config(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        assert orch.poll_interval == 10
        assert orch.max_per_gpu == 1
        assert orch.safety_margin_gb == 1.0
        assert orch.run_script == "train.py"
        assert orch.checkpoint_format == "safetensors"

    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_creates_directories(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        assert orch.queue_dir.is_dir()
        assert orch.running_dir.is_dir()
        assert orch.completed_dir.is_dir()
        assert orch.archive_dir.is_dir()
        assert orch.checkpoint_store.is_dir()

    @patch("autotune.orchestrator.query_gpus", return_value=[
        GPUInfo(0, 24576, 20000, 30),
    ])
    def test_detects_gpus(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        assert 0 in orch.gpu_allocations

    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_fallback_virtual_gpu(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        assert 0 in orch.gpu_allocations  # virtual GPU fallback


class TestStatePersistence:
    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_save_and_load_state(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        orch._save_state()
        assert orch.gpu_state_file.exists()
        data = json.loads(orch.gpu_state_file.read_text())
        assert "last_updated" in data
        assert data["total_running"] == 0


class TestOrphanRecovery:
    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_recovers_stale_running(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        # Simulate a stale running spec left from a crash
        stale = {"node_id": "node_0099", "move_type": "branch"}
        (orch.running_dir / "node_0099.json").write_text(json.dumps(stale))
        orch._recover_orphans()
        # Stale spec should be removed from running
        assert not (orch.running_dir / "node_0099.json").exists()
        # Completion should be written
        assert (orch.completed_dir / "node_0099.json").exists()
        comp = json.loads((orch.completed_dir / "node_0099.json").read_text())
        assert comp["status"] == "crash"


class TestScheduling:
    @patch("autotune.orchestrator.query_gpus", return_value=[
        GPUInfo(0, 24576, 20000, 30),
        GPUInfo(1, 16384, 10000, 60),
    ])
    def test_get_pending_sorted(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        spec_a = {"node_id": "a", "priority": 2, "submitted_at": "2026-03-28T01:00:00"}
        spec_b = {"node_id": "b", "priority": 1, "submitted_at": "2026-03-28T00:00:00"}
        (orch.queue_dir / "a.json").write_text(json.dumps(spec_a))
        (orch.queue_dir / "b.json").write_text(json.dumps(spec_b))
        pending = orch._get_pending()
        assert pending[0]["node_id"] == "b"  # lower priority number = higher priority

    @patch("autotune.orchestrator.query_gpus", return_value=[
        GPUInfo(0, 24576, 20000, 30),
        GPUInfo(1, 16384, 14000, 10),
    ])
    def test_find_best_gpu_tightest_fit(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        # GPU 0: 20000MB free = ~19.5GB, GPU 1: 14000MB = ~13.7GB
        # Need 10GB. Both fit. GPU 1 is tighter fit.
        gpu = orch._find_best_gpu(10.0)
        assert gpu == 1

    @patch("autotune.orchestrator.query_gpus", return_value=[
        GPUInfo(0, 24576, 2048, 90),
    ])
    def test_find_best_gpu_none_when_full(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        # GPU 0: 2GB free, safety=1.0, schedulable=1.0. Need 5GB — won't fit.
        gpu = orch._find_best_gpu(5.0)
        assert gpu is None


class TestLifecycle:
    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_mark_crashed_writes_completion(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        spec = {"node_id": "node_0001", "move_type": "branch"}
        orch._mark_crashed("node_0001", spec, "test error")
        comp_file = orch.completed_dir / "node_0001.json"
        assert comp_file.exists()
        data = json.loads(comp_file.read_text())
        assert data["status"] == "crash"
        assert "test error" in data["error"]


class TestCheckpointPruning:
    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_prunes_rank_2_checkpoints(self, mock_gpus, orch_project):
        orch = ExperimentOrchestrator(orch_project)
        # Create a fake checkpoint dir
        (orch.checkpoint_store / "node_0005").mkdir()
        (orch.checkpoint_store / "node_0005" / "model.bin").write_text("weights")

        # Create a DAG with that node at rank 2
        from autotune.dag import ExperimentDAG
        from autotune.types import ScoringConfig
        dag_path = orch_project / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=2)
        # Add 3 seeds so one gets rank 2
        dag.add_seed(theta={"lr": 1e-4}, description="a", techniques=(),
                     composite=0.9, task_score=0.9, cost=1.0)
        dag.add_seed(theta={"lr": 2e-4}, description="b", techniques=(),
                     composite=0.8, task_score=0.8, cost=0.5)
        dag.add_seed(theta={"lr": 3e-4}, description="c", techniques=(),
                     composite=0.5, task_score=0.5, cost=2.0)
        # Manually set node_0003 to rank 2 for deterministic test
        dag._data["nodes"]["node_0003"]["pareto_rank"] = 2
        dag._data["nodes"]["node_0003"]["status"] = "discard"
        # Also set up node_0005 which has the checkpoint
        dag._data["nodes"]["node_0005"] = dict(dag._data["nodes"]["node_0003"])
        dag._data["nodes"]["node_0005"]["id"] = "node_0005"
        dag.save()

        orch._prune_checkpoints(dag)
        assert not (orch.checkpoint_store / "node_0005").exists()


class TestCmdMethods:
    @patch("autotune.orchestrator.query_gpus", return_value=[])
    def test_cmd_status_no_pid(self, mock_gpus, orch_project, capsys):
        orch = ExperimentOrchestrator(orch_project)
        orch.cmd_status()
        captured = capsys.readouterr()
        assert "not running" in captured.out.lower() or "no pid" in captured.out.lower()
