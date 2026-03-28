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
