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
