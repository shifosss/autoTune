"""Experiment orchestrator: GPU scheduling, process management, checkpoint lifecycle."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

# ── defaults (overridable via config.yaml) ──────────────────────────────────

POLL_INTERVAL_SEC = 30
SAFETY_MARGIN_GB = 2.0
DEFAULT_TIMEOUT_HOURS = 8
MAX_CONCURRENT_PER_GPU = 2
DEFAULT_VRAM_ESTIMATE_GB = 0.5


@dataclass
class GPUInfo:
    index: int
    total_mb: int
    free_mb: int
    utilization: int

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024


@dataclass
class RunningExperiment:
    id: str
    spec: dict
    gpu: int
    process: subprocess.Popen
    log_file: IO
    log_path: Path
    started_at: float
    timeout_at: float
    estimated_vram_gb: float


def query_gpus() -> list[GPUInfo]:
    """Query available GPUs via nvidia-smi.

    Returns an empty list if nvidia-smi is not available or fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            gpus.append(GPUInfo(
                index=int(parts[0]),
                total_mb=int(parts[1]),
                free_mb=int(parts[2]),
                utilization=int(parts[3]),
            ))
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.warning("nvidia-smi failed: %s", e)
        return []


class ExperimentOrchestrator:
    """GPU-scheduling daemon that executes queued experiments."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.autotune_dir = project_root / "autotune"
        self.orch_dir = self.autotune_dir / "orchestrator"
        self.queue_dir = self.orch_dir / "queue"
        self.running_dir = self.orch_dir / "running"
        self.completed_dir = self.orch_dir / "completed"
        self.archive_dir = self.orch_dir / "archive"
        self.checkpoint_store = self.orch_dir / "checkpoints"
        self.pid_file = self.orch_dir / "orchestrator.pid"
        self.log_path = self.orch_dir / "orchestrator.log"
        self.gpu_state_file = self.orch_dir / "gpu_state.json"
        self.results_tsv = self.autotune_dir / "results.tsv"

        from autotune.runner import ExperimentRunner

        self.runner = ExperimentRunner(project_root)

        # Load config
        import yaml

        config_path = self.autotune_dir / "config.yaml"
        config: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        run_cfg = config.get("run", {})
        self.run_script = run_cfg.get("script", "train.py")
        self.run_command = run_cfg.get("command")
        self.checkpoint_format = run_cfg.get("checkpoint_format", "safetensors")

        ret_cfg = config.get("retention", {})
        self.retention_enabled = ret_cfg.get("enabled", False)
        self.retention_script = ret_cfg.get("script")
        self.retention_probe_set = ret_cfg.get("probe_set")

        orch_cfg = config.get("orchestrator", {})
        self.poll_interval = orch_cfg.get("poll_interval", POLL_INTERVAL_SEC)
        self.max_per_gpu = orch_cfg.get("max_per_gpu", MAX_CONCURRENT_PER_GPU)
        self.safety_margin_gb = orch_cfg.get("safety_margin_gb", SAFETY_MARGIN_GB)
        self.default_timeout_hours = orch_cfg.get("default_timeout_hours", DEFAULT_TIMEOUT_HOURS)
        self.retention_layers = orch_cfg.get("checkpoint_retention_layers", 2)

        merge_cfg = config.get("merge", {})
        self.default_merge_strategy = merge_cfg.get("default_strategy", "weighted_avg")

        # Runtime state
        self.running: dict[str, RunningExperiment] = {}
        self.gpu_allocations: dict[int, list[str]] = {}
        self._shutdown = False
        self.draining = False
        self._timed_out: dict[str, bool] = {}

        # GPU detection
        gpus = query_gpus()
        if gpus:
            for gpu in gpus:
                self.gpu_allocations[gpu.index] = []
        else:
            logger.warning("No GPUs detected, using virtual GPU 0")
            self.gpu_allocations[0] = []

        # Ensure directories exist
        for d in (self.queue_dir, self.running_dir, self.completed_dir,
                  self.archive_dir, self.checkpoint_store):
            d.mkdir(parents=True, exist_ok=True)

        self._load_state(recover=False)

    def _load_state(self, recover: bool = False) -> None:
        """Load persisted state. Stub — extended in Task 6 with orphan recovery."""
        if self.gpu_state_file.exists():
            try:
                json.loads(self.gpu_state_file.read_text())
            except (json.JSONDecodeError, KeyError):
                pass
