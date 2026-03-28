# Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `ExperimentOrchestrator` that executes queued experiments on GPUs, manages checkpoints, and feeds results back into the manifold + forgetting systems.

**Architecture:** Direct port of autoMIL's orchestrator (`/Users/chenzhang/Alex's Codebases/autoMIL/src/automil/orchestrator.py`) with three extensions: checkpoint preparation by move_type in `_launch()`, post-completion manifold + forgetting updates in `_handle_completion()`, and checkpoint pruning for Pareto rank >= 2. torch is optional — guarded at import time, only required for merge/graft moves.

**Tech Stack:** Python 3.11+, subprocess (process management), nvidia-smi (GPU detection), torch/safetensors (optional, checkpoint I/O), Click (CLI)

**Spec:** `docs/specs/2026-03-28-orchestrator-design.md`
**Reference:** `autoMIL/src/automil/orchestrator.py` (780 lines, proven patterns)

---

## File Structure

```
src/autotune/
├── orchestrator.py          # NEW: ExperimentOrchestrator class (~600 lines)
├── runner.py                # MODIFY: add merge_checkpoints, graft_checkpoint, I/O helpers (~320 lines)
├── cli.py                   # MODIFY: add orchestrator group + checkpoints command (~684 lines)
tests/
├── test_orchestrator.py     # NEW: orchestrator unit tests (~250 lines)
├── test_runner.py           # MODIFY: add merge/graft tests (~200 lines)
├── test_cli.py              # MODIFY: add orchestrator + checkpoints tests (~740 lines)
```

---

## Task 1: Runner Checkpoint I/O Helpers

**Files:**
- Modify: `src/autotune/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for _load_state_dict and _save_state_dict**

Append to `tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_runner.py::TestCheckpointIO -v`
Expected: FAIL (no `_load_state_dict` method)

- [ ] **Step 3: Implement _load_state_dict and _save_state_dict**

Add to `src/autotune/runner.py` at end of `ExperimentRunner` class:

```python
    def _load_state_dict(self, path: Path) -> dict:
        """Load a model state_dict from a file or directory.

        Supports .pt, .bin, .safetensors files and HuggingFace directories.
        Raises ImportError if neither torch nor safetensors is installed.
        """
        if path.is_dir():
            candidates = [
                "adapter_model.safetensors",
                "pytorch_model.bin",
                "model.safetensors",
            ]
            for name in candidates:
                if (path / name).exists():
                    return self._load_state_dict(path / name)
            raise FileNotFoundError(f"No checkpoint file found in {path}")

        # Try safetensors first
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file

                return load_file(str(path))
            except (ImportError, ModuleNotFoundError):
                pass

        # Fall back to torch
        try:
            import torch

            return torch.load(str(path), weights_only=True)
        except (ImportError, ModuleNotFoundError):
            pass

        raise ImportError(
            "torch is required for checkpoint operations. "
            "Install with: uv pip install -e '.[torch]'"
        )

    def _save_state_dict(self, state_dict: dict, path: Path, fmt: str) -> None:
        """Save a state_dict to disk.

        Args:
            state_dict: The state dict to save.
            path: Destination file path.
            fmt: Format — "safetensors" or "pytorch".
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "safetensors":
            try:
                from safetensors.torch import save_file

                save_file(state_dict, str(path))
                return
            except (ImportError, ModuleNotFoundError):
                pass

        import torch

        torch.save(state_dict, str(path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_runner.py::TestCheckpointIO -v`
Expected: PASS

- [ ] **Step 5: Run all runner tests for regression**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_runner.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/autotune/runner.py tests/test_runner.py
git commit -m "feat(runner): add _load_state_dict and _save_state_dict checkpoint I/O helpers"
```

---

## Task 2: Runner merge_checkpoints

**Files:**
- Modify: `src/autotune/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for merge_checkpoints**

Append to `tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_runner.py::TestMergeCheckpoints -v`
Expected: FAIL

- [ ] **Step 3: Implement merge_checkpoints**

Add to `src/autotune/runner.py` in `ExperimentRunner`:

```python
    def merge_checkpoints(
        self,
        parent_ids: tuple[str, ...],
        weights: tuple[float, ...],
        strategy: str,
        checkpoint_store: Path,
        dest_path: Path,
    ) -> None:
        """Merge multiple parent checkpoints into a single checkpoint.

        Args:
            parent_ids: IDs of parent nodes whose checkpoints to merge.
            weights: Per-parent weights for merging.
            strategy: One of "weighted_avg", "compose", "selective".
            checkpoint_store: Root directory containing per-node checkpoint dirs.
            dest_path: Where to write the merged checkpoint.

        Raises:
            ImportError: If torch is not installed.
            NotImplementedError: If strategy is "selective".
            ValueError: If strategy is unknown.
        """
        state_dicts = [
            self._load_state_dict(checkpoint_store / pid)
            for pid in parent_ids
        ]

        if strategy == "weighted_avg":
            merged = {}
            common_keys = set(state_dicts[0].keys())
            for sd in state_dicts[1:]:
                common_keys &= set(sd.keys())
            for key in common_keys:
                merged[key] = sum(
                    w * sd[key] for w, sd in zip(weights, state_dicts)
                )
            self._save_state_dict(merged, dest_path / "pytorch_model.bin", "pytorch")

        elif strategy == "compose":
            base = state_dicts[0]
            accumulated_delta = {k: base[k] * 0.0 for k in base}
            for sd in state_dicts[1:]:
                for k in accumulated_delta:
                    if k in sd:
                        accumulated_delta[k] = accumulated_delta[k] + (sd[k] - base[k])
            composed = {k: base[k] + accumulated_delta[k] for k in base}
            self._save_state_dict(composed, dest_path / "pytorch_model.bin", "pytorch")

        elif strategy == "selective":
            raise NotImplementedError(
                "selective merge requires layer_manifest config (not yet supported)"
            )
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        logger.info("Merged checkpoints %s via %s -> %s", parent_ids, strategy, dest_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_runner.py::TestMergeCheckpoints -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/runner.py tests/test_runner.py
git commit -m "feat(runner): add merge_checkpoints with weighted_avg and compose strategies"
```

---

## Task 3: Runner graft_checkpoint

**Files:**
- Modify: `src/autotune/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_runner.py::TestGraftCheckpoint -v`
Expected: FAIL

- [ ] **Step 3: Implement graft_checkpoint**

Add to `src/autotune/runner.py`:

```python
    def graft_checkpoint(
        self,
        source_id: str,
        source_parent_id: str,
        target_parent_id: str,
        checkpoint_store: Path,
        dest_path: Path,
    ) -> None:
        """Apply a technique delta from one branch to another parent.

        Computes delta = source - source_parent, then grafted = target_parent + delta.

        Raises:
            ImportError: If torch is not installed.
        """
        source = self._load_state_dict(checkpoint_store / source_id)
        source_parent = self._load_state_dict(checkpoint_store / source_parent_id)
        target_parent = self._load_state_dict(checkpoint_store / target_parent_id)

        grafted = {}
        for key in target_parent:
            if key in source and key in source_parent:
                delta = source[key] - source_parent[key]
                grafted[key] = target_parent[key] + delta
            else:
                grafted[key] = target_parent[key]

        self._save_state_dict(grafted, dest_path / "pytorch_model.bin", "pytorch")
        logger.info("Grafted %s (delta from %s) onto %s -> %s",
                     source_id, source_parent_id, target_parent_id, dest_path)
```

- [ ] **Step 4: Run tests and verify pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_runner.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/autotune/runner.py tests/test_runner.py
git commit -m "feat(runner): add graft_checkpoint with delta-transfer operation"
```

---

## Task 4: Orchestrator Module Scaffold

**Files:**
- Create: `src/autotune/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write tests for data structures and query_gpus**

Create `tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py::TestGPUInfo tests/test_orchestrator.py::TestQueryGPUs -v`
Expected: FAIL (no module)

- [ ] **Step 3: Create orchestrator.py with scaffold**

Create `src/autotune/orchestrator.py`. Port from autoMIL's structure (lines 1-180):

```python
"""Experiment orchestrator: GPU scheduling, process management, checkpoint lifecycle."""

from __future__ import annotations

import json
import logging
import os
import signal
import shlex
import shutil
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add module scaffold with GPUInfo, RunningExperiment, query_gpus"
```

---

## Task 5: Orchestrator __init__ and Config Loading

**Files:**
- Modify: `src/autotune/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write test for __init__ config loading**

Append to `tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py::TestOrchestratorInit -v`
Expected: FAIL

- [ ] **Step 3: Implement ExperimentOrchestrator.__init__**

Add to `src/autotune/orchestrator.py`, porting from autoMIL lines 120–179:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/autotune/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add __init__ with config loading and GPU detection"
```

---

## Task 6: State Persistence, Orphan Recovery, Scheduling

**Files:**
- Modify: `src/autotune/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write tests for state persistence, orphan recovery, and scheduling**

Append to `tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py::TestStatePersistence tests/test_orchestrator.py::TestOrphanRecovery tests/test_orchestrator.py::TestScheduling -v`
Expected: FAIL

- [ ] **Step 3: Implement _load_state, _save_state, _recover_orphans, _get_pending, _find_best_gpu, _pre_launch_check**

Add these methods to `ExperimentOrchestrator` in `src/autotune/orchestrator.py`. Port directly from autoMIL lines 206–322, adapting directory names (`automil` → `autotune`, timeout units from minutes to hours):

```python
    # ── state persistence ──
    # (replaces the stub _load_state from Task 5)

    def _load_state(self, recover: bool = False) -> None:
        if self.gpu_state_file.exists():
            try:
                json.loads(self.gpu_state_file.read_text())
            except (json.JSONDecodeError, KeyError):
                pass
        if recover:
            self._recover_orphans()

    def _save_state(self) -> None:
        gpus = query_gpus()
        gpu_details = []
        for gpu in gpus:
            alloc = self.gpu_allocations.get(gpu.index, [])
            alloc_vram = sum(
                self.running[nid].estimated_vram_gb for nid in alloc if nid in self.running
            )
            gpu_details.append({
                "index": gpu.index,
                "total_mb": gpu.total_mb,
                "free_mb": gpu.free_mb,
                "schedulable_free_gb": round(gpu.free_gb - self.safety_margin_gb - alloc_vram, 2),
                "running": alloc,
                "utilization_pct": gpu.utilization,
            })
        state = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_running": len(self.running),
            "queue_depth": len(list(self.queue_dir.glob("*.json"))),
            "gpus": gpu_details,
        }
        self.gpu_state_file.write_text(json.dumps(state, indent=2))

    # ── orphan recovery ──

    def _recover_orphans(self) -> None:
        for spec_file in sorted(self.running_dir.glob("*.json")):
            try:
                spec = json.loads(spec_file.read_text())
            except (json.JSONDecodeError, KeyError):
                spec_file.unlink()
                continue
            node_id = spec.get("node_id", spec_file.stem)
            logger.warning("Recovering orphan: %s", node_id)

            # Write crash completion
            completion = {
                "node_id": node_id,
                "status": "crash",
                "metrics": {},
                "error": "Orchestrator restarted while running",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            archive_node = self.archive_dir / node_id
            archive_node.mkdir(parents=True, exist_ok=True)
            (archive_node / "result.json").write_text(json.dumps(completion, indent=2))
            (self.completed_dir / f"{node_id}.json").write_text(json.dumps(completion, indent=2))
            spec_file.unlink()

            # Clean up worktree if it exists
            wt = self.runner.worktree_path(node_id)
            if wt.exists():
                try:
                    self.runner.cleanup_worktree(wt)
                except Exception as e:
                    logger.warning("Failed to cleanup orphan worktree %s: %s", wt, e)

    # ── scheduling ──

    def _get_pending(self) -> list[dict]:
        pending = []
        for f in self.queue_dir.glob("*.json"):
            try:
                spec = json.loads(f.read_text())
                spec["_file"] = f
                pending.append(spec)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping invalid queue spec: %s", f)
        pending.sort(key=lambda s: (s.get("priority", 1), s.get("submitted_at", "")))
        return pending

    def _find_best_gpu(self, needed_gb: float) -> int | None:
        candidates: list[tuple[int, float]] = []
        for gpu_idx, running_ids in self.gpu_allocations.items():
            if len(running_ids) >= self.max_per_gpu:
                continue
            alloc_vram = sum(
                self.running[nid].estimated_vram_gb for nid in running_ids if nid in self.running
            )
            gpus = query_gpus()
            gpu_info = next((g for g in gpus if g.index == gpu_idx), None)
            free_gb = gpu_info.free_gb if gpu_info else 24.0  # virtual GPU fallback
            schedulable = free_gb - self.safety_margin_gb - alloc_vram
            if schedulable >= needed_gb:
                candidates.append((gpu_idx, schedulable))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[1])  # tightest fit
        return candidates[0][0]

    def _pre_launch_check(self, gpu_id: int, needed_gb: float) -> bool:
        gpus = query_gpus()
        gpu = next((g for g in gpus if g.index == gpu_id), None)
        if gpu is None:
            return True  # virtual GPU, skip check
        return gpu.free_gb >= needed_gb + self.safety_margin_gb
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/autotune/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add state persistence, orphan recovery, GPU scheduling"
```

---

## Task 7: Orchestrator Launch and Completion Handling

**Files:**
- Modify: `src/autotune/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write test for _mark_crashed**

Append to `tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py::TestLifecycle -v`
Expected: FAIL

- [ ] **Step 3: Implement _launch, _check_running, _handle_completion, _handle_timeout, _mark_crashed, _append_results_tsv**

Add to `ExperimentOrchestrator`. This is the core lifecycle — port from autoMIL lines 326–571 with the three autoTune extensions. Key method outlines:

```python
    # ── experiment lifecycle ──

    def _launch(self, spec: dict, gpu_id: int) -> None:
        node_id = spec["node_id"]
        move_type = spec.get("move_type", "branch")
        parent_ids = spec.get("parent_ids", [])

        # Archive setup
        archive_node = self.archive_dir / node_id
        archive_node.mkdir(parents=True, exist_ok=True)
        clean_spec = {k: v for k, v in spec.items() if k != "_file"}
        (archive_node / "spec.json").write_text(json.dumps(clean_spec, indent=2))

        # Remove from queue immediately
        queue_file = spec.get("_file")
        if queue_file and Path(queue_file).exists():
            Path(queue_file).unlink()

        # Create worktree
        base_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.project_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        try:
            wt_path = self.runner.create_worktree(base_commit, node_id)
        except subprocess.CalledProcessError as e:
            self._mark_crashed(node_id, spec, f"Worktree creation failed: {e}")
            return

        # [NEW] Checkpoint preparation by move_type
        parent_ckpt_dir = wt_path / ".parent_checkpoint"
        try:
            if move_type in ("branch", "seed") and parent_ids:
                self.runner.prepare_branch(parent_ids[0], self.checkpoint_store, wt_path)
            elif move_type == "merge":
                parent_ckpt_dir.mkdir(parents=True, exist_ok=True)
                weights = tuple(spec.get("merge_weights") or [1.0 / len(parent_ids)] * len(parent_ids))
                strategy = spec.get("merge_strategy") or self.default_merge_strategy
                self.runner.merge_checkpoints(
                    parent_ids=tuple(parent_ids), weights=weights,
                    strategy=strategy, checkpoint_store=self.checkpoint_store,
                    dest_path=parent_ckpt_dir,
                )
            elif move_type == "graft" and len(parent_ids) >= 2:
                parent_ckpt_dir.mkdir(parents=True, exist_ok=True)
                # graft: parent_ids[0] = technique source, parent_ids[1] = target
                # Need source's parent from DAG
                source_parent = spec.get("graft_source_parent", parent_ids[0])
                self.runner.graft_checkpoint(
                    source_id=parent_ids[0], source_parent_id=source_parent,
                    target_parent_id=parent_ids[1],
                    checkpoint_store=self.checkpoint_store, dest_path=parent_ckpt_dir,
                )
        except ImportError as e:
            self._mark_crashed(node_id, spec, str(e))
            self.runner.cleanup_worktree(wt_path)
            return
        except (FileNotFoundError, OSError) as e:
            self._mark_crashed(node_id, spec, f"Checkpoint preparation failed: {e}")
            self.runner.cleanup_worktree(wt_path)
            return

        # Build environment
        ckpt_output_dir = wt_path / ".checkpoint_output"
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["AUTOTUNE_GPU"] = "0"
        env["AUTOTUNE_NODE_ID"] = node_id
        env["AUTOTUNE_CHECKPOINT_DIR"] = str(ckpt_output_dir)
        env["AUTOTUNE_PARENT_CHECKPOINT"] = str(parent_ckpt_dir)
        env["AUTOTUNE_MOVE_TYPE"] = move_type
        if self.retention_enabled and self.retention_script:
            env["AUTOTUNE_RETENTION_SCRIPT"] = self.retention_script
            if self.retention_probe_set:
                env["AUTOTUNE_RETENTION_PROBE"] = self.retention_probe_set
        for k, v in spec.get("env", {}).items():
            if k not in ("CUDA_VISIBLE_DEVICES", "AUTOTUNE_GPU"):
                env[k] = str(v)

        # Launch process
        log_path = archive_node / "run.log"
        log_fh = open(log_path, "w")
        if self.run_command:
            cmd = shlex.split(self.run_command)
        else:
            cmd = [sys.executable, self.run_script]

        try:
            process = subprocess.Popen(
                cmd, cwd=str(wt_path), stdout=log_fh, stderr=subprocess.STDOUT, env=env,
            )
        except OSError as e:
            log_fh.close()
            self._mark_crashed(node_id, spec, f"Process launch failed: {e}")
            self.runner.cleanup_worktree(wt_path)
            return

        timeout_hours = spec.get("timeout_hours", self.default_timeout_hours)
        estimated_vram = spec.get("estimated_vram_gb", DEFAULT_VRAM_ESTIMATE_GB)
        exp = RunningExperiment(
            id=node_id, spec=spec, gpu=gpu_id, process=process,
            log_file=log_fh, log_path=log_path,
            started_at=time.time(), timeout_at=time.time() + timeout_hours * 3600,
            estimated_vram_gb=estimated_vram,
        )
        self.running[node_id] = exp
        self.gpu_allocations.setdefault(gpu_id, []).append(node_id)

        # Copy spec to running dir for orphan recovery
        (self.running_dir / f"{node_id}.json").write_text(json.dumps(clean_spec, indent=2))
        logger.info("Launched %s on GPU %d (pid=%d)", node_id, gpu_id, process.pid)

    def _check_running(self) -> None:
        for node_id in list(self.running):
            exp = self.running[node_id]
            retcode = exp.process.poll()
            if retcode is not None:
                self._handle_completion(node_id, retcode)
            elif time.time() > exp.timeout_at:
                self._handle_timeout(node_id)

    def _handle_completion(self, node_id: str, returncode: int) -> None:
        exp = self.running.pop(node_id)
        if node_id in self.gpu_allocations.get(exp.gpu, []):
            self.gpu_allocations[exp.gpu].remove(node_id)
        exp.log_file.close()

        wt_path = self.runner.worktree_path(node_id)
        archive_node = self.archive_dir / node_id
        archive_node.mkdir(parents=True, exist_ok=True)
        result = self.runner.collect_result(wt_path, archive_node)

        # Determine status
        if result and result.get("status") == "completed":
            status = "completed"
        else:
            status = self._infer_failure_status(node_id, returncode)

        elapsed = time.time() - exp.started_at
        metrics = result.get("metrics", {}) if result else {}
        composite = result.get("composite") if result else None
        peak_vram = result.get("peak_vram_mb") if result else None

        completion: dict[str, Any] = {
            "node_id": node_id,
            "status": status,
            "metrics": metrics,
            "composite": composite,
            "elapsed_seconds": round(elapsed, 1),
            "peak_vram_mb": peak_vram,
            "gpu": exp.gpu,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if status != "completed":
            completion["error"] = self._tail_log(exp.log_path, 20)
            completion["log_location"] = str(exp.log_path)

        (self.completed_dir / f"{node_id}.json").write_text(json.dumps(completion, indent=2))

        # [NEW] Collect checkpoint if saved
        if result and result.get("checkpoint_saved"):
            ckpt_output = wt_path / ".checkpoint_output"
            if ckpt_output.exists():
                self.runner.collect_checkpoint(ckpt_output, self.checkpoint_store, node_id)

        # Clean up running spec and worktree
        running_spec = self.running_dir / f"{node_id}.json"
        if running_spec.exists():
            running_spec.unlink()
        self.runner.cleanup_worktree(wt_path)
        self._timed_out.pop(node_id, None)

        # [NEW] Post-completion intelligence
        try:
            self._post_completion_update(node_id, status, completion)
        except Exception as e:
            logger.error("Post-completion update failed for %s: %s", node_id, e)

        self._append_results_tsv(completion)
        logger.info("Completed %s: status=%s elapsed=%.0fs", node_id, status, elapsed)

    def _infer_failure_status(self, node_id: str, returncode: int) -> str:
        if self._timed_out.get(node_id):
            return "timeout"
        log_path = self.archive_dir / node_id / "run.log"
        if log_path.exists():
            tail = log_path.read_text()[-2000:]
            if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
                return "oom"
        if returncode != 0:
            return "crash"
        return "completed"

    def _tail_log(self, log_path: Path, n: int) -> str:
        if not log_path.exists():
            return ""
        lines = log_path.read_text().splitlines()
        return "\n".join(lines[-n:])

    def _post_completion_update(self, node_id: str, status: str, completion: dict) -> None:
        """Update DAG, manifold, and forgetting after experiment completion."""
        from autotune.dag import ExperimentDAG
        from autotune.forgetting import ForgettingDetector
        from autotune.manifold import ManifoldNavigator

        dag_path = self.autotune_dir / "dag.json"
        if not dag_path.exists():
            return

        dag = ExperimentDAG.load(dag_path)
        config = {}
        config_path = self.autotune_dir / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        if status == "completed":
            metrics = completion.get("metrics", {})
            metrics["checkpoint_id"] = node_id if (self.checkpoint_store / node_id).exists() else None
            dag.promote(node_id, metrics)
        else:
            dag.mark_failed(node_id, status, completion.get("error"))

        dag.rerank_pareto()
        dag.recalculate_potential()

        # Update manifold
        params = config.get("parameters", {})
        param_names = list(params.keys())
        if param_names:
            manifold_path = self.autotune_dir / "manifold_state.json"
            if manifold_path.exists():
                manifold = ManifoldNavigator.load(manifold_path)
            else:
                param_bounds = {n: (s.get("range", [0, 1])[0], s.get("range", [0, 1])[1])
                                for n, s in params.items()}
                manifold = ManifoldNavigator(param_names=param_names, param_bounds=param_bounds)
            executed = dag.all_executed()
            if len(executed) >= 10:
                points = [dict(n.theta) for n in executed]
                objectives = [(n.composite or 0.0,) for n in executed]
                manifold.update(points, objectives)
                manifold.save(manifold_path)

        # Check forgetting cliff
        if status == "completed":
            node = dag.get_node(node_id)
            if node and node.retention_score is not None:
                forgetting_path = self.autotune_dir / "forgetting.json"
                threshold = config.get("scoring", {}).get("retention_cliff_threshold", -0.10)
                if forgetting_path.exists():
                    forgetting = ForgettingDetector.load(forgetting_path)
                else:
                    forgetting = ForgettingDetector(cliff_threshold=threshold)
                parents = dag.parents(node_id)
                cliff = forgetting.check_cliff(node, parents)
                if cliff:
                    logger.warning("Forgetting cliff detected for %s: delta=%.3f",
                                   node_id, cliff.delta_retention)
                forgetting.save(forgetting_path)

        # Prune checkpoints
        self._prune_checkpoints(dag)
        dag.save()

    def _handle_timeout(self, node_id: str) -> None:
        exp = self.running[node_id]
        logger.warning("Timeout: killing %s (pid=%d)", node_id, exp.process.pid)
        exp.process.terminate()
        try:
            exp.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            exp.process.kill()
        self._timed_out[node_id] = True
        self._handle_completion(node_id, -9)

    def _mark_crashed(self, node_id: str, spec: dict, error: str) -> None:
        archive_node = self.archive_dir / node_id
        archive_node.mkdir(parents=True, exist_ok=True)
        completion: dict[str, Any] = {
            "node_id": node_id,
            "status": "crash",
            "metrics": {},
            "error": error,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (archive_node / "result.json").write_text(json.dumps(completion, indent=2))
        (archive_node / "spec.json").write_text(json.dumps(
            {k: v for k, v in spec.items() if k != "_file"}, indent=2
        ))
        (self.completed_dir / f"{node_id}.json").write_text(json.dumps(completion, indent=2))
        logger.error("Marked %s as crashed: %s", node_id, error)

    def _prune_checkpoints(self, dag: Any = None) -> None:
        """Delete checkpoints for nodes with pareto_rank >= retention_layers."""
        if dag is None:
            from autotune.dag import ExperimentDAG
            dag_path = self.autotune_dir / "dag.json"
            if not dag_path.exists():
                return
            dag = ExperimentDAG.load(dag_path)

        for node in dag.all_executed():
            if node.pareto_rank is not None and node.pareto_rank >= self.retention_layers:
                ckpt_dir = self.checkpoint_store / node.id
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                    logger.info("Pruned checkpoint: %s (rank=%d)", node.id, node.pareto_rank)

    def _append_results_tsv(self, completion: dict) -> None:
        header = "node_id\ttask_score\tretention_score\tcost\tcomposite\tvram_mb\telapsed_min\tstatus\n"
        if not self.results_tsv.exists() or self.results_tsv.stat().st_size == 0:
            self.results_tsv.write_text(header)
        metrics = completion.get("metrics", {})
        row = (
            f"{completion['node_id']}\t"
            f"{metrics.get('task_score', '')}\t"
            f"{metrics.get('retention_score', '')}\t"
            f"{metrics.get('cost', '')}\t"
            f"{completion.get('composite', '')}\t"
            f"{completion.get('peak_vram_mb', '')}\t"
            f"{round(completion.get('elapsed_seconds', 0) / 60, 1)}\t"
            f"{completion['status']}\n"
        )
        with open(self.results_tsv, "a") as f:
            f.write(row)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/autotune/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add launch, completion, timeout, crash handling, checkpoint pruning"
```

---

## Task 8: Orchestrator Main Loop and CLI Methods

**Files:**
- Modify: `src/autotune/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write tests for tick and cmd methods**

Append to `tests/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py::TestCheckpointPruning tests/test_orchestrator.py::TestCmdMethods -v`
Expected: FAIL

- [ ] **Step 3: Implement tick, run, cmd_start, cmd_status, cmd_stop**

Add to `ExperimentOrchestrator`:

```python
    # ── main loop ──

    def tick(self) -> None:
        self._check_running()
        if not self.draining:
            for spec in self._get_pending():
                needed_gb = spec.get("estimated_vram_gb", DEFAULT_VRAM_ESTIMATE_GB)
                gpu = self._find_best_gpu(needed_gb)
                if gpu is not None and self._pre_launch_check(gpu, needed_gb):
                    self._launch(spec, gpu)
        self._save_state()

    def run(self) -> None:
        self.runner.prune_stale_worktrees()
        self._recover_orphans()

        def _signal_handler(signum, frame):
            logger.info("Received signal %d, shutting down...", signum)
            self._shutdown = True

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        self.pid_file.write_text(str(os.getpid()))
        logger.info(
            "Orchestrator started (pid=%d, gpus=%s, poll=%ds)",
            os.getpid(), list(self.gpu_allocations.keys()), self.poll_interval,
        )

        try:
            while not self._shutdown:
                try:
                    self.tick()
                except Exception:
                    logger.exception("Error in tick")
                time.sleep(self.poll_interval)

            # Graceful drain
            if self.running:
                logger.info("Draining %d running experiments...", len(self.running))
                self.draining = True
                while self.running:
                    self._check_running()
                    time.sleep(5)
        finally:
            if self.pid_file.exists():
                self.pid_file.unlink()
            self._save_state()
            logger.info("Orchestrator stopped.")

    # ── CLI methods ──

    def cmd_start(self) -> None:
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                os.kill(pid, 0)  # check if alive
                print(f"Orchestrator already running (pid={pid})")
                return
            except (OSError, ValueError):
                self.pid_file.unlink(missing_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler(),
            ],
        )
        self.run()

    def cmd_status(self) -> None:
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                os.kill(pid, 0)
                print(f"Orchestrator running (pid={pid})")
            except (OSError, ValueError):
                print("Orchestrator not running (stale PID file)")
        else:
            print("Orchestrator not running (no PID file)")

        if self.gpu_state_file.exists():
            state = json.loads(self.gpu_state_file.read_text())
            print(f"  Last updated: {state.get('last_updated', 'unknown')}")
            print(f"  Queue depth:  {state.get('queue_depth', 0)}")
            print(f"  Running:      {state.get('total_running', 0)}")
            for gpu in state.get("gpus", []):
                print(
                    f"  GPU {gpu['index']}: {gpu.get('schedulable_free_gb', '?')}GB free, "
                    f"{gpu.get('utilization_pct', '?')}% util, "
                    f"running={gpu.get('running', [])}"
                )

    def cmd_stop(self) -> None:
        if not self.pid_file.exists():
            print("Orchestrator not running (no PID file)")
            return
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to orchestrator (pid={pid})")
        except (OSError, ValueError) as e:
            print(f"Failed to stop orchestrator: {e}")
            self.pid_file.unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_orchestrator.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/autotune/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add tick/run main loop and cmd_start/status/stop CLI methods"
```

---

## Task 9: CLI orchestrator Group and checkpoints Command

**Files:**
- Modify: `src/autotune/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write tests for CLI orchestrator and checkpoints commands**

Append to `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_cli.py::TestOrchestratorCLI tests/test_cli.py::TestCheckpoints -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI commands**

Append to `src/autotune/cli.py` (after the `reconcile` command):

```python
@main.group()
def orchestrator() -> None:
    """Manage the GPU scheduler daemon."""


@orchestrator.command("start")
def orch_start() -> None:
    """Start the orchestrator daemon (foreground)."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.orchestrator import ExperimentOrchestrator

    orch = ExperimentOrchestrator(root)
    orch.cmd_start()


@orchestrator.command("stop")
def orch_stop() -> None:
    """Stop the running orchestrator."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.orchestrator import ExperimentOrchestrator

    orch = ExperimentOrchestrator(root)
    orch.cmd_stop()


@orchestrator.command("status")
def orch_status() -> None:
    """Show orchestrator status."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.orchestrator import ExperimentOrchestrator

    orch = ExperimentOrchestrator(root)
    orch.cmd_status()


@main.command()
@click.option("--prune", is_flag=True, help="Delete checkpoints for pareto_rank >= 2")
def checkpoints(prune: bool) -> None:
    """List or prune managed checkpoints."""
    import shutil

    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    store = root / "autotune" / "orchestrator" / "checkpoints"
    if not store.exists():
        click.echo("No checkpoints directory.")
        return

    dag_path = root / "autotune" / "dag.json"
    node_ranks: dict[str, int | None] = {}
    if dag_path.exists():
        from autotune.dag import ExperimentDAG

        dag = ExperimentDAG.load(dag_path)
        for node in dag.all_executed():
            node_ranks[node.id] = node.pareto_rank

    ckpt_dirs = sorted(d for d in store.iterdir() if d.is_dir())
    if not ckpt_dirs:
        click.echo("No checkpoints found.")
        return

    pruned = 0
    for d in ckpt_dirs:
        node_id = d.name
        rank = node_ranks.get(node_id)
        size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)

        if prune and rank is not None and rank >= 2:
            shutil.rmtree(d)
            click.echo(f"  Pruned: {node_id} (rank={rank}, {size_mb:.1f}MB)")
            pruned += 1
        else:
            click.echo(f"  {node_id}: rank={rank}, {size_mb:.1f}MB")

    if prune:
        click.echo(f"Pruned {pruned} checkpoints.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_cli.py::TestOrchestratorCLI tests/test_cli.py::TestCheckpoints -v`
Expected: PASS

- [ ] **Step 5: Run all CLI tests for regression**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_cli.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/autotune/cli.py tests/test_cli.py
git commit -m "feat(cli): add orchestrator start|stop|status and checkpoints --prune commands"
```

---

## Task 10: Final Verification

**Files:** All modified files

- [ ] **Step 1: Run full test suite**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest -v`
Expected: All pass (should be ~110+ tests)

- [ ] **Step 2: Run linter**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run ruff check src/ tests/`
Expected: All checks passed

- [ ] **Step 3: Run formatter**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run ruff format src/ tests/`
Fix any formatting issues, then verify: `uv run ruff format --check src/ tests/`

- [ ] **Step 4: Verify CLI shows all commands**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run autotune --help`
Expected: All 13 commands listed: check, checkpoints, explain, frontier, init, orchestrator, propose, rank, reconcile, status, submit, suggest

- [ ] **Step 5: Verify file sizes under 800 lines**

Run: `wc -l src/autotune/orchestrator.py src/autotune/runner.py src/autotune/cli.py`
Expected: All under 800 lines

- [ ] **Step 6: Commit any lint/format fixes**

```bash
git add -A
git commit -m "chore: fix lint warnings and apply ruff formatting"
```

- [ ] **Step 7: Run existing integration tests to verify no regressions**

Run: `cd "/Users/chenzhang/Alex's Codebases/autoTune" && uv run pytest tests/test_integration.py -v`
Expected: Both E2E tests pass

---

## Summary

| Task | What It Delivers | Files |
|------|-----------------|-------|
| 1 | Checkpoint I/O helpers (_load_state_dict, _save_state_dict) | runner.py, test_runner.py |
| 2 | merge_checkpoints (weighted_avg, compose) | runner.py, test_runner.py |
| 3 | graft_checkpoint (delta-transfer) | runner.py, test_runner.py |
| 4 | Orchestrator scaffold (GPUInfo, query_gpus) | orchestrator.py, test_orchestrator.py |
| 5 | Orchestrator __init__ + config loading | orchestrator.py, test_orchestrator.py |
| 6 | State persistence + orphan recovery + scheduling | orchestrator.py, test_orchestrator.py |
| 7 | Launch + completion + checkpoint management | orchestrator.py, test_orchestrator.py |
| 8 | Main loop (tick/run) + cmd_start/status/stop | orchestrator.py, test_orchestrator.py |
| 9 | CLI orchestrator group + checkpoints command | cli.py, test_cli.py |
| 10 | Final verification (all tests, lint, format) | All files |
