# autoTune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the autoTune finetuning framework — a DAG-based, manifold-navigated, forgetting-aware autonomous experiment system for neural network and LLM finetuning.

**Architecture:** Sovereign Core + Optuna Substrate. Seven core modules (dag, manifold, forgetting, navigator, orchestrator, runner, cli) plus visualization and templates. Overlay model onto existing ML repos. Agent-driven exploration with manifold-guided numerical proposals.

**Tech Stack:** Python 3.11+, Click (CLI), PyYAML, Jinja2, NumPy, Optuna (CMA-ES + NSGA-III), aiohttp + watchdog (viz), ForceGraph3D (frontend)

**Spec:** `docs/specs/2026-03-27-autotune-system-architecture-design.md`

---

## File Structure

```
autoTune/
├── pyproject.toml
├── src/autotune/
│   ├── __init__.py
│   ├── types.py           # Frozen dataclasses: DAGNode, DAGMeta, ScoringConfig, etc.
│   ├── dag.py             # ExperimentDAG: CRUD, Pareto ranking, reconciliation
│   ├── manifold.py        # ManifoldNavigator: CMA-ES covariance, PC extraction
│   ├── forgetting.py      # ForgettingDetector: cliff detection, danger zones
│   ├── navigator.py       # StrategyNavigator: combines dag + manifold + forgetting
│   ├── runner.py          # ExperimentRunner: worktree + checkpoint ops
│   ├── orchestrator.py    # ExperimentOrchestrator: GPU scheduler daemon
│   ├── cli.py             # Click CLI entry point
│   ├── templates/
│   │   ├── config.yaml.j2
│   │   ├── program.md.j2
│   │   ├── learnings.md.j2
│   │   └── .gitignore.j2
│   └── viz/
│       ├── __init__.py
│       ├── server.py
│       └── static/
│           ├── index.html
│           └── app.js
├── tests/
│   ├── __init__.py
│   ├── test_types.py
│   ├── test_dag.py
│   ├── test_manifold.py
│   ├── test_forgetting.py
│   ├── test_navigator.py
│   ├── test_runner.py
│   ├── test_cli.py
│   └── test_integration.py
└── docs/
    └── specs/
        └── 2026-03-27-autotune-system-architecture-design.md
```

---

## Phase 1: Project Scaffolding

### Task 1: Initialize Python Package

**Files:**
- Create: `pyproject.toml`
- Create: `src/autotune/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "autotune"
version = "0.1.0"
description = "Autonomous finetuning framework with DAG-structured checkpoint management and manifold-based hyperparameter navigation"
requires-python = ">=3.11"
dependencies = [
    "click>=8.0",
    "pyyaml>=6.0",
    "jinja2>=3.0",
    "numpy>=1.24",
    "optuna>=4.0",
    "aiohttp>=3.9",
    "watchdog>=3.0",
]

[project.optional-dependencies]
torch = ["torch>=2.0"]
safetensors = ["safetensors>=0.4"]
peft = ["peft>=0.7"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[project.scripts]
autotune = "autotune.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/autotune"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init**

```python
# src/autotune/__init__.py
"""autoTune: Autonomous finetuning framework."""
```

- [ ] **Step 3: Create tests init**

```python
# tests/__init__.py
```

- [ ] **Step 4: Install in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: successful install, `autotune` command available (will fail until cli.py exists, that's fine)

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml src/autotune/__init__.py tests/__init__.py
git commit -m "chore: initialize autoTune package"
```

---

## Phase 2: Core Data Structures

### Task 2: Define Frozen Dataclasses

**Files:**
- Create: `src/autotune/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write tests for type construction and immutability**

```python
# tests/test_types.py
import pytest
from autotune.types import (
    DAGNode, DAGMeta, ScoringConfig, TechniqueStats,
    ManifoldState, DangerZone, MoveProposal, CliffReport,
)


class TestDAGNode:
    def test_create_seed_node(self):
        node = DAGNode(
            id="node_0001",
            parent_ids=(),
            move_type="seed",
            merge_strategy=None,
            merge_weights=None,
            theta={"lr": 2e-4, "rank": 16},
            manifold_coords=None,
            strategy_desc="baseline",
            technique_tags=(),
            task_score=0.72,
            retention_score=0.98,
            cost=1.5,
            composite=0.72,
            checkpoint_id="node_0001",
            depth=0,
            potential=0.0,
            pareto_rank=0,
            node_type="executed",
            status="keep",
            created_at="2026-03-27T00:00:00",
        )
        assert node.id == "node_0001"
        assert node.parent_ids == ()
        assert node.move_type == "seed"

    def test_create_merge_node(self):
        node = DAGNode(
            id="node_0010",
            parent_ids=("node_0005", "node_0008"),
            move_type="merge",
            merge_strategy="weighted_avg",
            merge_weights=(0.6, 0.4),
            theta={"lr": 3e-4, "rank": 24},
            manifold_coords=(0.5, -0.2),
            strategy_desc="merge task-strong with retention-strong",
            technique_tags=("merge_weighted",),
            task_score=None,
            retention_score=None,
            cost=None,
            composite=None,
            checkpoint_id=None,
            depth=3,
            potential=0.0,
            pareto_rank=None,
            node_type="proposed",
            status="pending",
            created_at="2026-03-27T01:00:00",
        )
        assert node.parent_ids == ("node_0005", "node_0008")
        assert node.merge_weights == (0.6, 0.4)

    def test_node_is_frozen(self):
        node = DAGNode(
            id="node_0001", parent_ids=(), move_type="seed",
            merge_strategy=None, merge_weights=None,
            theta={"lr": 2e-4}, manifold_coords=None,
            strategy_desc="baseline", technique_tags=(),
            task_score=0.5, retention_score=None, cost=1.0,
            composite=0.5, checkpoint_id=None, depth=0,
            potential=0.0, pareto_rank=0,
            node_type="executed", status="keep",
            created_at="2026-03-27T00:00:00",
        )
        with pytest.raises(AttributeError):
            node.status = "discard"


class TestScoringConfig:
    def test_defaults(self):
        sc = ScoringConfig(
            exploration_weight=0.005,
            diversity_weight=0.01,
            retention_cliff_threshold=-0.10,
        )
        assert sc.exploration_weight == 0.005


class TestDangerZone:
    def test_create(self):
        dz = DangerZone(
            center=(0.5, -0.3),
            radius=0.15,
            cause_technique="lr_increase",
            severity=0.12,
        )
        assert dz.cause_technique == "lr_increase"


class TestMoveProposal:
    def test_branch_proposal(self):
        mp = MoveProposal(
            move_type="branch",
            parent_ids=("node_0005",),
            suggested_theta={"lr": 4.2e-4, "rank": 28},
            merge_strategy=None,
            merge_weights=None,
            direction_label="PC1 (+capacity)",
            score=0.847,
            reasoning="high retention budget, PC1 consistently improves task",
        )
        assert mp.move_type == "branch"
        assert mp.score == 0.847
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autotune.types'`

- [ ] **Step 3: Implement types.py**

```python
# src/autotune/types.py
"""Frozen dataclasses for autoTune's core data model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    exploration_weight: float
    diversity_weight: float
    retention_cliff_threshold: float


@dataclass(frozen=True)
class DAGMeta:
    next_id: int
    total_executed: int
    total_proposed: int
    best_composite: float | None
    best_node_id: str | None
    objective_count: int
    scoring: ScoringConfig


@dataclass(frozen=True)
class DAGNode:
    id: str
    parent_ids: tuple[str, ...]
    move_type: str  # "branch" | "merge" | "graft" | "seed"

    merge_strategy: str | None
    merge_weights: tuple[float, ...] | None

    theta: dict[str, float]
    manifold_coords: tuple[float, ...] | None

    strategy_desc: str
    technique_tags: tuple[str, ...]

    task_score: float | None
    retention_score: float | None
    cost: float | None
    composite: float | None

    checkpoint_id: str | None

    depth: int
    potential: float
    pareto_rank: int | None

    node_type: str  # "executed" | "proposed"
    status: str
    created_at: str


@dataclass(frozen=True)
class TechniqueStats:
    times_tried: int
    avg_task_delta: float
    avg_retention_delta: float
    best_parent_delta: float


@dataclass(frozen=True)
class DangerZone:
    center: tuple[float, ...]
    radius: float
    cause_technique: str
    severity: float


@dataclass(frozen=True)
class ManifoldState:
    param_names: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]
    principal_components: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    effective_dim: int
    mean: tuple[float, ...]
    step_sizes: tuple[float, ...]
    generation: int


@dataclass(frozen=True)
class MoveProposal:
    move_type: str  # "branch" | "merge" | "graft"
    parent_ids: tuple[str, ...]
    suggested_theta: dict[str, float]
    merge_strategy: str | None
    merge_weights: tuple[float, ...] | None
    direction_label: str
    score: float
    reasoning: str


@dataclass(frozen=True)
class CliffReport:
    node_id: str
    parent_ids: tuple[str, ...]
    delta_retention: float
    cause_techniques: tuple[str, ...]
    severity: float
    danger_zone: DangerZone
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/types.py tests/test_types.py
git commit -m "feat: add frozen dataclasses for core data model"
```

---

## Phase 3: Leaf Modules (No Internal Dependencies)

### Task 3: Implement ExperimentDAG — Node CRUD

**Files:**
- Create: `src/autotune/dag.py`
- Create: `tests/test_dag.py`

- [ ] **Step 1: Write tests for node creation and queries**

```python
# tests/test_dag.py
import json
import tempfile
from pathlib import Path

import pytest
from autotune.dag import ExperimentDAG
from autotune.types import ScoringConfig


def _make_dag(tmp_path: Path) -> ExperimentDAG:
    path = tmp_path / "dag.json"
    scoring = ScoringConfig(
        exploration_weight=0.005,
        diversity_weight=0.01,
        retention_cliff_threshold=-0.10,
    )
    return ExperimentDAG.create(path, scoring, objective_count=3)


class TestNodeCRUD:
    def test_add_seed(self, tmp_path):
        dag = _make_dag(tmp_path)
        node = dag.add_seed(
            theta={"lr": 2e-4, "rank": 16},
            description="baseline",
            techniques=("baseline",),
            composite=0.72,
            task_score=0.72,
            retention_score=0.98,
            cost=1.5,
        )
        assert node.id == "node_0001"
        assert node.move_type == "seed"
        assert node.status == "keep"
        assert node.parent_ids == ()

    def test_add_proposed_branch(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(
            theta={"lr": 2e-4}, description="base", techniques=(),
            composite=0.7, task_score=0.7, retention_score=0.9, cost=1.0,
        )
        node = dag.add_proposed(
            parent_ids=("node_0001",),
            move_type="branch",
            description="increase lr",
            techniques=("lr_increase",),
            theta={"lr": 5e-4},
        )
        assert node.id == "node_0002"
        assert node.node_type == "proposed"
        assert node.status == "pending"
        assert node.parent_ids == ("node_0001",)

    def test_add_proposed_merge(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={"lr": 2e-4}, description="a", techniques=(),
                     composite=0.7, task_score=0.7, retention_score=0.9, cost=1.0)
        dag.add_seed(theta={"lr": 5e-4}, description="b", techniques=(),
                     composite=0.8, task_score=0.8, retention_score=0.8, cost=1.2)
        node = dag.add_proposed(
            parent_ids=("node_0001", "node_0002"),
            move_type="merge",
            description="merge a+b",
            techniques=("merge_weighted",),
            theta={"lr": 3.5e-4},
            merge_strategy="weighted_avg",
            merge_weights=(0.5, 0.5),
        )
        assert node.move_type == "merge"
        assert node.parent_ids == ("node_0001", "node_0002")
        assert node.merge_strategy == "weighted_avg"

    def test_get_node(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={"lr": 2e-4}, description="base", techniques=(),
                     composite=0.7, task_score=0.7, retention_score=0.9, cost=1.0)
        assert dag.get_node("node_0001") is not None
        assert dag.get_node("nonexistent") is None

    def test_parents_and_children(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={"lr": 2e-4}, description="base", techniques=(),
                     composite=0.7, task_score=0.7, retention_score=0.9, cost=1.0)
        dag.add_proposed(
            parent_ids=("node_0001",), move_type="branch",
            description="child", techniques=(), theta={"lr": 3e-4},
        )
        parents = dag.parents("node_0002")
        assert len(parents) == 1
        assert parents[0].id == "node_0001"
        children = dag.children("node_0001")
        assert len(children) == 1
        assert children[0].id == "node_0002"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dag.py::TestNodeCRUD -v`
Expected: FAIL

- [ ] **Step 3: Implement dag.py — node CRUD and queries**

```python
# src/autotune/dag.py
"""Experiment DAG: node CRUD, Pareto ranking, reconciliation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from autotune.types import (
    DAGMeta, DAGNode, ScoringConfig, TechniqueStats,
)


class ExperimentDAG:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self._path = path
        self._data = data

    @staticmethod
    def create(path: Path, scoring: ScoringConfig, objective_count: int = 3) -> ExperimentDAG:
        data: dict[str, Any] = {
            "schema_version": 1,
            "meta": {
                "next_id": 1,
                "total_executed": 0,
                "total_proposed": 0,
                "best_composite": None,
                "best_node_id": None,
                "objective_count": objective_count,
                "scoring": {
                    "exploration_weight": scoring.exploration_weight,
                    "diversity_weight": scoring.diversity_weight,
                    "retention_cliff_threshold": scoring.retention_cliff_threshold,
                },
            },
            "nodes": {},
            "technique_stats": {},
        }
        return ExperimentDAG(path, data)

    @staticmethod
    def load(path: Path) -> ExperimentDAG:
        with open(path) as f:
            data = json.load(f)
        return ExperimentDAG(path, data)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.rename(tmp, str(self._path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ── helpers ──

    def _next_id(self) -> str:
        nid = self._data["meta"]["next_id"]
        self._data["meta"]["next_id"] = nid + 1
        return f"node_{nid:04d}"

    def _node_to_dataclass(self, d: dict[str, Any]) -> DAGNode:
        return DAGNode(
            id=d["id"],
            parent_ids=tuple(d["parent_ids"]),
            move_type=d["move_type"],
            merge_strategy=d.get("merge_strategy"),
            merge_weights=tuple(d["merge_weights"]) if d.get("merge_weights") else None,
            theta=dict(d["theta"]),
            manifold_coords=tuple(d["manifold_coords"]) if d.get("manifold_coords") else None,
            strategy_desc=d["strategy_desc"],
            technique_tags=tuple(d["technique_tags"]),
            task_score=d.get("task_score"),
            retention_score=d.get("retention_score"),
            cost=d.get("cost"),
            composite=d.get("composite"),
            checkpoint_id=d.get("checkpoint_id"),
            depth=d["depth"],
            potential=d.get("potential", 0.0),
            pareto_rank=d.get("pareto_rank"),
            node_type=d["node_type"],
            status=d["status"],
            created_at=d["created_at"],
        )

    def _compute_depth(self, parent_ids: tuple[str, ...]) -> int:
        if not parent_ids:
            return 0
        return 1 + max(
            self._data["nodes"][pid]["depth"]
            for pid in parent_ids
            if pid in self._data["nodes"]
        )

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    # ── node operations ──

    def add_seed(
        self,
        theta: dict[str, float],
        description: str,
        techniques: tuple[str, ...],
        composite: float,
        task_score: float,
        retention_score: float | None = None,
        cost: float | None = None,
        checkpoint_id: str | None = None,
    ) -> DAGNode:
        node_id = self._next_id()
        node_data: dict[str, Any] = {
            "id": node_id,
            "parent_ids": [],
            "move_type": "seed",
            "merge_strategy": None,
            "merge_weights": None,
            "theta": dict(theta),
            "manifold_coords": None,
            "strategy_desc": description,
            "technique_tags": list(techniques),
            "task_score": task_score,
            "retention_score": retention_score,
            "cost": cost,
            "composite": composite,
            "checkpoint_id": checkpoint_id,
            "depth": 0,
            "potential": 0.0,
            "pareto_rank": 0,
            "node_type": "executed",
            "status": "keep",
            "created_at": self._timestamp(),
        }
        self._data["nodes"][node_id] = node_data
        self._data["meta"]["total_executed"] += 1
        if (
            self._data["meta"]["best_composite"] is None
            or composite > self._data["meta"]["best_composite"]
        ):
            self._data["meta"]["best_composite"] = composite
            self._data["meta"]["best_node_id"] = node_id
        return self._node_to_dataclass(node_data)

    def add_proposed(
        self,
        parent_ids: tuple[str, ...],
        move_type: str,
        description: str,
        techniques: tuple[str, ...],
        theta: dict[str, float],
        merge_strategy: str | None = None,
        merge_weights: tuple[float, ...] | None = None,
    ) -> DAGNode:
        node_id = self._next_id()
        depth = self._compute_depth(parent_ids)
        node_data: dict[str, Any] = {
            "id": node_id,
            "parent_ids": list(parent_ids),
            "move_type": move_type,
            "merge_strategy": merge_strategy,
            "merge_weights": list(merge_weights) if merge_weights else None,
            "theta": dict(theta),
            "manifold_coords": None,
            "strategy_desc": description,
            "technique_tags": list(techniques),
            "task_score": None,
            "retention_score": None,
            "cost": None,
            "composite": None,
            "checkpoint_id": None,
            "depth": depth,
            "potential": 0.0,
            "pareto_rank": None,
            "node_type": "proposed",
            "status": "pending",
            "created_at": self._timestamp(),
        }
        self._data["nodes"][node_id] = node_data
        self._data["meta"]["total_proposed"] += 1
        return self._node_to_dataclass(node_data)

    def mark_running(self, node_id: str) -> DAGNode:
        node = self._data["nodes"][node_id]
        if node["status"] != "pending":
            msg = f"Cannot mark {node_id} as running: status is {node['status']}"
            raise ValueError(msg)
        node["status"] = "running"
        return self._node_to_dataclass(node)

    def promote(self, node_id: str, metrics: dict[str, Any]) -> DAGNode:
        node = self._data["nodes"][node_id]
        node["node_type"] = "executed"
        node["task_score"] = metrics.get("task_score")
        node["retention_score"] = metrics.get("retention_score")
        node["cost"] = metrics.get("cost")
        node["composite"] = metrics.get("composite")
        node["checkpoint_id"] = metrics.get("checkpoint_id")
        # status will be set by rerank_pareto(); default to "keep" for now
        node["status"] = "keep"
        self._data["meta"]["total_executed"] += 1
        best = self._data["meta"]["best_composite"]
        if best is None or (node["composite"] and node["composite"] > best):
            self._data["meta"]["best_composite"] = node["composite"]
            self._data["meta"]["best_node_id"] = node_id
        return self._node_to_dataclass(node)

    def mark_failed(self, node_id: str, status: str, error: str | None = None) -> DAGNode:
        node = self._data["nodes"][node_id]
        node["node_type"] = "executed"
        node["status"] = status
        self._data["meta"]["total_executed"] += 1
        return self._node_to_dataclass(node)

    def cancel(self, node_id: str) -> DAGNode:
        node = self._data["nodes"][node_id]
        node["status"] = "cancelled"
        return self._node_to_dataclass(node)

    # ── queries ──

    def get_node(self, node_id: str) -> DAGNode | None:
        d = self._data["nodes"].get(node_id)
        return self._node_to_dataclass(d) if d else None

    def parents(self, node_id: str) -> tuple[DAGNode, ...]:
        node = self._data["nodes"].get(node_id)
        if not node:
            return ()
        return tuple(
            self._node_to_dataclass(self._data["nodes"][pid])
            for pid in node["parent_ids"]
            if pid in self._data["nodes"]
        )

    def children(self, node_id: str) -> list[DAGNode]:
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if node_id in n["parent_ids"]
        ]

    def ancestors(self, node_id: str) -> list[DAGNode]:
        visited: set[str] = set()
        result: list[DAGNode] = []

        def _walk(nid: str) -> None:
            node = self._data["nodes"].get(nid)
            if not node:
                return
            for pid in node["parent_ids"]:
                if pid not in visited:
                    visited.add(pid)
                    result.append(self._node_to_dataclass(self._data["nodes"][pid]))
                    _walk(pid)

        _walk(node_id)
        return result

    def kept_nodes(self) -> list[DAGNode]:
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if n.get("pareto_rank") is not None and n["pareto_rank"] <= 1
        ]

    def all_executed(self) -> list[DAGNode]:
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if n["node_type"] == "executed" and n["status"] not in ("oom", "crash", "timeout")
        ]

    def all_proposed(self, status: str | None = None) -> list[DAGNode]:
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if n["node_type"] == "proposed"
            and (status is None or n["status"] == status)
        ]

    def meta(self) -> DAGMeta:
        m = self._data["meta"]
        sc = m["scoring"]
        return DAGMeta(
            next_id=m["next_id"],
            total_executed=m["total_executed"],
            total_proposed=m["total_proposed"],
            best_composite=m["best_composite"],
            best_node_id=m["best_node_id"],
            objective_count=m["objective_count"],
            scoring=ScoringConfig(
                exploration_weight=sc["exploration_weight"],
                diversity_weight=sc["diversity_weight"],
                retention_cliff_threshold=sc["retention_cliff_threshold"],
            ),
        )

    # ── technique stats ──

    def technique_stats(self, tag: str) -> TechniqueStats:
        ts = self._data["technique_stats"].get(tag)
        if not ts:
            return TechniqueStats(
                times_tried=0, avg_task_delta=0.0,
                avg_retention_delta=0.0, best_parent_delta=0.0,
            )
        return TechniqueStats(
            times_tried=ts["times_tried"],
            avg_task_delta=ts["avg_task_delta"],
            avg_retention_delta=ts["avg_retention_delta"],
            best_parent_delta=ts["best_parent_delta"],
        )

    def update_technique_stats(
        self, tag: str, task_delta: float, retention_delta: float,
    ) -> None:
        ts = self._data["technique_stats"]
        if tag not in ts:
            ts[tag] = {
                "times_tried": 0,
                "avg_task_delta": 0.0,
                "avg_retention_delta": 0.0,
                "best_parent_delta": 0.0,
                "_total_task_delta": 0.0,
                "_total_retention_delta": 0.0,
            }
        entry = ts[tag]
        entry["times_tried"] += 1
        entry["_total_task_delta"] += task_delta
        entry["_total_retention_delta"] += retention_delta
        entry["avg_task_delta"] = entry["_total_task_delta"] / entry["times_tried"]
        entry["avg_retention_delta"] = entry["_total_retention_delta"] / entry["times_tried"]
        if task_delta > entry["best_parent_delta"]:
            entry["best_parent_delta"] = task_delta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dag.py::TestNodeCRUD -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/dag.py tests/test_dag.py
git commit -m "feat(dag): add ExperimentDAG with node CRUD and queries"
```

---

### Task 4: Implement ExperimentDAG — Pareto Ranking

**Files:**
- Modify: `src/autotune/dag.py`
- Modify: `tests/test_dag.py`

- [ ] **Step 1: Write tests for Pareto ranking**

```python
# append to tests/test_dag.py

class TestParetoRanking:
    def test_non_dominated_sorting_3obj(self, tmp_path):
        dag = _make_dag(tmp_path)
        # A dominates nothing, B dominates nothing, C dominated by both
        dag.add_seed(theta={}, description="A", techniques=(),
                     composite=0.9, task_score=0.9, retention_score=0.7, cost=2.0)
        dag.add_seed(theta={}, description="B", techniques=(),
                     composite=0.8, task_score=0.7, retention_score=0.9, cost=1.0)
        dag.add_seed(theta={}, description="C", techniques=(),
                     composite=0.6, task_score=0.6, retention_score=0.6, cost=3.0)
        dag.rerank_pareto()
        a = dag.get_node("node_0001")
        b = dag.get_node("node_0002")
        c = dag.get_node("node_0003")
        assert a.pareto_rank == 0
        assert b.pareto_rank == 0
        assert c.pareto_rank == 1
        assert a.status == "keep"
        assert b.status == "keep"
        assert c.status == "keep"  # rank 1 is still "keep"

    def test_rank_2_is_discard(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={}, description="best", techniques=(),
                     composite=0.9, task_score=0.9, retention_score=0.9, cost=1.0)
        dag.add_seed(theta={}, description="mid", techniques=(),
                     composite=0.7, task_score=0.7, retention_score=0.7, cost=2.0)
        dag.add_seed(theta={}, description="worst", techniques=(),
                     composite=0.5, task_score=0.5, retention_score=0.5, cost=3.0)
        dag.rerank_pareto()
        assert dag.get_node("node_0001").pareto_rank == 0
        assert dag.get_node("node_0002").pareto_rank == 1
        assert dag.get_node("node_0003").pareto_rank == 2
        assert dag.get_node("node_0003").status == "discard"

    def test_2obj_mode_no_retention(self, tmp_path):
        path = tmp_path / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(path, scoring, objective_count=2)
        dag.add_seed(theta={}, description="A", techniques=(),
                     composite=0.9, task_score=0.9, cost=2.0)
        dag.add_seed(theta={}, description="B", techniques=(),
                     composite=0.7, task_score=0.7, cost=1.0)
        dag.rerank_pareto()
        # both non-dominated: A better task, B better cost
        assert dag.get_node("node_0001").pareto_rank == 0
        assert dag.get_node("node_0002").pareto_rank == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dag.py::TestParetoRanking -v`
Expected: FAIL — `rerank_pareto` not defined

- [ ] **Step 3: Implement rerank_pareto in dag.py**

Add this method to `ExperimentDAG`:

```python
    def rerank_pareto(self) -> None:
        executed = [
            n for n in self._data["nodes"].values()
            if n["node_type"] == "executed"
            and n["status"] not in ("oom", "crash", "timeout")
        ]
        if not executed:
            return
        obj_count = self._data["meta"]["objective_count"]
        remaining = list(executed)
        rank = 0
        while remaining:
            front: list[dict[str, Any]] = []
            for candidate in remaining:
                dominated = False
                for other in remaining:
                    if other["id"] == candidate["id"]:
                        continue
                    if self._dominates(other, candidate, obj_count):
                        dominated = True
                        break
                if not dominated:
                    front.append(candidate)
            for node in front:
                node["pareto_rank"] = rank
                node["status"] = "keep" if rank <= 1 else "discard"
            remaining = [n for n in remaining if n not in front]
            rank += 1

    @staticmethod
    def _dominates(a: dict, b: dict, obj_count: int) -> bool:
        task_a = a.get("task_score") or 0.0
        task_b = b.get("task_score") or 0.0
        cost_a = a.get("cost") or float("inf")
        cost_b = b.get("cost") or float("inf")
        if obj_count == 3:
            ret_a = a.get("retention_score") or 0.0
            ret_b = b.get("retention_score") or 0.0
            at_least_as_good = (
                task_a >= task_b and ret_a >= ret_b and cost_a <= cost_b
            )
            strictly_better = (
                task_a > task_b or ret_a > ret_b or cost_a < cost_b
            )
        else:
            at_least_as_good = task_a >= task_b and cost_a <= cost_b
            strictly_better = task_a > task_b or cost_a < cost_b
        return at_least_as_good and strictly_better
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dag.py::TestParetoRanking -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/dag.py tests/test_dag.py
git commit -m "feat(dag): add NSGA-II style non-dominated Pareto ranking"
```

---

### Task 5: Implement ExperimentDAG — UCB Scoring & Persistence

**Files:**
- Modify: `src/autotune/dag.py`
- Modify: `tests/test_dag.py`

- [ ] **Step 1: Write tests for UCB scoring and save/load**

```python
# append to tests/test_dag.py
import math


class TestUCBScoring:
    def test_potential_increases_with_fewer_children(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={"lr": 2e-4}, description="A", techniques=(),
                     composite=0.8, task_score=0.8, retention_score=0.9, cost=1.0)
        dag.add_seed(theta={"lr": 3e-4}, description="B", techniques=(),
                     composite=0.8, task_score=0.8, retention_score=0.9, cost=1.0)
        # Give A two children, B zero
        dag.add_proposed(parent_ids=("node_0001",), move_type="branch",
                         description="c1", techniques=(), theta={})
        dag.add_proposed(parent_ids=("node_0001",), move_type="branch",
                         description="c2", techniques=(), theta={})
        dag.recalculate_potential()
        a = dag.get_node("node_0001")
        b = dag.get_node("node_0002")
        assert b.potential > a.potential  # B has no children, more to explore

    def test_retention_health_penalty(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={}, description="healthy", techniques=(),
                     composite=0.8, task_score=0.8, retention_score=0.95, cost=1.0)
        dag.add_seed(theta={}, description="forgetting", techniques=(),
                     composite=0.8, task_score=0.8, retention_score=0.05, cost=1.0)
        dag.recalculate_potential()
        healthy = dag.get_node("node_0001")
        forgetting = dag.get_node("node_0002")
        assert healthy.potential > forgetting.potential


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={"lr": 2e-4, "rank": 16}, description="base",
                     techniques=("baseline",), composite=0.72,
                     task_score=0.72, retention_score=0.98, cost=1.5)
        dag.save()
        loaded = ExperimentDAG.load(tmp_path / "dag.json")
        node = loaded.get_node("node_0001")
        assert node is not None
        assert node.theta == {"lr": 2e-4, "rank": 16}
        assert node.composite == 0.72

    def test_save_is_atomic(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={}, description="base", techniques=(),
                     composite=0.5, task_score=0.5, retention_score=0.9, cost=1.0)
        dag.save()
        # File should exist and be valid JSON
        path = tmp_path / "dag.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "nodes" in data
        assert "node_0001" in data["nodes"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dag.py::TestUCBScoring -v && uv run pytest tests/test_dag.py::TestPersistence -v`
Expected: FAIL — `recalculate_potential` not defined

- [ ] **Step 3: Implement recalculate_potential in dag.py**

Add this method to `ExperimentDAG`:

```python
    def recalculate_potential(self) -> None:
        meta = self._data["meta"]
        total = meta["total_executed"]
        w_e = meta["scoring"]["exploration_weight"]
        cliff = meta["scoring"]["retention_cliff_threshold"]
        if total < 1:
            return
        for node in self._data["nodes"].values():
            if node["node_type"] != "executed":
                continue
            if node["status"] in ("oom", "crash", "timeout"):
                node["potential"] = 0.0
                continue
            composite = node.get("composite") or 0.0
            child_count = sum(
                1 for n in self._data["nodes"].values()
                if node["id"] in n["parent_ids"]
            )
            exploration = w_e * math.sqrt(math.log(total) / (1 + child_count))
            ret = node.get("retention_score")
            if ret is None:
                health = 1.0
            elif ret >= abs(cliff):
                health = ret
            else:
                health = ret * 0.5
            node["potential"] = (composite + exploration) * health
```

Add the `import math` at the top of `dag.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dag.py::TestUCBScoring tests/test_dag.py::TestPersistence -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/dag.py tests/test_dag.py
git commit -m "feat(dag): add UCB scoring with retention health and save/load persistence"
```

---

### Task 6: Implement ExperimentDAG — Reconciliation

**Files:**
- Modify: `src/autotune/dag.py`
- Modify: `tests/test_dag.py`

- [ ] **Step 1: Write tests for reconciliation**

```python
# append to tests/test_dag.py

class TestReconciliation:
    def test_promote_from_completed_dir(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={"lr": 2e-4}, description="base", techniques=(),
                     composite=0.7, task_score=0.7, retention_score=0.9, cost=1.0)
        dag.add_proposed(parent_ids=("node_0001",), move_type="branch",
                         description="test", techniques=("lr_increase",),
                         theta={"lr": 5e-4})
        dag.mark_running("node_0002")

        # Create orchestrator directories
        queue = tmp_path / "queue"
        running = tmp_path / "running"
        completed = tmp_path / "completed"
        archive = tmp_path / "archive"
        for d in (queue, running, completed, archive):
            d.mkdir()

        # Write completion notification
        completion = {
            "node_id": "node_0002",
            "status": "completed",
            "metrics": {
                "task_score": 0.85,
                "retention_score": 0.88,
                "cost": 1.2,
                "composite": 0.85,
            },
        }
        (completed / "node_0002.json").write_text(json.dumps(completion))

        dag.reconcile(queue, running, completed, archive)
        node = dag.get_node("node_0002")
        assert node.node_type == "executed"
        assert node.task_score == 0.85
        assert node.composite == 0.85

    def test_stale_running_reset_to_pending(self, tmp_path):
        dag = _make_dag(tmp_path)
        dag.add_seed(theta={}, description="base", techniques=(),
                     composite=0.7, task_score=0.7, retention_score=0.9, cost=1.0)
        dag.add_proposed(parent_ids=("node_0001",), move_type="branch",
                         description="test", techniques=(), theta={})
        dag.mark_running("node_0002")

        queue = tmp_path / "queue"
        running = tmp_path / "running"
        completed = tmp_path / "completed"
        archive = tmp_path / "archive"
        for d in (queue, running, completed, archive):
            d.mkdir()

        # node_0002 is "running" in DAG but NOT in orchestrator dirs
        dag.reconcile(queue, running, completed, archive)
        node = dag.get_node("node_0002")
        assert node.status == "pending"  # reset stale running
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dag.py::TestReconciliation -v`
Expected: FAIL — `reconcile` not defined

- [ ] **Step 3: Implement reconcile in dag.py**

Add this method to `ExperimentDAG`:

```python
    def reconcile(
        self,
        queue_dir: Path,
        running_dir: Path,
        completed_dir: Path,
        archive_dir: Path,
    ) -> None:
        # Build set of node IDs the orchestrator knows about
        orch_ids: set[str] = set()
        for d in (queue_dir, running_dir):
            if d.exists():
                for f in d.iterdir():
                    if f.suffix == ".json":
                        orch_ids.add(f.stem)

        # Process completions
        if completed_dir.exists():
            for f in sorted(completed_dir.iterdir()):
                if f.suffix != ".json":
                    continue
                completion = json.loads(f.read_text())
                node_id = completion["node_id"]
                node = self._data["nodes"].get(node_id)
                if not node or node["node_type"] == "executed":
                    continue
                metrics = completion.get("metrics", {})
                status = completion.get("status", "completed")
                if status == "completed":
                    self.promote(node_id, metrics)
                else:
                    self.mark_failed(node_id, status)

        # Reset stale running nodes
        for node in self._data["nodes"].values():
            if (
                node["node_type"] == "proposed"
                and node["status"] == "running"
                and node["id"] not in orch_ids
            ):
                node["status"] = "pending"

        # Re-score everything
        self.rerank_pareto()
        self.recalculate_potential()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dag.py::TestReconciliation -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/dag.py tests/test_dag.py
git commit -m "feat(dag): add reconciliation with completion processing and stale reset"
```

---

### Task 7: Implement ManifoldNavigator

**Files:**
- Create: `src/autotune/manifold.py`
- Create: `tests/test_manifold.py`

- [ ] **Step 1: Write tests for manifold learning and proposals**

```python
# tests/test_manifold.py
import numpy as np
import pytest
from autotune.manifold import ManifoldNavigator


class TestManifoldUpdate:
    def test_learns_covariance_from_points(self):
        nav = ManifoldNavigator(
            param_names=("lr", "rank"),
            param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
        )
        # Points along a diagonal — lr and rank positively correlated
        points = [
            {"lr": 1e-4, "rank": 8},
            {"lr": 2e-4, "rank": 16},
            {"lr": 3e-4, "rank": 24},
            {"lr": 4e-4, "rank": 32},
            {"lr": 5e-4, "rank": 40},
            {"lr": 6e-4, "rank": 48},
            {"lr": 7e-4, "rank": 56},
            {"lr": 8e-4, "rank": 60},
            {"lr": 9e-4, "rank": 62},
            {"lr": 1e-3, "rank": 64},
        ]
        objectives = [(0.6 + i * 0.03,) for i in range(10)]
        nav.update(points, objectives)
        state = nav.state()
        assert state.effective_dim >= 1
        # PC1 should capture most variance (diagonal direction)
        assert state.eigenvalues[0] > state.eigenvalues[1] * 2

    def test_not_enough_points_stays_generation_0(self):
        nav = ManifoldNavigator(
            param_names=("lr", "rank"),
            param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
        )
        nav.update(
            [{"lr": 2e-4, "rank": 16}],
            [(0.7,)],
        )
        assert nav.state().generation == 0  # not enough points


class TestManifoldProposals:
    def _make_trained_nav(self) -> ManifoldNavigator:
        nav = ManifoldNavigator(
            param_names=("lr", "rank"),
            param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
        )
        points = [{"lr": i * 1e-4, "rank": 8 + i * 6} for i in range(1, 11)]
        objectives = [(0.5 + i * 0.04,) for i in range(10)]
        nav.update(points, objectives)
        return nav

    def test_propose_direction_within_bounds(self):
        nav = self._make_trained_nav()
        from_theta = {"lr": 5e-4, "rank": 32}
        result = nav.propose_direction(from_theta)
        assert 1e-5 <= result["lr"] <= 1e-3
        assert 4 <= result["rank"] <= 64

    def test_propose_candidates_returns_n(self):
        nav = self._make_trained_nav()
        candidates = nav.propose_candidates({"lr": 5e-4, "rank": 32}, n=5)
        assert len(candidates) == 5
        for c in candidates:
            assert "lr" in c
            assert "rank" in c


class TestManifoldPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        nav = ManifoldNavigator(
            param_names=("lr", "rank"),
            param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
        )
        points = [{"lr": i * 1e-4, "rank": 8 + i * 6} for i in range(1, 11)]
        objectives = [(0.5 + i * 0.04,) for i in range(10)]
        nav.update(points, objectives)
        path = tmp_path / "manifold_state.json"
        nav.save(path)
        loaded = ManifoldNavigator.load(path)
        assert loaded.state().generation == nav.state().generation
        assert loaded.state().effective_dim == nav.state().effective_dim
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manifold.py -v`
Expected: FAIL

- [ ] **Step 3: Implement manifold.py**

```python
# src/autotune/manifold.py
"""Manifold navigator: CMA-ES covariance learning and PC-based proposals."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from autotune.types import ManifoldState

_MIN_POINTS = 10
_VARIANCE_THRESHOLD = 0.95
_COVARIANCE_LR = 0.3


class ManifoldNavigator:
    def __init__(
        self,
        param_names: tuple[str, ...] | list[str],
        param_bounds: dict[str, tuple[float, float]],
    ) -> None:
        self._param_names = tuple(param_names)
        self._param_bounds = dict(param_bounds)
        d = len(self._param_names)
        self._covariance = np.eye(d)
        self._eigenvalues = np.ones(d)
        self._eigenvectors = np.eye(d)
        self._mean = np.full(d, 0.5)
        self._step_sizes = np.full(d, 0.1)
        self._effective_dim = d
        self._generation = 0

    # ── normalization ──

    def _normalize(self, theta: dict[str, float]) -> np.ndarray:
        result = np.zeros(len(self._param_names))
        for i, name in enumerate(self._param_names):
            lo, hi = self._param_bounds[name]
            val = theta.get(name, (lo + hi) / 2)
            result[i] = (val - lo) / (hi - lo) if hi > lo else 0.5
        return np.clip(result, 0.0, 1.0)

    def _denormalize(self, vec: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = {}
        for i, name in enumerate(self._param_names):
            lo, hi = self._param_bounds[name]
            result[name] = lo + float(np.clip(vec[i], 0.0, 1.0)) * (hi - lo)
        return result

    # ── learning ──

    def update(
        self,
        points: list[dict[str, float]],
        objectives: list[tuple[float, ...]],
    ) -> ManifoldState:
        if len(points) < _MIN_POINTS:
            return self.state()

        # Normalize all points
        normed = np.array([self._normalize(p) for p in points])
        scores = np.array([obj[0] for obj in objectives])

        # Rank by score, take top half
        ranked_idx = np.argsort(scores)[::-1]
        mu = max(len(ranked_idx) // 2, 2)
        top_idx = ranked_idx[:mu]

        # Weights (log-linear decrease)
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = raw_weights / raw_weights.sum()

        # Weighted mean
        top_points = normed[top_idx]
        new_mean = weights @ top_points

        # Update covariance
        diffs = top_points - self._mean
        weighted_outer = sum(
            w * np.outer(d, d) for w, d in zip(weights, diffs)
        )
        self._covariance = (
            (1 - _COVARIANCE_LR) * self._covariance
            + _COVARIANCE_LR * weighted_outer
        )
        self._mean = new_mean

        # Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance)
        sort_idx = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[sort_idx]
        self._eigenvectors = eigenvectors[:, sort_idx]

        # Effective dimensionality
        total_var = self._eigenvalues.sum()
        if total_var > 0:
            cumulative = np.cumsum(self._eigenvalues) / total_var
            self._effective_dim = int(np.searchsorted(cumulative, _VARIANCE_THRESHOLD)) + 1
        else:
            self._effective_dim = len(self._param_names)

        self._generation += 1
        return self.state()

    # ── proposals ──

    def propose_direction(
        self,
        from_theta: dict[str, float],
        pc_index: int | None = None,
    ) -> dict[str, float]:
        from_norm = self._normalize(from_theta)
        if self._generation == 0:
            # Random within bounds
            direction = np.random.randn(len(self._param_names))
            direction /= np.linalg.norm(direction)
            step = 0.1
        else:
            if pc_index is None:
                # Sample PC weighted by eigenvalue
                probs = self._eigenvalues[:self._effective_dim]
                probs = probs / probs.sum()
                pc_index = int(np.random.choice(self._effective_dim, p=probs))
            direction = self._eigenvectors[:, pc_index]
            step = float(self._step_sizes[pc_index])

        new_norm = from_norm + step * direction
        new_norm = np.clip(new_norm, 0.0, 1.0)
        return self._denormalize(new_norm)

    def propose_candidates(
        self,
        from_theta: dict[str, float],
        n: int,
    ) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        from_norm = self._normalize(from_theta)
        for _ in range(n):
            if self._generation == 0:
                sample = np.random.uniform(0, 1, len(self._param_names))
            else:
                sample = np.random.multivariate_normal(
                    from_norm, self._covariance * 0.01,
                )
            sample = np.clip(sample, 0.0, 1.0)
            results.append(self._denormalize(sample))
        return results

    # ── queries ──

    def project(self, theta: dict[str, float]) -> tuple[float, ...]:
        normed = self._normalize(theta)
        centered = normed - self._mean
        coords = self._eigenvectors[:, :self._effective_dim].T @ centered
        return tuple(float(c) for c in coords)

    def effective_dimensionality(self) -> int:
        return self._effective_dim

    def parameter_importance(self) -> dict[str, float]:
        result: dict[str, float] = {}
        total = float(self._eigenvalues.sum())
        if total == 0:
            return {name: 1.0 / len(self._param_names) for name in self._param_names}
        for i, name in enumerate(self._param_names):
            importance = sum(
                float(self._eigenvalues[j]) * abs(float(self._eigenvectors[i, j]))
                for j in range(self._effective_dim)
            )
            result[name] = importance / total
        return result

    def principal_components(self) -> list[dict[str, float]]:
        pcs: list[dict[str, float]] = []
        for j in range(self._effective_dim):
            pc: dict[str, float] = {}
            for i, name in enumerate(self._param_names):
                pc[name] = float(self._eigenvectors[i, j])
            pcs.append(pc)
        return pcs

    # ── state ──

    def state(self) -> ManifoldState:
        return ManifoldState(
            param_names=self._param_names,
            covariance=tuple(tuple(float(v) for v in row) for row in self._covariance),
            principal_components=tuple(
                tuple(float(v) for v in self._eigenvectors[:, j])
                for j in range(self._effective_dim)
            ),
            eigenvalues=tuple(float(v) for v in self._eigenvalues),
            effective_dim=self._effective_dim,
            mean=tuple(float(v) for v in self._mean),
            step_sizes=tuple(float(v) for v in self._step_sizes),
            generation=self._generation,
        )

    def save(self, path: Path | None = None) -> None:
        if path is None:
            msg = "path required"
            raise ValueError(msg)
        state = self.state()
        data = {
            "param_names": list(state.param_names),
            "param_bounds": {k: list(v) for k, v in self._param_bounds.items()},
            "covariance": [list(row) for row in state.covariance],
            "principal_components": [list(pc) for pc in state.principal_components],
            "eigenvalues": list(state.eigenvalues),
            "effective_dim": state.effective_dim,
            "mean": list(state.mean),
            "step_sizes": list(state.step_sizes),
            "generation": state.generation,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp, str(path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def load(path: Path) -> ManifoldNavigator:
        with open(path) as f:
            data = json.load(f)
        nav = ManifoldNavigator(
            param_names=data["param_names"],
            param_bounds={k: tuple(v) for k, v in data["param_bounds"].items()},
        )
        nav._covariance = np.array(data["covariance"])
        nav._eigenvalues = np.array(data["eigenvalues"])
        d = len(data["param_names"])
        # Reconstruct full eigenvector matrix
        pcs = data["principal_components"]
        nav._eigenvectors = np.eye(d)
        for j, pc in enumerate(pcs):
            nav._eigenvectors[:, j] = pc
        nav._mean = np.array(data["mean"])
        nav._step_sizes = np.array(data["step_sizes"])
        nav._effective_dim = data["effective_dim"]
        nav._generation = data["generation"]
        return nav
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifold.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/manifold.py tests/test_manifold.py
git commit -m "feat(manifold): add ManifoldNavigator with CMA-ES covariance learning"
```

---

### Task 8: Implement ForgettingDetector

**Files:**
- Create: `src/autotune/forgetting.py`
- Create: `tests/test_forgetting.py`

- [ ] **Step 1: Write tests for cliff detection and danger zones**

```python
# tests/test_forgetting.py
import pytest
from autotune.forgetting import ForgettingDetector
from autotune.types import DAGNode, DangerZone


def _make_node(
    node_id: str, retention: float | None = None,
    parent_ids: tuple[str, ...] = (),
    move_type: str = "branch",
    techniques: tuple[str, ...] = (),
) -> DAGNode:
    return DAGNode(
        id=node_id, parent_ids=parent_ids, move_type=move_type,
        merge_strategy=None, merge_weights=None,
        theta={}, manifold_coords=(0.0,),
        strategy_desc="", technique_tags=techniques,
        task_score=0.8, retention_score=retention, cost=1.0,
        composite=0.8, checkpoint_id=None, depth=1,
        potential=0.0, pareto_rank=0, node_type="executed",
        status="keep", created_at="2026-03-27",
    )


class TestCliffDetection:
    def test_detects_cliff(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        parent = _make_node("p", retention=0.95)
        child = _make_node("c", retention=0.80, parent_ids=("p",),
                           techniques=("lr_increase",))
        report = detector.check_cliff(child, (parent,))
        assert report is not None
        assert report.delta_retention == pytest.approx(-0.15)
        assert report.cause_techniques == ("lr_increase",)

    def test_no_cliff_when_retention_stable(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        parent = _make_node("p", retention=0.95)
        child = _make_node("c", retention=0.92, parent_ids=("p",))
        report = detector.check_cliff(child, (parent,))
        assert report is None

    def test_merge_uses_max_parent_retention(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        p1 = _make_node("p1", retention=0.90)
        p2 = _make_node("p2", retention=0.95)
        child = _make_node("c", retention=0.80, parent_ids=("p1", "p2"),
                           move_type="merge", techniques=("merge_weighted",))
        report = detector.check_cliff(child, (p1, p2))
        assert report is not None
        # delta should be vs max parent retention (0.95)
        assert report.delta_retention == pytest.approx(-0.15)

    def test_no_cliff_when_no_retention(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        parent = _make_node("p", retention=None)
        child = _make_node("c", retention=None, parent_ids=("p",))
        assert detector.check_cliff(child, (parent,)) is None


class TestDangerZones:
    def test_add_and_query(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        detector.add_danger_zone(
            center=(0.5, 0.3), radius=0.15,
            cause="lr_increase", severity=0.15,
        )
        zones = detector.danger_zones()
        assert len(zones) == 1
        assert zones[0].cause_technique == "lr_increase"

    def test_is_in_danger_zone(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        detector.add_danger_zone(
            center=(0.5, 0.3), radius=0.15,
            cause="lr_increase", severity=0.15,
        )
        assert detector.is_in_danger_zone_coords((0.5, 0.35))
        assert not detector.is_in_danger_zone_coords((0.9, 0.9))


class TestForgettingPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        detector.add_danger_zone(
            center=(0.5,), radius=0.2, cause="dropout", severity=0.12,
        )
        path = tmp_path / "forgetting.json"
        detector.save(path)
        loaded = ForgettingDetector.load(path)
        assert len(loaded.danger_zones()) == 1
        assert loaded.danger_zones()[0].cause_technique == "dropout"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forgetting.py -v`
Expected: FAIL

- [ ] **Step 3: Implement forgetting.py**

```python
# src/autotune/forgetting.py
"""Forgetting detector: retention cliff detection and danger zones."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from autotune.types import CliffReport, DAGNode, DangerZone


class ForgettingDetector:
    def __init__(self, cliff_threshold: float = -0.10) -> None:
        self._cliff_threshold = cliff_threshold
        self._danger_zones: list[DangerZone] = []

    def check_cliff(
        self,
        node: DAGNode,
        parents: tuple[DAGNode, ...],
    ) -> CliffReport | None:
        if node.retention_score is None:
            return None
        parent_retentions = [
            p.retention_score for p in parents if p.retention_score is not None
        ]
        if not parent_retentions:
            return None
        best_parent_retention = max(parent_retentions)
        delta = node.retention_score - best_parent_retention
        if delta >= self._cliff_threshold:
            return None
        severity = abs(delta)
        coords = node.manifold_coords or ()
        dz = DangerZone(
            center=coords,
            radius=severity * 0.5,
            cause_technique=(
                node.technique_tags[0] if node.technique_tags else "unknown"
            ),
            severity=severity,
        )
        self._danger_zones.append(dz)
        return CliffReport(
            node_id=node.id,
            parent_ids=node.parent_ids,
            delta_retention=delta,
            cause_techniques=node.technique_tags,
            severity=severity,
            danger_zone=dz,
        )

    def is_in_danger_zone_coords(self, coords: tuple[float, ...]) -> bool:
        for dz in self._danger_zones:
            if not dz.center:
                continue
            min_len = min(len(coords), len(dz.center))
            dist = math.sqrt(sum(
                (coords[i] - dz.center[i]) ** 2 for i in range(min_len)
            ))
            if dist <= dz.radius:
                return True
        return False

    def add_danger_zone(
        self,
        center: tuple[float, ...],
        radius: float,
        cause: str,
        severity: float,
    ) -> None:
        self._danger_zones.append(DangerZone(
            center=center, radius=radius,
            cause_technique=cause, severity=severity,
        ))

    def danger_zones(self) -> tuple[DangerZone, ...]:
        return tuple(self._danger_zones)

    def worst_technique_for_retention(self) -> str | None:
        if not self._danger_zones:
            return None
        from collections import Counter
        counts = Counter(dz.cause_technique for dz in self._danger_zones)
        return counts.most_common(1)[0][0]

    def save(self, path: Path) -> None:
        data = {
            "cliff_threshold": self._cliff_threshold,
            "danger_zones": [
                {
                    "center": list(dz.center),
                    "radius": dz.radius,
                    "cause_technique": dz.cause_technique,
                    "severity": dz.severity,
                }
                for dz in self._danger_zones
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp, str(path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def load(path: Path) -> ForgettingDetector:
        with open(path) as f:
            data = json.load(f)
        detector = ForgettingDetector(cliff_threshold=data["cliff_threshold"])
        for dz_data in data.get("danger_zones", []):
            detector._danger_zones.append(DangerZone(
                center=tuple(dz_data["center"]),
                radius=dz_data["radius"],
                cause_technique=dz_data["cause_technique"],
                severity=dz_data["severity"],
            ))
        return detector
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_forgetting.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/forgetting.py tests/test_forgetting.py
git commit -m "feat(forgetting): add ForgettingDetector with cliff detection and danger zones"
```

---

### Task 9: Implement ExperimentRunner

**Files:**
- Create: `src/autotune/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write tests for worktree and checkpoint operations**

```python
# tests/test_runner.py
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
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True, check=True,
    )
    (repo / "train.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, capture_output=True, check=True,
    )
    return repo


class TestWorktreeLifecycle:
    def test_create_and_cleanup(self, project_repo):
        runner = ExperimentRunner(project_repo)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
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
            cwd=project_repo, capture_output=True, text=True, check=True,
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
            cwd=project_repo, capture_output=True, text=True, check=True,
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
            cwd=project_repo, capture_output=True, text=True, check=True,
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
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        wt = runner.create_worktree(commit, "node_0001")
        # Simulate training script saving a checkpoint
        ckpt_dir = wt / "checkpoint_output"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.safetensors").write_text("fake weights")
        store = project_repo / "checkpoints"
        store.mkdir()
        ckpt_id = runner.collect_checkpoint(ckpt_dir, store, "node_0001")
        assert ckpt_id == "node_0001"
        assert (store / "node_0001" / "model.safetensors").exists()
        runner.cleanup_worktree(wt)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement runner.py**

```python
# src/autotune/runner.py
"""Experiment runner: git worktree isolation and checkpoint operations."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class ExperimentRunner:
    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._wt_base = project_root / ".autotune_worktrees"
        self._wt_base.mkdir(exist_ok=True)

    def worktree_path(self, node_id: str) -> Path:
        return self._wt_base / node_id

    def create_worktree(self, base_commit: str, node_id: str) -> Path:
        wt_path = self.worktree_path(node_id)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_path), base_commit],
            cwd=self._root,
            capture_output=True,
            check=True,
        )
        return wt_path

    def apply_overlay(
        self,
        wt_path: Path,
        overlay_dir: Path,
        deletions: list[str],
    ) -> None:
        skip = {"spec.json", "run.log", "result.json"}
        if overlay_dir.exists():
            for src_file in overlay_dir.rglob("*"):
                if src_file.is_dir():
                    continue
                rel = src_file.relative_to(overlay_dir)
                if rel.name in skip:
                    continue
                dest = wt_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest)
        for deletion in deletions:
            target = wt_path / deletion
            if target.exists():
                target.unlink()

    def collect_result(self, wt_path: Path, archive_dir: Path) -> dict | None:
        result_path = wt_path / "result.json"
        if not result_path.exists():
            return None
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / "result.json"
        shutil.copy2(result_path, dest)
        with open(dest) as f:
            return json.load(f)

    def collect_checkpoint(
        self,
        checkpoint_src: Path,
        checkpoint_store: Path,
        node_id: str,
    ) -> str | None:
        if not checkpoint_src.exists():
            return None
        dest = checkpoint_store / node_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(checkpoint_src, dest)
        return node_id

    def prepare_branch(self, checkpoint_id: str, checkpoint_store: Path, wt_path: Path) -> None:
        src = checkpoint_store / checkpoint_id
        if not src.exists():
            return
        dest = wt_path / ".parent_checkpoint"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)

    def cleanup_worktree(self, wt_path: Path) -> None:
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_path)],
                cwd=self._root,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            if wt_path.exists():
                shutil.rmtree(wt_path)
            self.prune_stale_worktrees()

    def prune_stale_worktrees(self) -> None:
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self._root,
            capture_output=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/runner.py tests/test_runner.py
git commit -m "feat(runner): add ExperimentRunner with worktree and checkpoint ops"
```

---

## Phase 4: Composition Modules

### Task 10: Implement StrategyNavigator

**Files:**
- Create: `src/autotune/navigator.py`
- Create: `tests/test_navigator.py`

- [ ] **Step 1: Write tests for move suggestions**

```python
# tests/test_navigator.py
import pytest
from autotune.dag import ExperimentDAG
from autotune.forgetting import ForgettingDetector
from autotune.manifold import ManifoldNavigator
from autotune.navigator import StrategyNavigator
from autotune.types import ScoringConfig


def _setup(tmp_path):
    scoring = ScoringConfig(0.005, 0.01, -0.10)
    dag = ExperimentDAG.create(tmp_path / "dag.json", scoring, objective_count=3)
    manifold = ManifoldNavigator(
        param_names=("lr", "rank"),
        param_bounds={"lr": (1e-5, 1e-3), "rank": (4, 64)},
    )
    forgetting = ForgettingDetector(cliff_threshold=-0.10)
    return dag, manifold, forgetting


class TestSuggestMoves:
    def test_suggests_branch_from_kept_nodes(self, tmp_path):
        dag, manifold, forgetting = _setup(tmp_path)
        dag.add_seed(theta={"lr": 2e-4, "rank": 16}, description="base",
                     techniques=(), composite=0.72, task_score=0.72,
                     retention_score=0.95, cost=1.0)
        dag.recalculate_potential()
        nav = StrategyNavigator(dag, manifold, forgetting)
        moves = nav.suggest_moves(n=3)
        assert len(moves) >= 1
        assert moves[0].move_type == "branch"
        assert moves[0].parent_ids == ("node_0001",)

    def test_suggests_merge_when_two_kept_nodes(self, tmp_path):
        dag, manifold, forgetting = _setup(tmp_path)
        dag.add_seed(theta={"lr": 2e-4, "rank": 16}, description="A",
                     techniques=(), composite=0.8, task_score=0.85,
                     retention_score=0.7, cost=1.0)
        dag.add_seed(theta={"lr": 5e-4, "rank": 32}, description="B",
                     techniques=(), composite=0.75, task_score=0.7,
                     retention_score=0.95, cost=1.5)
        dag.recalculate_potential()
        nav = StrategyNavigator(dag, manifold, forgetting)
        moves = nav.suggest_moves(n=5)
        move_types = [m.move_type for m in moves]
        assert "merge" in move_types


class TestExplainMethods:
    def test_explain_manifold_returns_string(self, tmp_path):
        dag, manifold, forgetting = _setup(tmp_path)
        nav = StrategyNavigator(dag, manifold, forgetting)
        result = nav.explain_manifold()
        assert isinstance(result, str)

    def test_explain_frontier_returns_string(self, tmp_path):
        dag, manifold, forgetting = _setup(tmp_path)
        dag.add_seed(theta={"lr": 2e-4, "rank": 16}, description="base",
                     techniques=(), composite=0.72, task_score=0.72,
                     retention_score=0.95, cost=1.0)
        dag.rerank_pareto()
        nav = StrategyNavigator(dag, manifold, forgetting)
        result = nav.explain_frontier()
        assert "node_0001" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_navigator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement navigator.py**

```python
# src/autotune/navigator.py
"""Strategy navigator: combines DAG scores, manifold, and forgetting."""

from __future__ import annotations

import math

from autotune.dag import ExperimentDAG
from autotune.forgetting import ForgettingDetector
from autotune.manifold import ManifoldNavigator
from autotune.types import DAGNode, MoveProposal


class StrategyNavigator:
    def __init__(
        self,
        dag: ExperimentDAG,
        manifold: ManifoldNavigator,
        forgetting: ForgettingDetector,
    ) -> None:
        self._dag = dag
        self._manifold = manifold
        self._forgetting = forgetting

    def suggest_moves(self, n: int = 5, budget_hours: float | None = None) -> list[MoveProposal]:
        proposals: list[MoveProposal] = []
        kept = self._dag.kept_nodes()
        if not kept:
            return proposals
        meta = self._dag.meta()

        # Branch proposals from each kept node
        for node in kept:
            theta = self._manifold.propose_direction(node.theta)
            coords = self._manifold.project(theta)
            if self._forgetting.is_in_danger_zone_coords(coords):
                continue
            score = self._score_branch(node, meta)
            proposals.append(MoveProposal(
                move_type="branch",
                parent_ids=(node.id,),
                suggested_theta=theta,
                merge_strategy=None,
                merge_weights=None,
                direction_label=self._direction_label(node.theta, theta),
                score=score,
                reasoning=self._branch_reasoning(node),
            ))

        # Merge proposals from pairs of kept nodes
        for i, a in enumerate(kept):
            for b in kept[i + 1:]:
                div = self._diversity(a, b)
                if div < 0.1:
                    continue
                score = self._score_merge(a, b, div, meta)
                merged_theta = {
                    k: 0.5 * a.theta.get(k, 0) + 0.5 * b.theta.get(k, 0)
                    for k in set(a.theta) | set(b.theta)
                }
                proposals.append(MoveProposal(
                    move_type="merge",
                    parent_ids=(a.id, b.id),
                    suggested_theta=merged_theta,
                    merge_strategy="weighted_avg",
                    merge_weights=(0.5, 0.5),
                    direction_label=f"merge {a.id}+{b.id}",
                    score=score,
                    reasoning=self._merge_reasoning(a, b, div),
                ))

        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals[:n]

    def score_move(self, move: MoveProposal) -> float:
        return move.score

    def explain_manifold(self) -> str:
        state = self._manifold.state()
        if state.generation == 0:
            return "Manifold not yet learned (need 10+ evaluated points)."
        pcs = self._manifold.principal_components()
        importance = self._manifold.parameter_importance()
        lines = [
            f"Manifold: {state.effective_dim} effective dimensions "
            f"(generation {state.generation})",
            "",
            "Parameter importance:",
        ]
        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {imp:.1%}")
        lines.append("")
        for i, pc in enumerate(pcs[:3]):
            top = sorted(pc.items(), key=lambda x: -abs(x[1]))[:3]
            desc = ", ".join(f"{n}{'↑' if v > 0 else '↓'}" for n, v in top)
            lines.append(f"  PC{i + 1}: {desc}")
        return "\n".join(lines)

    def explain_frontier(self) -> str:
        front = self._dag.kept_nodes()
        if not front:
            return "No Pareto front yet."
        lines = ["Pareto front (rank 0-1 nodes):", ""]
        for node in sorted(front, key=lambda n: -(n.composite or 0)):
            ret = f", ret={node.retention_score:.3f}" if node.retention_score else ""
            lines.append(
                f"  {node.id}: task={node.task_score:.3f}{ret}, "
                f"cost={node.cost:.2f}, rank={node.pareto_rank}"
            )
        return "\n".join(lines)

    def recommend_strategy(self) -> str:
        moves = self.suggest_moves(n=1)
        if not moves:
            return "No moves available. Add seed experiments first."
        m = moves[0]
        return f"Recommend: {m.move_type} {m.direction_label} (score={m.score:.3f}). {m.reasoning}"

    # ── scoring helpers ──

    def _score_branch(self, parent: DAGNode, meta: DAGMeta) -> float:
        composite = parent.composite or 0.0
        w_e = meta.scoring.exploration_weight
        total = max(meta.total_executed, 1)
        child_count = len(self._dag.children(parent.id))
        exploration = w_e * math.sqrt(math.log(total) / (1 + child_count))
        health = self._retention_health(parent, meta.scoring.retention_cliff_threshold)
        return (composite + exploration) * health

    def _score_merge(
        self, a: DAGNode, b: DAGNode, diversity: float, meta: DAGMeta,
    ) -> float:
        avg_composite = 0.5 * ((a.composite or 0) + (b.composite or 0))
        w_e = meta.scoring.exploration_weight
        w_div = meta.scoring.diversity_weight
        total = max(meta.total_executed, 1)
        exploration = w_e * math.sqrt(math.log(total) / 1)
        complementarity = 1.0
        if a.task_score and b.task_score and a.retention_score and b.retention_score:
            complementarity = 1.0 + (
                abs(a.task_score - b.task_score)
                * abs(a.retention_score - b.retention_score)
            )
        return (avg_composite + exploration + w_div * diversity) * complementarity

    @staticmethod
    def _retention_health(node: DAGNode, cliff_threshold: float) -> float:
        if node.retention_score is None:
            return 1.0
        if node.retention_score >= abs(cliff_threshold):
            return node.retention_score
        return node.retention_score * 0.5

    def _diversity(self, a: DAGNode, b: DAGNode) -> float:
        coords_a = self._manifold.project(a.theta)
        coords_b = self._manifold.project(b.theta)
        if not coords_a or not coords_b:
            return 0.5
        dot = sum(x * y for x, y in zip(coords_a, coords_b))
        mag_a = math.sqrt(sum(x ** 2 for x in coords_a)) or 1e-10
        mag_b = math.sqrt(sum(x ** 2 for x in coords_b)) or 1e-10
        cosine = dot / (mag_a * mag_b)
        return 1.0 - max(-1.0, min(1.0, cosine))

    def _direction_label(self, from_theta: dict, to_theta: dict) -> str:
        changes: list[str] = []
        for k in from_theta:
            if k in to_theta:
                delta = to_theta[k] - from_theta[k]
                if abs(delta) > 1e-10:
                    direction = "↑" if delta > 0 else "↓"
                    changes.append(f"{k}{direction}")
        return ", ".join(changes[:3]) if changes else "small step"

    def _branch_reasoning(self, parent: DAGNode) -> str:
        ret_str = ""
        if parent.retention_score is not None:
            ret_str = f", retention budget={parent.retention_score:.2f}"
        return f"parent composite={parent.composite:.3f}{ret_str}"

    def _merge_reasoning(self, a: DAGNode, b: DAGNode, div: float) -> str:
        return (
            f"diversity={div:.2f}, "
            f"{a.id} task={a.task_score:.3f} ret={a.retention_score}, "
            f"{b.id} task={b.task_score:.3f} ret={b.retention_score}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_navigator.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/navigator.py tests/test_navigator.py
git commit -m "feat(navigator): add StrategyNavigator with branch/merge suggestions"
```

---

## Phase 5: CLI & Templates

### Task 11: Implement CLI — init, status, check

**Files:**
- Create: `src/autotune/cli.py`
- Create: `src/autotune/templates/config.yaml.j2`
- Create: `src/autotune/templates/program.md.j2`
- Create: `src/autotune/templates/learnings.md.j2`
- Create: `src/autotune/templates/.gitignore.j2`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write tests for init and status**

```python
# tests/test_cli.py
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from autotune.cli import main


@pytest.fixture
def project_dir(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True, check=True)
    (repo / "train.py").write_text("print('train')")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True)
    return repo


class TestInit:
    def test_creates_overlay(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Create Jinja2 templates**

```yaml
# src/autotune/templates/config.yaml.j2
project:
  name: "{{ project_name }}"
  base_model: "{{ base_model | default('meta-llama/Llama-3-8B') }}"

task:
  type: "{{ task_type | default('instruction_tuning') }}"

run:
  script: "train.py"
  checkpoint_format: "safetensors"

retention:
  enabled: false
  script: null
  probe_set: null

parameters:
  lr:
    type: float
    range: [1.0e-5, 1.0e-3]
    log_scale: true
  rank:
    type: int
    range: [4, 64]
  alpha:
    type: int
    range: [8, 128]
  dropout:
    type: float
    range: [0.0, 0.3]
  warmup_ratio:
    type: float
    range: [0.0, 0.15]

merge:
  default_strategy: "weighted_avg"
  custom_hook: null

orchestrator:
  poll_interval: 30
  max_per_gpu: 2
  safety_margin_gb: 2.0
  default_timeout_hours: 8
  checkpoint_retention_layers: 2

scoring:
  exploration_weight: 0.005
  diversity_weight: 0.01
  retention_cliff_threshold: -0.10

files:
  editable:
    - "train.py"
    - "configs/"
  readonly: []
```

```markdown
# src/autotune/templates/program.md.j2
# {{ project_name }} — autoTune Program

## Objective
Iteratively improve the finetuned model by exploring hyperparameter configurations,
checkpoint branches, and merges. Use the manifold navigator for numerical guidance
and your own judgment for strategic decisions.

## Loop
1. `autotune reconcile` — sync DAG with completed experiments
2. `autotune explain` — read manifold summary and Pareto frontier
3. `autotune suggest --n 5` — get move suggestions
4. Decide: accept, modify, or override a suggestion
5. Implement changes to training script / config
6. `autotune propose` then `autotune submit`
7. Update `learnings.md` with insights
8. Wait for orchestrator, repeat
```

```markdown
# src/autotune/templates/learnings.md.j2
# {{ project_name }} — Learnings

Record insights from experiments here. The agent updates this file
after each experiment cycle.
```

```
# src/autotune/templates/.gitignore.j2
dag.json
manifold_state.json
results.tsv
orchestrator/queue/
orchestrator/running/
orchestrator/completed/
orchestrator/archive/
orchestrator/checkpoints/
orchestrator/*.pid
orchestrator/*.log
orchestrator/gpu_state.json
```

- [ ] **Step 4: Implement cli.py with init and status commands**

```python
# src/autotune/cli.py
"""autoTune CLI entry point."""

from __future__ import annotations

import fnmatch
from pathlib import Path

import click
import yaml
from jinja2 import Environment, FileSystemLoader


def _find_project_root() -> Path | None:
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "autotune" / "config.yaml").exists():
            return parent
    return None


@click.group()
def main() -> None:
    """autoTune: Autonomous finetuning framework."""


@main.command()
@click.option("--task", "task_type", default="instruction_tuning")
@click.option("--base-model", default="meta-llama/Llama-3-8B")
def init(task_type: str, base_model: str) -> None:
    """Initialize autoTune overlay in the current repo."""
    cwd = Path.cwd()
    if not (cwd / ".git").exists():
        raise click.ClickException("Not a git repository. Run 'git init' first.")
    if (cwd / "autotune" / "config.yaml").exists():
        raise click.ClickException("autoTune already initialized.")

    # Create directories
    dirs = [
        "autotune",
        "autotune/orchestrator/queue",
        "autotune/orchestrator/running",
        "autotune/orchestrator/completed",
        "autotune/orchestrator/archive",
        "autotune/orchestrator/checkpoints",
    ]
    for d in dirs:
        (cwd / d).mkdir(parents=True, exist_ok=True)

    # Render templates
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {
        "project_name": cwd.name,
        "task_type": task_type,
        "base_model": base_model,
    }
    templates = {
        "config.yaml.j2": "autotune/config.yaml",
        "program.md.j2": "autotune/program.md",
        "learnings.md.j2": "autotune/learnings.md",
        ".gitignore.j2": "autotune/.gitignore",
    }
    for tmpl_name, out_path in templates.items():
        template = env.get_template(tmpl_name)
        rendered = template.render(context)
        (cwd / out_path).write_text(rendered)

    click.echo("autoTune initialized.")


@main.command()
def status() -> None:
    """Show experiment status."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    orch = root / "autotune" / "orchestrator"
    queue_count = len(list((orch / "queue").glob("*.json"))) if (orch / "queue").exists() else 0
    completed_count = len(list((orch / "completed").glob("*.json"))) if (orch / "completed").exists() else 0
    dag_path = root / "autotune" / "dag.json"

    click.echo(f"Queue: {queue_count} pending")
    click.echo(f"Completed: {completed_count} awaiting reconcile")

    if dag_path.exists():
        from autotune.dag import ExperimentDAG
        dag = ExperimentDAG.load(dag_path)
        meta = dag.meta()
        click.echo(f"Executed: {meta.total_executed}")
        click.echo(f"Proposed: {meta.total_proposed}")
        if meta.best_composite is not None:
            click.echo(f"Best composite: {meta.best_composite:.4f} ({meta.best_node_id})")
    else:
        click.echo("No experiments yet.")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/autotune/cli.py src/autotune/templates/ tests/test_cli.py
git commit -m "feat(cli): add init and status commands with Jinja2 templates"
```

---

### Task 12: Implement CLI — suggest, explain, frontier, reconcile

**Files:**
- Modify: `src/autotune/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write tests for suggest and explain**

```python
# append to tests/test_cli.py

class TestSuggest:
    def test_suggest_after_seed(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        # Manually create a DAG with a seed node
        from autotune.dag import ExperimentDAG
        from autotune.types import ScoringConfig
        dag_path = project_dir / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=3)
        dag.add_seed(theta={"lr": 2e-4, "rank": 16}, description="base",
                     techniques=(), composite=0.72, task_score=0.72,
                     retention_score=0.95, cost=1.0)
        dag.recalculate_potential()
        dag.save()
        result = runner.invoke(main, ["suggest", "--n", "3"])
        assert result.exit_code == 0
        assert "BRANCH" in result.output or "branch" in result.output.lower()


class TestExplain:
    def test_explain_no_manifold(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["explain"])
        assert result.exit_code == 0
        assert "not yet learned" in result.output.lower() or "no experiments" in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::TestSuggest tests/test_cli.py::TestExplain -v`
Expected: FAIL

- [ ] **Step 3: Add suggest, explain, frontier, reconcile commands to cli.py**

Append to `src/autotune/cli.py`:

```python
@main.command()
@click.option("--n", default=5, help="Number of suggestions")
def suggest(n: int) -> None:
    """Get move suggestions from the navigator."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        raise click.ClickException("No DAG found. Submit seed experiments first.")

    from autotune.dag import ExperimentDAG
    from autotune.forgetting import ForgettingDetector
    from autotune.manifold import ManifoldNavigator
    from autotune.navigator import StrategyNavigator

    dag = ExperimentDAG.load(dag_path)
    config = _load_config(root)
    params = config.get("parameters", {})
    manifold = _load_or_create_manifold(root, params)
    forgetting = _load_or_create_forgetting(root, config)
    nav = StrategyNavigator(dag, manifold, forgetting)
    moves = nav.suggest_moves(n=n)
    if not moves:
        click.echo("No moves available.")
        return
    click.echo(f"Move suggestions (top {len(moves)}):\n")
    for i, m in enumerate(moves, 1):
        parents = ", ".join(m.parent_ids)
        click.echo(f"{i}. {m.move_type.upper()} from {parents} (score: {m.score:.3f})")
        click.echo(f"   Direction: {m.direction_label}")
        click.echo(f"   Reasoning: {m.reasoning}")
        click.echo()


@main.command()
def explain() -> None:
    """Explain manifold, frontier, and recommended strategy."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    dag_path = root / "autotune" / "dag.json"

    from autotune.forgetting import ForgettingDetector
    from autotune.manifold import ManifoldNavigator
    from autotune.navigator import StrategyNavigator

    config = _load_config(root)
    params = config.get("parameters", {})
    manifold = _load_or_create_manifold(root, params)
    forgetting = _load_or_create_forgetting(root, config)

    if not dag_path.exists():
        click.echo("No experiments yet.")
        click.echo(manifold_text := f"\n{StrategyNavigator.__name__} — " if False else "")
        nav_temp = StrategyNavigator(None, manifold, forgetting)
        click.echo(nav_temp.explain_manifold())
        return

    from autotune.dag import ExperimentDAG
    dag = ExperimentDAG.load(dag_path)
    nav = StrategyNavigator(dag, manifold, forgetting)
    click.echo(nav.explain_manifold())
    click.echo()
    click.echo(nav.explain_frontier())
    click.echo()
    click.echo(nav.recommend_strategy())


@main.command()
def frontier() -> None:
    """Show current Pareto front."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        raise click.ClickException("No DAG found.")
    from autotune.dag import ExperimentDAG
    from autotune.forgetting import ForgettingDetector
    from autotune.manifold import ManifoldNavigator
    from autotune.navigator import StrategyNavigator
    dag = ExperimentDAG.load(dag_path)
    config = _load_config(root)
    params = config.get("parameters", {})
    manifold = _load_or_create_manifold(root, params)
    forgetting = _load_or_create_forgetting(root, config)
    nav = StrategyNavigator(dag, manifold, forgetting)
    click.echo(nav.explain_frontier())


@main.command()
def reconcile() -> None:
    """Sync DAG with orchestrator state."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        click.echo("No DAG to reconcile.")
        return
    from autotune.dag import ExperimentDAG
    dag = ExperimentDAG.load(dag_path)
    orch = root / "autotune" / "orchestrator"
    dag.reconcile(
        orch / "queue", orch / "running",
        orch / "completed", orch / "archive",
    )
    dag.save()
    meta = dag.meta()
    click.echo(f"Reconciled. Executed: {meta.total_executed}, Best: {meta.best_composite}")


# ── helpers ──

def _load_config(root: Path) -> dict:
    config_path = root / "autotune" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _load_or_create_manifold(root: Path, params: dict) -> "ManifoldNavigator":
    from autotune.manifold import ManifoldNavigator
    manifold_path = root / "autotune" / "manifold_state.json"
    if manifold_path.exists():
        return ManifoldNavigator.load(manifold_path)
    param_names = list(params.keys())
    param_bounds = {}
    for name, spec in params.items():
        r = spec.get("range", [0, 1])
        param_bounds[name] = (r[0], r[1])
    return ManifoldNavigator(param_names=param_names, param_bounds=param_bounds)


def _load_or_create_forgetting(root: Path, config: dict) -> "ForgettingDetector":
    from autotune.forgetting import ForgettingDetector
    forgetting_path = root / "autotune" / "forgetting.json"
    if forgetting_path.exists():
        return ForgettingDetector.load(forgetting_path)
    threshold = config.get("scoring", {}).get("retention_cliff_threshold", -0.10)
    return ForgettingDetector(cliff_threshold=threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/autotune/cli.py tests/test_cli.py
git commit -m "feat(cli): add suggest, explain, frontier, reconcile commands"
```

---

## Phase 6: Integration Tests

### Task 13: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test for full init → seed → suggest flow**

```python
# tests/test_integration.py
import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from autotune.cli import main
from autotune.dag import ExperimentDAG
from autotune.types import ScoringConfig


@pytest.fixture
def full_project(tmp_path):
    repo = tmp_path / "project"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True, check=True)
    (repo / "train.py").write_text("print('train')")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True)
    return repo


class TestEndToEnd:
    def test_init_seed_suggest_flow(self, full_project, monkeypatch):
        monkeypatch.chdir(full_project)
        runner = CliRunner()

        # Init
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0

        # Create DAG with seed
        dag_path = full_project / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=3)
        dag.add_seed(
            theta={"lr": 2e-4, "rank": 16, "alpha": 32, "dropout": 0.1, "warmup_ratio": 0.05},
            description="baseline LoRA",
            techniques=("baseline",),
            composite=0.72,
            task_score=0.72,
            retention_score=0.95,
            cost=1.5,
        )
        dag.recalculate_potential()
        dag.rerank_pareto()
        dag.save()

        # Status
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "0.7200" in result.output

        # Suggest
        result = runner.invoke(main, ["suggest", "--n", "3"])
        assert result.exit_code == 0

        # Explain
        result = runner.invoke(main, ["explain"])
        assert result.exit_code == 0

        # Frontier
        result = runner.invoke(main, ["frontier"])
        assert result.exit_code == 0
        assert "node_0001" in result.output

    def test_reconcile_promotes_completed(self, full_project, monkeypatch):
        monkeypatch.chdir(full_project)
        runner = CliRunner()
        runner.invoke(main, ["init"])

        dag_path = full_project / "autotune" / "dag.json"
        scoring = ScoringConfig(0.005, 0.01, -0.10)
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=3)
        dag.add_seed(theta={"lr": 2e-4}, description="base", techniques=(),
                     composite=0.7, task_score=0.7, retention_score=0.9, cost=1.0)
        dag.add_proposed(parent_ids=("node_0001",), move_type="branch",
                         description="test", techniques=("lr_increase",),
                         theta={"lr": 5e-4})
        dag.mark_running("node_0002")
        dag.save()

        # Simulate completion
        completed_dir = full_project / "autotune" / "orchestrator" / "completed"
        completion = {
            "node_id": "node_0002",
            "status": "completed",
            "metrics": {
                "task_score": 0.85,
                "retention_score": 0.88,
                "cost": 1.2,
                "composite": 0.85,
            },
        }
        (completed_dir / "node_0002.json").write_text(json.dumps(completion))

        result = runner.invoke(main, ["reconcile"])
        assert result.exit_code == 0
        assert "0.85" in result.output

        dag = ExperimentDAG.load(dag_path)
        node = dag.get_node("node_0002")
        assert node.node_type == "executed"
        assert node.composite == 0.85
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_integration.py -v`
Expected: all PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for init-seed-suggest-reconcile flow"
```

---

## Phase 7: Orchestrator & Visualization (deferred)

### Task 14: Implement ExperimentOrchestrator

> **Note:** The orchestrator follows the same pattern as autoMIL's `orchestrator.py` but adds checkpoint management. It should be implemented after the core modules are stable. Key additions:
>
> **Files:** `src/autotune/orchestrator.py`, `tests/test_orchestrator.py`
>
> - GPU scheduling with best-fit bin packing (port from autoMIL)
> - `_launch()`: set `AUTOTUNE_CHECKPOINT_DIR`, `AUTOTUNE_PARENT_CHECKPOINT`, call `runner.merge_checkpoints()` for merge moves
> - `_handle_completion()`: call `runner.collect_checkpoint()`, trigger `manifold.update()` and `forgetting.check_cliff()`
> - `_prune_checkpoints()`: delete checkpoints for `pareto_rank >= 2`
> - `_recover_orphans()`: only in `run()`, never on `__init__`
> - Daemon loop with PID file

### Task 15: Implement Visualization

> **Note:** The viz system follows the same pattern as autoMIL's `viz/server.py` + `viz/static/app.js`. Key additions:
>
> **Files:** `src/autotune/viz/server.py`, `src/autotune/viz/static/index.html`, `src/autotune/viz/static/app.js`
>
> - Watch `dag.json` AND `manifold_state.json`
> - 5 tabs: DAG View, Pareto Frontier, Manifold Explorer, Leaderboard, Technique Effectiveness
> - Diamond-shaped merge nodes in ForceGraph3D
> - Pareto front scatter plot (2D/3D)
> - Manifold PC projection with danger zones

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|------------------|
| 1 | 1 | Package scaffold, installable |
| 2 | 2 | Frozen data types |
| 3 | 3-9 | All leaf modules: dag, manifold, forgetting, runner |
| 4 | 10 | Navigator (composition layer) |
| 5 | 11-12 | CLI with all agent-facing commands |
| 6 | 13 | End-to-end integration tests |
| 7 | 14-15 | Orchestrator + visualization (deferred, follows autoMIL patterns) |

After Phase 6 completes, the framework is functionally usable: agents can `init`, create seed experiments, get `suggest`ions, `propose` and `submit` experiments, `reconcile` completions, and view the `frontier`. The orchestrator and viz are deferred because they closely follow autoMIL's proven patterns and can be ported with minimal design risk.
