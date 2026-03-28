"""Experiment DAG: node CRUD, Pareto ranking, UCB scoring, reconciliation.

The ExperimentDAG stores experiment nodes that form a directed acyclic graph.
Nodes represent training runs; merge operations combine multiple parents,
making this a true DAG (not a tree). The DAG supports NSGA-II Pareto ranking,
DAG-UCB exploration scoring, and filesystem-based reconciliation with the
experiment orchestrator.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from autotune.types import DAGMeta, DAGNode, ScoringConfig, TechniqueStats

logger = logging.getLogger(__name__)


class ExperimentDAG:
    """Core DAG for experiment tracking, ranking, and scoring.

    The DAG is stored as a JSON dict internally and persisted via atomic
    writes. All public methods return frozen DAGNode dataclasses; internal
    state is managed through plain dicts for efficient mutation during
    batch operations (rerank, recalculate).

    Args:
        path: Filesystem path for JSON persistence.
        data: Internal state dict (schema_version, meta, nodes, technique_stats).
    """

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self._path = path
        self._data = data

    # ── factory methods ──

    @staticmethod
    def create(
        path: Path,
        scoring: ScoringConfig,
        objective_count: int = 3,
    ) -> ExperimentDAG:
        """Create a new empty ExperimentDAG.

        Args:
            path: Filesystem path where the DAG will be persisted.
            scoring: UCB scoring hyperparameters.
            objective_count: Number of Pareto objectives (2 or 3).

        Returns:
            A fresh ExperimentDAG with no nodes.
        """
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
        """Load an ExperimentDAG from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            The deserialized ExperimentDAG.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded DAG from %s (%d nodes)", path, len(data.get("nodes", {})))
        return ExperimentDAG(path, data)

    def save(self) -> None:
        """Persist the DAG to disk via atomic write (write-to-temp then rename).

        Raises:
            OSError: If the write or rename fails.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.rename(tmp, str(self._path))
            logger.debug("Saved DAG to %s", self._path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ── internal helpers ──

    def _next_id(self) -> str:
        """Generate the next sequential node ID and advance the counter."""
        nid = self._data["meta"]["next_id"]
        self._data["meta"]["next_id"] = nid + 1
        return f"node_{nid:04d}"

    def _node_to_dataclass(self, d: dict[str, Any]) -> DAGNode:
        """Convert an internal node dict to a frozen DAGNode dataclass."""
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
        """Compute DAG depth as 1 + max parent depth."""
        if not parent_ids:
            return 0
        return 1 + max(
            self._data["nodes"][pid]["depth"] for pid in parent_ids if pid in self._data["nodes"]
        )

    def _timestamp(self) -> str:
        """Return current UTC time as ISO-8601 string."""
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    def _update_best(self, node_id: str, composite: float | None) -> None:
        """Update best_composite and best_node_id if this node is the new best."""
        best = self._data["meta"]["best_composite"]
        if composite is not None and (best is None or composite > best):
            self._data["meta"]["best_composite"] = composite
            self._data["meta"]["best_node_id"] = node_id

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
        """Add a seed node (no parents, already executed).

        Args:
            theta: Hyperparameter dict.
            description: Human-readable rationale.
            techniques: Technique tags applied.
            composite: Composite score.
            task_score: Primary task metric.
            retention_score: Retention metric, or None.
            cost: Training cost in GPU-hours, or None.
            checkpoint_id: Checkpoint artifact ID, or None.

        Returns:
            The created DAGNode (frozen).
        """
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
        self._update_best(node_id, composite)
        logger.info("Added seed node %s (composite=%.4f)", node_id, composite)
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
        """Add a proposed (not yet executed) node to the DAG.

        Args:
            parent_ids: Parent node IDs.
            move_type: One of "branch", "merge", "graft".
            description: Human-readable rationale.
            techniques: Technique tags to apply.
            theta: Proposed hyperparameters.
            merge_strategy: Merge algorithm name, or None.
            merge_weights: Per-parent weights, or None.

        Returns:
            The created DAGNode (frozen, status="pending").

        Raises:
            KeyError: If a parent ID is not in the DAG.
        """
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
        logger.info("Added proposed node %s (move=%s, parents=%s)", node_id, move_type, parent_ids)
        return self._node_to_dataclass(node_data)

    def mark_running(self, node_id: str) -> DAGNode:
        """Transition a pending node to running status.

        Args:
            node_id: The node to mark as running.

        Returns:
            The updated DAGNode (frozen).

        Raises:
            ValueError: If the node is not in "pending" status.
            KeyError: If the node does not exist.
        """
        node = self._data["nodes"][node_id]
        if node["status"] != "pending":
            msg = f"Cannot mark {node_id} as running: status is {node['status']}"
            raise ValueError(msg)
        node["status"] = "running"
        logger.info("Marked %s as running", node_id)
        return self._node_to_dataclass(node)

    def promote(self, node_id: str, metrics: dict[str, Any]) -> DAGNode:
        """Promote a running/pending node to executed with metrics.

        Args:
            node_id: The node to promote.
            metrics: Dict with task_score, retention_score, cost, composite, checkpoint_id.

        Returns:
            The updated DAGNode (frozen, node_type="executed", status="keep").
        """
        node = self._data["nodes"][node_id]
        node["node_type"] = "executed"
        node["task_score"] = metrics.get("task_score")
        node["retention_score"] = metrics.get("retention_score")
        node["cost"] = metrics.get("cost")
        node["composite"] = metrics.get("composite")
        node["checkpoint_id"] = metrics.get("checkpoint_id")
        node["status"] = "keep"
        self._data["meta"]["total_executed"] += 1
        self._update_best(node_id, node["composite"])
        logger.info("Promoted %s (composite=%s)", node_id, node["composite"])
        return self._node_to_dataclass(node)

    def mark_failed(self, node_id: str, status: str, error: str | None = None) -> DAGNode:
        """Mark a node as failed with a specific failure status.

        Args:
            node_id: The node that failed.
            status: Failure status (e.g. "oom", "crash", "timeout").
            error: Optional error message for logging.

        Returns:
            The updated DAGNode (frozen).
        """
        node = self._data["nodes"][node_id]
        node["node_type"] = "executed"
        node["status"] = status
        logger.warning("Marked %s as failed (status=%s, error=%s)", node_id, status, error)
        return self._node_to_dataclass(node)

    def cancel(self, node_id: str) -> DAGNode:
        """Cancel a proposed node.

        Args:
            node_id: The node to cancel.

        Returns:
            The updated DAGNode (frozen, status="cancelled").
        """
        node = self._data["nodes"][node_id]
        node["status"] = "cancelled"
        logger.info("Cancelled %s", node_id)
        return self._node_to_dataclass(node)

    # ── queries ──

    def get_node(self, node_id: str) -> DAGNode | None:
        """Get a single node by ID, or None if not found.

        Args:
            node_id: The node ID to look up.

        Returns:
            The DAGNode if found, else None.
        """
        d = self._data["nodes"].get(node_id)
        return self._node_to_dataclass(d) if d else None

    def parents(self, node_id: str) -> tuple[DAGNode, ...]:
        """Get the parent nodes of a given node.

        Args:
            node_id: The child node ID.

        Returns:
            Tuple of parent DAGNodes.
        """
        node = self._data["nodes"].get(node_id)
        if not node:
            return ()
        return tuple(
            self._node_to_dataclass(self._data["nodes"][pid])
            for pid in node["parent_ids"]
            if pid in self._data["nodes"]
        )

    def children(self, node_id: str) -> list[DAGNode]:
        """Get the child nodes of a given node.

        Args:
            node_id: The parent node ID.

        Returns:
            List of child DAGNodes.
        """
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if node_id in n["parent_ids"]
        ]

    def ancestors(self, node_id: str) -> list[DAGNode]:
        """Get all ancestors of a node (transitive parents).

        Args:
            node_id: The starting node ID.

        Returns:
            List of all ancestor DAGNodes (unordered).
        """
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
        """Get all nodes with Pareto rank 0 or 1 (not discarded).

        Returns:
            List of kept DAGNodes.
        """
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if n.get("pareto_rank") is not None and n["pareto_rank"] <= 1
        ]

    def pareto_front(self, layer: int = 0) -> list[DAGNode]:
        """Get nodes at the specified Pareto rank layer.

        Args:
            layer: Pareto rank to select (default 0 = non-dominated front).

        Returns:
            List of DAGNodes at the given Pareto rank.
        """
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if n.get("pareto_rank") == layer
        ]

    def lineage(self, node_id: str) -> list[list[DAGNode]]:
        """Get all root-to-node paths through the DAG.

        Args:
            node_id: Target node ID.

        Returns:
            List of paths, each path is a list of DAGNodes from root to target.
        """
        paths: list[list[DAGNode]] = []

        def _trace(nid: str, current_path: list[str]) -> None:
            node = self._data["nodes"].get(nid)
            if not node:
                return
            current_path = [nid, *current_path]
            parent_ids = node["parent_ids"]
            if not parent_ids:
                paths.append(
                    [self._node_to_dataclass(self._data["nodes"][n]) for n in current_path]
                )
                return
            for pid in parent_ids:
                _trace(pid, current_path)

        _trace(node_id, [])
        return paths

    def rank_proposals(self, n: int = 10, max_per_branch: int = 3) -> list[DAGNode]:
        """Rank pending proposed nodes by parent potential.

        Args:
            n: Maximum number of proposals to return.
            max_per_branch: Maximum proposals per parent node.

        Returns:
            Sorted list of proposed DAGNodes.
        """
        pending = [
            nd
            for nd in self._data["nodes"].values()
            if nd["node_type"] == "proposed" and nd["status"] == "pending"
        ]
        parent_counts: dict[str, int] = {}
        scored: list[tuple[float, dict[str, Any]]] = []
        for nd in pending:
            parent_potential = max(
                (
                    self._data["nodes"].get(pid, {}).get("potential", 0.0)
                    for pid in nd["parent_ids"]
                ),
                default=0.0,
            )
            scored.append((parent_potential, nd))
        scored.sort(key=lambda x: x[0], reverse=True)
        result: list[DAGNode] = []
        for _score, nd in scored:
            if len(result) >= n:
                break
            parent_key = ",".join(nd["parent_ids"])
            count = parent_counts.get(parent_key, 0)
            if count >= max_per_branch:
                continue
            parent_counts[parent_key] = count + 1
            result.append(self._node_to_dataclass(nd))
        return result

    def all_executed(self) -> list[DAGNode]:
        """Get all successfully executed nodes (excludes oom/crash/timeout).

        Returns:
            List of executed DAGNodes.
        """
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if n["node_type"] == "executed" and n["status"] not in ("oom", "crash", "timeout")
        ]

    def all_proposed(self, status: str | None = None) -> list[DAGNode]:
        """Get all proposed nodes, optionally filtered by status.

        Args:
            status: If provided, only return nodes with this status.

        Returns:
            List of proposed DAGNodes.
        """
        return [
            self._node_to_dataclass(n)
            for n in self._data["nodes"].values()
            if n["node_type"] == "proposed" and (status is None or n["status"] == status)
        ]

    def meta(self) -> DAGMeta:
        """Get the DAG metadata as a frozen dataclass.

        Returns:
            A DAGMeta snapshot.
        """
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
        """Get aggregate statistics for a named technique.

        Args:
            tag: The technique tag to look up.

        Returns:
            TechniqueStats for the tag, or a zero-valued default if unknown.
        """
        ts = self._data["technique_stats"].get(tag)
        if not ts:
            return TechniqueStats(
                times_tried=0,
                avg_task_delta=0.0,
                avg_retention_delta=0.0,
                best_parent_delta=0.0,
            )
        return TechniqueStats(
            times_tried=ts["times_tried"],
            avg_task_delta=ts["avg_task_delta"],
            avg_retention_delta=ts["avg_retention_delta"],
            best_parent_delta=ts["best_parent_delta"],
        )

    def update_technique_stats(
        self,
        tag: str,
        task_delta: float,
        retention_delta: float,
    ) -> None:
        """Record a new observation for a technique's effectiveness.

        Args:
            tag: The technique tag.
            task_delta: Task score improvement over parent.
            retention_delta: Retention score change over parent.
        """
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

    # ── Pareto ranking (NSGA-II non-dominated sorting) ──

    def rerank_pareto(self) -> None:
        """Re-run NSGA-II non-dominated sorting on all executed nodes.

        Assigns pareto_rank to each executed node:
        - Rank 0: Pareto-optimal (non-dominated)
        - Rank 1: Dominated only by rank 0 (kept)
        - Rank 2+: Deeply dominated (discarded)

        Nodes with status in (oom, crash, timeout) are excluded.
        """
        executed = [
            n
            for n in self._data["nodes"].values()
            if n["node_type"] == "executed" and n["status"] not in ("oom", "crash", "timeout")
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
        logger.debug("Pareto rerank complete: %d fronts", rank)

    @staticmethod
    def _dominates(a: dict[str, Any], b: dict[str, Any], obj_count: int) -> bool:
        """Check if node a dominates node b in multi-objective space.

        Domination means a is at least as good in all objectives and strictly
        better in at least one. Objectives: task_score (maximize),
        retention_score (maximize, 3-obj only), cost (minimize).

        Args:
            a: First node dict.
            b: Second node dict.
            obj_count: Number of objectives (2 or 3).

        Returns:
            True if a dominates b.
        """
        task_a = 0.0 if a.get("task_score") is None else a["task_score"]
        task_b = 0.0 if b.get("task_score") is None else b["task_score"]
        cost_a = float("inf") if a.get("cost") is None else a["cost"]
        cost_b = float("inf") if b.get("cost") is None else b["cost"]
        if obj_count == 3:
            ret_a = 0.0 if a.get("retention_score") is None else a["retention_score"]
            ret_b = 0.0 if b.get("retention_score") is None else b["retention_score"]
            at_least_as_good = task_a >= task_b and ret_a >= ret_b and cost_a <= cost_b
            strictly_better = task_a > task_b or ret_a > ret_b or cost_a < cost_b
        else:
            at_least_as_good = task_a >= task_b and cost_a <= cost_b
            strictly_better = task_a > task_b or cost_a < cost_b
        return at_least_as_good and strictly_better

    # ── UCB scoring ──

    def recalculate_potential(self) -> None:
        """Recalculate DAG-UCB potential scores for all executed nodes.

        Formula: potential = (composite + exploration_bonus) * retention_health

        The exploration bonus is UCB-style: w_e * sqrt(log(N) / (1 + child_count)).
        Retention health penalizes nodes below the cliff threshold.
        """
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
                1 for n in self._data["nodes"].values() if node["id"] in n["parent_ids"]
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

    # ── reconciliation ──

    def reconcile(
        self,
        queue_dir: Path,
        running_dir: Path,
        completed_dir: Path,
        archive_dir: Path,
    ) -> None:
        """Synchronize DAG state with orchestrator filesystem directories.

        1. Scans completed_dir for finished experiments and promotes them.
        2. Detects stale "running" nodes not tracked by orchestrator and
           resets them to "pending".
        3. Re-runs Pareto ranking and UCB scoring.

        Args:
            queue_dir: Directory containing queued experiment JSON files.
            running_dir: Directory containing running experiment JSON files.
            completed_dir: Directory containing completed experiment JSON files.
            archive_dir: Directory for archiving processed completions.
        """
        # Collect node IDs the orchestrator knows about
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
                    logger.info("Reconciled %s: promoted with metrics", node_id)
                else:
                    self.mark_failed(node_id, status)
                    logger.info("Reconciled %s: marked as %s", node_id, status)
                archive_dir.mkdir(parents=True, exist_ok=True)
                f.rename(archive_dir / f.name)

        # Reset stale running nodes not tracked by orchestrator
        for node in self._data["nodes"].values():
            if (
                node["node_type"] == "proposed"
                and node["status"] == "running"
                and node["id"] not in orch_ids
            ):
                node["status"] = "pending"
                logger.warning("Reset stale running node %s to pending", node["id"])

        self.rerank_pareto()
        self.recalculate_potential()
