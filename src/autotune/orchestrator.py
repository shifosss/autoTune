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
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    total_mb=int(parts[1]),
                    free_mb=int(parts[2]),
                    utilization=int(parts[3]),
                )
            )
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
        for d in (
            self.queue_dir,
            self.running_dir,
            self.completed_dir,
            self.archive_dir,
            self.checkpoint_store,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self._load_state(recover=False)

    # ── state persistence ──

    def _load_state(self, recover: bool = False) -> None:
        """Load persisted GPU state and optionally recover orphaned experiments."""
        if self.gpu_state_file.exists():
            try:
                json.loads(self.gpu_state_file.read_text())
            except (json.JSONDecodeError, KeyError):
                pass
        if recover:
            self._recover_orphans()

    def _save_state(self) -> None:
        """Persist current GPU allocation and queue state to gpu_state.json."""
        gpus = query_gpus()
        gpu_details = []
        for gpu in gpus:
            alloc = self.gpu_allocations.get(gpu.index, [])
            gpu_details.append(
                {
                    "index": gpu.index,
                    "total_mb": gpu.total_mb,
                    "free_mb": gpu.free_mb,
                    "schedulable_free_gb": round(
                        gpu.free_gb - self.safety_margin_gb, 2
                    ),
                    "running": alloc,
                    "utilization_pct": gpu.utilization,
                }
            )
        state = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_running": len(self.running),
            "queue_depth": len(list(self.queue_dir.glob("*.json"))),
            "gpus": gpu_details,
        }
        self.gpu_state_file.write_text(json.dumps(state, indent=2))

    # ── orphan recovery ──

    def _recover_orphans(self) -> None:
        """Recover stale running specs left by a previous orchestrator crash."""
        for spec_file in sorted(self.running_dir.glob("*.json")):
            try:
                spec = json.loads(spec_file.read_text())
            except (json.JSONDecodeError, KeyError):
                spec_file.unlink()
                continue
            node_id = spec.get("node_id", spec_file.stem)
            logger.warning("Recovering orphan: %s", node_id)

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

            wt = self.runner.worktree_path(node_id)
            if wt.exists():
                try:
                    self.runner.cleanup_worktree(wt)
                except Exception as e:
                    logger.warning("Failed to cleanup orphan worktree %s: %s", wt, e)

    # ── scheduling ──

    def _get_pending(self) -> list[dict]:
        """Return queue specs sorted by priority then submission time."""
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
        """Find the GPU with tightest fit that can accommodate needed_gb VRAM.

        Uses nvidia-smi reported free memory (which already reflects running
        processes) minus a safety margin.  The max_per_gpu concurrency limit
        prevents over-scheduling beyond what the GPU can handle.

        Returns:
            GPU index, or None if no GPU has sufficient free memory.
        """
        candidates: list[tuple[int, float]] = []
        gpus = query_gpus()
        for gpu_idx, running_ids in self.gpu_allocations.items():
            if len(running_ids) >= self.max_per_gpu:
                continue
            gpu_info = next((g for g in gpus if g.index == gpu_idx), None)
            free_gb = gpu_info.free_gb if gpu_info else 24.0  # virtual GPU fallback
            schedulable = free_gb - self.safety_margin_gb
            if schedulable >= needed_gb:
                candidates.append((gpu_idx, schedulable))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[1])  # tightest fit first
        return candidates[0][0]

    def _pre_launch_check(self, gpu_id: int, needed_gb: float) -> bool:
        """Re-check GPU free memory immediately before launching."""
        gpus = query_gpus()
        gpu = next((g for g in gpus if g.index == gpu_id), None)
        if gpu is None:
            return True  # virtual GPU, skip check
        return gpu.free_gb >= needed_gb + self.safety_margin_gb

    def _resolve_launch_base_commit(self, spec: dict) -> str:
        """Validate and normalize the commit pinned in a queued spec."""
        base_commit = spec.get("base_commit")
        if not base_commit:
            raise ValueError("Queue spec missing required base_commit.")

        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{base_commit}^{{commit}}"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        if result.returncode != 0 or not commit:
            raise ValueError(f"Queue spec has invalid base_commit: {base_commit}")
        return commit

    # ── experiment lifecycle ──

    def _launch(self, spec: dict, gpu_id: int) -> None:
        """Launch a queued experiment on the given GPU."""
        node_id = spec["node_id"]
        move_type = spec.get("move_type", "branch")
        parent_ids = spec.get("parent_ids", [])

        archive_node = self.archive_dir / node_id
        archive_node.mkdir(parents=True, exist_ok=True)
        clean_spec = {k: v for k, v in spec.items() if k != "_file"}
        clean_spec["provenance"] = {
            "base_commit": clean_spec.get("base_commit"),
            "files": list(clean_spec.get("files", [])),
            "overlay_snapshot_present": (archive_node / "overlay").exists(),
        }
        (archive_node / "spec.json").write_text(json.dumps(clean_spec, indent=2))

        queue_file = spec.get("_file")
        if queue_file and Path(queue_file).exists():
            Path(queue_file).unlink()

        try:
            base_commit = self._resolve_launch_base_commit(spec)
        except ValueError as e:
            self._mark_crashed(node_id, clean_spec, str(e))
            return
        clean_spec["base_commit"] = base_commit
        clean_spec["provenance"]["base_commit"] = base_commit
        (archive_node / "spec.json").write_text(json.dumps(clean_spec, indent=2))
        try:
            wt_path = self.runner.create_worktree(base_commit, node_id)
        except subprocess.CalledProcessError as e:
            self._mark_crashed(node_id, clean_spec, f"Worktree creation failed: {e}")
            return

        # Apply file overlay if files were snapshot at queue time.
        overlay_dir = archive_node / "overlay"
        if overlay_dir.exists():
            self.runner.apply_overlay(wt_path, overlay_dir, [])

        parent_ckpt_dir = wt_path / ".parent_checkpoint"
        try:
            if move_type in ("branch", "seed") and parent_ids:
                self.runner.prepare_branch(parent_ids[0], self.checkpoint_store, wt_path)
            elif move_type == "merge":
                parent_ckpt_dir.mkdir(parents=True, exist_ok=True)
                weights = tuple(
                    spec.get("merge_weights") or [1.0 / len(parent_ids)] * len(parent_ids)
                )
                strategy = spec.get("merge_strategy") or self.default_merge_strategy
                self.runner.merge_checkpoints(
                    parent_ids=tuple(parent_ids),
                    weights=weights,
                    strategy=strategy,
                    checkpoint_store=self.checkpoint_store,
                    dest_path=parent_ckpt_dir,
                )
            elif move_type == "graft" and len(parent_ids) >= 2:
                parent_ckpt_dir.mkdir(parents=True, exist_ok=True)
                source_parent = spec.get("graft_source_parent", parent_ids[0])
                self.runner.graft_checkpoint(
                    source_id=parent_ids[0],
                    source_parent_id=source_parent,
                    target_parent_id=parent_ids[1],
                    checkpoint_store=self.checkpoint_store,
                    dest_path=parent_ckpt_dir,
                )
        except ImportError as e:
            self._mark_crashed(node_id, clean_spec, str(e))
            self.runner.cleanup_worktree(wt_path)
            return
        except (FileNotFoundError, OSError) as e:
            self._mark_crashed(node_id, clean_spec, f"Checkpoint preparation failed: {e}")
            self.runner.cleanup_worktree(wt_path)
            return

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

        log_path = archive_node / "run.log"
        log_fh = open(log_path, "w")
        if self.run_command:
            cmd = shlex.split(self.run_command)
        else:
            cmd = [sys.executable, self.run_script]

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(wt_path),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except OSError as e:
            log_fh.close()
            self._mark_crashed(node_id, clean_spec, f"Process launch failed: {e}")
            self.runner.cleanup_worktree(wt_path)
            return

        timeout_hours = spec.get("timeout_hours", self.default_timeout_hours)
        estimated_vram = spec.get("estimated_vram_gb", DEFAULT_VRAM_ESTIMATE_GB)
        exp = RunningExperiment(
            id=node_id,
            spec=clean_spec,
            gpu=gpu_id,
            process=process,
            log_file=log_fh,
            log_path=log_path,
            started_at=time.time(),
            timeout_at=time.time() + timeout_hours * 3600,
            estimated_vram_gb=estimated_vram,
        )
        self.running[node_id] = exp
        self.gpu_allocations.setdefault(gpu_id, []).append(node_id)
        (self.running_dir / f"{node_id}.json").write_text(json.dumps(clean_spec, indent=2))
        logger.info(
            "Launched %s on GPU %d (pid=%d, commit=%s)",
            node_id,
            gpu_id,
            process.pid,
            base_commit,
        )

    def _check_running(self) -> None:
        """Poll all running experiments for completion or timeout."""
        for node_id in list(self.running):
            exp = self.running[node_id]
            retcode = exp.process.poll()
            if retcode is not None:
                self._handle_completion(node_id, retcode)
            elif time.time() > exp.timeout_at:
                self._handle_timeout(node_id)

    def _handle_completion(self, node_id: str, returncode: int) -> None:
        """Process a finished experiment: collect artifacts, update intelligence."""
        exp = self.running.pop(node_id)
        if node_id in self.gpu_allocations.get(exp.gpu, []):
            self.gpu_allocations[exp.gpu].remove(node_id)
        exp.log_file.close()

        wt_path = self.runner.worktree_path(node_id)
        archive_node = self.archive_dir / node_id
        archive_node.mkdir(parents=True, exist_ok=True)
        result = self.runner.collect_result(wt_path, archive_node)

        if result and result.get("status") == "completed":
            status = "completed"
        else:
            status = self._infer_failure_status(node_id, returncode)

        elapsed = time.time() - exp.started_at
        metrics, composite, peak_vram = self._normalize_result(result)

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

        if result and result.get("checkpoint_saved"):
            ckpt_output = wt_path / ".checkpoint_output"
            if ckpt_output.exists():
                self.runner.collect_checkpoint(ckpt_output, self.checkpoint_store, node_id)

        running_spec = self.running_dir / f"{node_id}.json"
        if running_spec.exists():
            running_spec.unlink()
        self.runner.cleanup_worktree(wt_path)
        self._timed_out.pop(node_id, None)

        try:
            self._post_completion_update(node_id, status, completion)
        except Exception as e:
            logger.error("Post-completion update failed for %s: %s", node_id, e)

        self._append_results_tsv(completion)
        logger.info("Completed %s: status=%s elapsed=%.0fs", node_id, status, elapsed)

    def _infer_failure_status(self, node_id: str, returncode: int) -> str:
        """Infer failure type from timeout flag, OOM log pattern, or return code."""
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

    @staticmethod
    def _normalize_result(
        result: dict | None,
    ) -> tuple[dict[str, Any], float | None, int | None]:
        """Normalize a training script's result.json into internal format.

        Accepts both the flat format documented for training scripts::

            {"status": "completed", "task_score": 0.85, "cost": 1.5}

        and the nested format used internally::

            {"status": "completed", "metrics": {"task_score": 0.85}, "composite": 0.85}

        Returns:
            (metrics_dict, composite, peak_vram_mb) tuple.
        """
        if not result:
            return {}, None, None

        metrics = result.get("metrics", {})
        # If flat keys exist at top level but metrics is empty, build metrics
        # from the flat fields so both formats are accepted.
        _METRIC_KEYS = ("task_score", "retention_score", "cost")
        if not metrics:
            flat = {k: result[k] for k in _METRIC_KEYS if k in result}
            if flat:
                metrics = flat

        composite = result.get("composite")
        if composite is None and metrics.get("task_score") is not None:
            composite = metrics["task_score"]

        peak_vram = result.get("peak_vram_mb")
        return metrics, composite, peak_vram

    def _sync_manifold_coords(self, dag: Any, manifold: Any) -> None:
        """Project every executed node into the current manifold basis."""
        for node in dag.all_executed():
            dag.update_manifold_coords(node.id, manifold.project(node.theta))

    def _rebuild_forgetting_state(
        self,
        dag: Any,
        cliff_threshold: float,
        current_node_id: str | None = None,
    ) -> Any:
        """Recompute forgetting danger zones from DAG history in the active basis.

        Danger zones are basis-dependent. When the manifold later trains, the
        coordinate system changes, so previously recorded zones must be
        regenerated from grounded DAG nodes instead of incrementally appended.
        """
        from autotune.forgetting import ForgettingDetector

        forgetting = ForgettingDetector(cliff_threshold=cliff_threshold)
        current_cliff = None
        for node in sorted(
            dag.all_executed(),
            key=lambda item: (item.depth, item.created_at, item.id),
        ):
            if node.retention_score is None or not node.parent_ids:
                continue
            refreshed = dag.get_node(node.id)
            if refreshed is None:
                continue
            cliff = forgetting.check_cliff(refreshed, dag.parents(node.id))
            if node.id == current_node_id:
                current_cliff = cliff

        forgetting.save(self.autotune_dir / "forgetting.json")
        return current_cliff

    def _tail_log(self, log_path: Path, n: int) -> str:
        """Return the last n lines of a log file."""
        if not log_path.exists():
            return ""
        lines = log_path.read_text().splitlines()
        return "\n".join(lines[-n:])

    def _post_completion_update(self, node_id: str, status: str, completion: dict) -> None:
        """Update DAG, manifold, and forgetting after experiment completion."""
        from autotune.dag import ExperimentDAG
        from autotune.manifold import ManifoldNavigator

        dag_path = self.autotune_dir / "dag.json"
        if not dag_path.exists():
            return

        dag = ExperimentDAG.load(dag_path)
        config: dict = {}
        config_path = self.autotune_dir / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        if status == "completed":
            metrics = dict(completion.get("metrics", {}))
            metrics["composite"] = completion.get("composite")
            metrics["checkpoint_id"] = (
                node_id if (self.checkpoint_store / node_id).exists() else None
            )
            dag.promote(node_id, metrics)
        else:
            dag.mark_failed(node_id, status, completion.get("error"))

        dag.rerank_pareto()
        dag.recalculate_potential()

        params = config.get("parameters", {})
        param_names = list(params.keys())
        if param_names:
            manifold_path = self.autotune_dir / "manifold_state.json"
            if manifold_path.exists():
                manifold = ManifoldNavigator.load(manifold_path)
            else:
                param_bounds = {
                    n: (s.get("range", [0, 1])[0], s.get("range", [0, 1])[1])
                    for n, s in params.items()
                }
                manifold = ManifoldNavigator(param_names=param_names, param_bounds=param_bounds)
            executed = dag.all_executed()
            if len(executed) >= 10:
                points = [dict(n.theta) for n in executed]
                objectives = [(n.composite or 0.0,) for n in executed]
                manifold.update(points, objectives)
            self._sync_manifold_coords(dag, manifold)
            manifold.save(manifold_path)

        if status == "completed":
            node = dag.get_node(node_id)
            threshold = config.get("scoring", {}).get("retention_cliff_threshold", -0.10)
            if not param_names:
                if node and node.retention_score is not None:
                    logger.warning(
                        "Skipping forgetting zone rebuild for %s: no parameter space configured",
                        node_id,
                    )
            else:
                cliff = self._rebuild_forgetting_state(
                    dag,
                    cliff_threshold=threshold,
                    current_node_id=node_id,
                )
                if cliff:
                    logger.warning(
                        "Forgetting cliff detected for %s: delta=%.3f",
                        node_id,
                        cliff.delta_retention,
                    )

        self._prune_checkpoints(dag)
        dag.save()

    def _handle_timeout(self, node_id: str) -> None:
        """Kill a timed-out experiment and process it as a completion."""
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
        """Write a crash completion record without an associated running experiment."""
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
        (archive_node / "spec.json").write_text(
            json.dumps({k: v for k, v in spec.items() if k != "_file"}, indent=2)
        )
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
        """Append a row to results.tsv."""
        header = (
            "node_id\ttask_score\tretention_score\tcost\tcomposite\tvram_mb\telapsed_min\tstatus\n"
        )
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

    # ── main loop ──

    def tick(self) -> None:
        """Run one scheduling cycle: check running, launch pending, save state."""
        self._check_running()
        if not self.draining:
            for spec in self._get_pending():
                needed_gb = spec.get("estimated_vram_gb", DEFAULT_VRAM_ESTIMATE_GB)
                gpu = self._find_best_gpu(needed_gb)
                if gpu is not None and self._pre_launch_check(gpu, needed_gb):
                    self._launch(spec, gpu)
        self._save_state()

    def run(self) -> None:
        """Start the main orchestrator loop (blocks until shutdown signal)."""
        self.runner.prune_stale_worktrees()
        self._recover_orphans()

        def _signal_handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, shutting down...", signum)
            self._shutdown = True

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        self.pid_file.write_text(str(os.getpid()))
        logger.info(
            "Orchestrator started (pid=%d, gpus=%s, poll=%ds)",
            os.getpid(),
            list(self.gpu_allocations.keys()),
            self.poll_interval,
        )

        try:
            while not self._shutdown:
                try:
                    self.tick()
                except Exception:
                    logger.exception("Error in tick")
                time.sleep(self.poll_interval)

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
        """Start the orchestrator (checks for existing process first)."""
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                os.kill(pid, 0)
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
        """Print orchestrator status to stdout."""
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
        """Send SIGTERM to the running orchestrator process."""
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
