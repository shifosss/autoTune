# autoTune Orchestrator Design

**Date**: 2026-03-28
**Status**: Draft
**Scope**: `ExperimentOrchestrator`, runner checkpoint operations, CLI commands
**Depends on**: Architecture spec (2026-03-27), Phase 6.5 implementation (complete)

---

## 1. Overview

The orchestrator is the execution engine that bridges "proposed experiments" and "completed results." It reads experiment specs from `orchestrator/queue/`, manages GPU scheduling, spawns training processes with the correct env vars and checkpoints, collects results, updates the manifold and forgetting detector, and prunes stale checkpoints.

This design is a direct port of autoMIL's `ExperimentOrchestrator` with three extensions for autoTune's checkpoint-aware DAG:

1. **Checkpoint preparation in `_launch()`** — branch copies parent; merge combines parents via state_dict math; graft applies technique deltas.
2. **Post-completion intelligence in `_handle_completion()`** — triggers `manifold.update()` and `forgetting.check_cliff()` after each result.
3. **Checkpoint pruning via `_prune_checkpoints()`** — deletes checkpoint artifacts for Pareto rank >= 2 nodes.

### Design Principle: torch as Optional

`torch` (and `safetensors`) remain optional dependencies. The orchestrator starts and runs branch/seed experiments without them. Merge and graft moves fail at launch time with a clear error if torch is not installed. This matches autoMIL's pattern of optional dependencies and keeps branch-only workflows (the most common early-campaign pattern) accessible without a heavy install.

---

## 2. Files Changed

| File | Change | Current Lines | Projected |
|------|--------|--------------|-----------|
| `src/autotune/orchestrator.py` | **New file** | 0 | ~600 |
| `src/autotune/runner.py` | Add `merge_checkpoints`, `graft_checkpoint`, `_load_state_dict`, `_save_state_dict` | 197 | ~320 |
| `src/autotune/cli.py` | Add `orchestrator start\|stop\|status` + `checkpoints [--prune]` | 564 | ~684 |
| `tests/test_orchestrator.py` | **New file** | 0 | ~250 |
| `tests/test_runner.py` | Add merge/graft/load tests | 128 | ~200 |
| `tests/test_cli.py` | Add orchestrator + checkpoints tests | 678 | ~740 |

All files stay under 800 lines.

---

## 3. Data Structures

Defined at module level in `orchestrator.py`:

```python
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
```

Both are plain dataclasses (not frozen — `RunningExperiment` holds mutable handles).

---

## 4. Constants

```python
POLL_INTERVAL_SEC = 30
SAFETY_MARGIN_GB = 2.0
DEFAULT_TIMEOUT_HOURS = 8
MAX_CONCURRENT_PER_GPU = 2
DEFAULT_VRAM_ESTIMATE_GB = 0.5
```

All overridable via `autotune/config.yaml` under the `orchestrator:` section.

---

## 5. `ExperimentOrchestrator` Class

### 5.1 `__init__(self, project_root: Path)`

Initializes:
- Directory paths: `autotune_dir`, `orch_dir`, `queue_dir`, `running_dir`, `completed_dir`, `archive_dir`, `checkpoint_store`
- File paths: `pid_file` (`orchestrator.pid`), `log_file` (`orchestrator.log`), `gpu_state_file` (`gpu_state.json`), `results_tsv` (`results.tsv`)
- `self.runner = ExperimentRunner(project_root)`
- Config loading from `autotune/config.yaml`:
  - `run.script` (default `"train.py"`), `run.command` (optional override)
  - `run.checkpoint_format` (default `"safetensors"`)
  - `retention.enabled`, `retention.script`, `retention.probe_set`
  - `orchestrator.*` overrides for constants above
  - `merge.default_strategy` (default `"weighted_avg"`)
- Runtime state:
  - `self.running: dict[str, RunningExperiment]`
  - `self.gpu_allocations: dict[int, list[str]]`
  - `self._shutdown: bool = False`
  - `self.draining: bool = False`
- GPU detection via `query_gpus()`; falls back to `{0: []}` if nvidia-smi fails
- Directory creation (`mkdir(parents=True, exist_ok=True)`)
- `_load_state(recover=False)` — loads counter, does NOT recover orphans yet

### 5.2 `query_gpus() -> list[GPUInfo]` (module-level function)

Runs `nvidia-smi --query-gpu=index,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits`. Returns `list[GPUInfo]`. Returns `[]` on any failure (FileNotFoundError, timeout, parse error).

### 5.3 State Persistence

**`_load_state(recover: bool = False)`**
Reads `gpu_state.json` to restore counter. Calls `_recover_orphans()` only if `recover=True`.

**`_save_state()`**
Writes `gpu_state.json` with: counter, last_updated, total_running, queue_depth, per-GPU details (total_mb, free_mb, schedulable_free_gb, running list, utilization). Called at the end of every `tick()`.

### 5.4 Orphan Recovery: `_recover_orphans()`

Scans `running/` for `*.json` files left from a previous orchestrator instance. For each:
1. Read spec → get `node_id`
2. Write `completed/<node_id>.json` with `status: "crash"`, error: `"Orchestrator restarted while running"`
3. Write `archive/<node_id>/result.json` with crash status
4. Delete the `running/` spec file
5. Call `runner.cleanup_worktree()`

### 5.5 GPU Scheduling

**`_get_pending() -> list[dict]`**
Reads all `queue/*.json`, adds `"_file"` key (Path), sorts by `(priority ASC, submitted_at ASC)`.

**`_find_best_gpu(needed_gb: float) -> int | None`**
Best-fit bin packing:
1. For each GPU: skip if `len(running_on) >= max_per_gpu`
2. Compute `schedulable = free_gb - safety_margin - sum(allocated VRAM on this GPU)`
3. Keep GPUs where `schedulable >= needed_gb`
4. Return GPU with smallest schedulable headroom (tightest fit)
5. Return `None` if no GPU fits

**`_pre_launch_check(gpu_id: int, needed_gb: float) -> bool`**
Re-queries nvidia-smi right before launch to confirm `free_gb >= needed_gb + safety_margin`. Returns `False` if check fails (GPU under unexpected load).

### 5.6 Experiment Launch: `_launch(spec: dict, gpu_id: int)`

Ported from autoMIL with checkpoint extensions:

1. Create `archive/<node_id>/` directory
2. Save cleaned spec to `archive/<node_id>/spec.json`
3. Remove spec from `queue/` (prevents infinite retry on crash)
4. Call `runner.create_worktree(base_commit, node_id)`; on failure → `_mark_crashed()`
5. Apply overlay if `spec["overlay_dir"]` exists
6. **[NEW] Checkpoint preparation by move_type:**
   - `"branch"` or `"seed"`: call `runner.prepare_branch(parent_checkpoint_id, checkpoint_store, wt_path)`
   - `"merge"`: call `runner.merge_checkpoints(parent_ids, weights, strategy, checkpoint_store, wt_path / ".parent_checkpoint")`. Catches `ImportError` → `_mark_crashed("torch required for merge")`
   - `"graft"`: call `runner.graft_checkpoint(source_id, source_parent_id, target_parent_id, checkpoint_store, wt_path / ".parent_checkpoint")`. Same torch guard.
7. Build env dict:
   - `CUDA_VISIBLE_DEVICES=<gpu_id>`
   - `AUTOTUNE_GPU=0` (logical device after masking)
   - `AUTOTUNE_NODE_ID=<node_id>`
   - `AUTOTUNE_CHECKPOINT_DIR=<wt_path>/.checkpoint_output/` (worktree-local; collected to store on success)
   - `AUTOTUNE_PARENT_CHECKPOINT=<wt_path>/.parent_checkpoint/`
   - `AUTOTUNE_MOVE_TYPE=<move_type>`
   - If retention enabled: `AUTOTUNE_RETENTION_SCRIPT=<script>`, `AUTOTUNE_RETENTION_PROBE=<probe_set>`
   - Any extra keys from `spec["env"]`
8. Open `archive/<node_id>/run.log`
9. Spawn `subprocess.Popen(cmd, cwd=wt_path, stdout=log_fh, stderr=STDOUT, env=env)`
10. Create `RunningExperiment`, add to `self.running` and `self.gpu_allocations`
11. Copy spec to `running/<node_id>.json` for orphan recovery

### 5.7 Completion Handling: `_handle_completion(node_id: str, returncode: int)`

Ported from autoMIL with post-completion intelligence:

1. Pop from `self.running`, remove from `gpu_allocations`
2. Close `log_file`
3. Call `runner.collect_result(wt_path, archive)` → parse `result.json`
4. Determine status: if result exists and `status == "completed"` → completed. Else inspect log for OOM patterns → `"oom"`. Check `_timed_out` → `"timeout"`. Check `returncode != 0` → `"crash"`.
5. Build completion dict: `id, status, composite, metrics, elapsed_seconds, peak_vram_mb, gpu, completed_at`
6. For failures: add last 20 lines of log as `error`
7. Write `completed/<node_id>.json`
8. **[NEW] Collect checkpoint:** call `runner.collect_checkpoint(wt_path / checkpoint_dir, checkpoint_store, node_id)` if result indicates `checkpoint_saved: true`
9. Delete `running/<node_id>.json`
10. Call `runner.cleanup_worktree(wt_path)`
11. **[NEW] Post-completion intelligence (inside try/except — failures here must not lose the completion):**
    - Load DAG, call `dag.promote()` or `dag.mark_failed()` based on status
    - Call `dag.rerank_pareto()` + `dag.recalculate_potential()`
    - Gather all executed node thetas + objectives, call `manifold.update(points, objectives)` if >= 10 points
    - For promoted nodes: load parents, call `forgetting.check_cliff(node, parents)`
    - Call `_prune_checkpoints()`
    - Save DAG, manifold, forgetting state to disk
12. Append to `results.tsv`

### 5.8 Timeout Handling: `_handle_timeout(node_id: str)`

1. Send `SIGTERM` to process
2. Sleep 5 seconds
3. If still alive: `SIGKILL`
4. Set `_timed_out[node_id] = True`
5. Calls `_handle_completion(node_id, returncode=-9)`

### 5.9 Crash Marking: `_mark_crashed(node_id: str, spec: dict, error: str)`

Used when launch fails before a process exists. Writes `archive/result.json` and `completed/<node_id>.json` both with `status="crash"`.

### 5.10 Checkpoint Pruning: `_prune_checkpoints()` [NEW]

After Pareto reranking:
1. Load DAG, iterate all nodes
2. For nodes with `pareto_rank >= checkpoint_retention_layers` (config, default 2):
   - If `checkpoint_store/<node_id>/` exists, `shutil.rmtree()` it
   - Log what was pruned
3. Node stays in DAG — only the checkpoint artifact is removed

### 5.11 Results TSV: `_append_results_tsv(completion: dict)`

Appends one row to `autotune/results.tsv`. Header:
```
node_id	task_score	retention_score	cost	composite	vram_gb	elapsed_min	status	description
```

### 5.12 Main Loop: `tick()` and `run()`

**`tick()`**
1. `_check_running()` — poll all live processes
2. If not `self.draining`: `_get_pending()` → for each spec, `_find_best_gpu()`, `_pre_launch_check()`, `_launch()`
3. `_save_state()`

**`run()`**
1. `runner.prune_stale_worktrees()`
2. `_recover_orphans()`
3. Install SIGTERM + SIGINT handlers → set `self._shutdown = True`
4. Write PID to `orchestrator.pid`
5. Main loop: `while not self._shutdown: tick(); sleep(poll_interval)`. Tick errors caught and logged.
6. Graceful drain: set `self.draining = True`, loop `_check_running(); sleep(5)` until `self.running` is empty
7. `finally`: delete PID file, `_save_state()`

### 5.13 CLI Instance Methods

**`cmd_start()`**: Check PID file (is process alive?). Setup logging (file + console). Call `self.run()`.

**`cmd_status()`**: Check PID validity. Read `gpu_state.json`. Print: PID, uptime, queue depth, running count, per-GPU details.

**`cmd_stop()`**: Send `SIGTERM` to PID from file.

---

## 6. Runner Extensions

### 6.1 `_load_state_dict(path: Path) -> dict`

Private helper. Loading precedence:
1. If path is a directory, look for `adapter_model.safetensors` → `pytorch_model.bin` → `model.safetensors`
2. Try `safetensors.torch.load_file(path)` if safetensors installed and file is `.safetensors`
3. Fall back to `torch.load(path, weights_only=True)`
4. Raise `ImportError` if torch not installed

### 6.2 `_save_state_dict(state_dict: dict, path: Path, fmt: str)`

Private helper. `fmt` is `"safetensors"` or `"pytorch"`:
- `"safetensors"`: use `safetensors.torch.save_file(state_dict, path)`, fall back to torch if safetensors not installed
- `"pytorch"`: use `torch.save(state_dict, path)`

### 6.3 `merge_checkpoints(parent_ids, weights, strategy, checkpoint_store, dest_path)`

```python
def merge_checkpoints(
    self,
    parent_ids: tuple[str, ...],
    weights: tuple[float, ...],
    strategy: str,
    checkpoint_store: Path,
    dest_path: Path,
) -> None:
```

- **`weighted_avg`**: Load all parent state_dicts. For each key present in ALL dicts: `merged[key] = sum(w_i * parent_i[key])`. Save to `dest_path`.
- **`compose`**: Load first parent as base. For each subsequent parent: `delta = parent_i - base`. Accumulate deltas. Save `base + accumulated_delta`.
- **`selective`**: Raises `NotImplementedError("selective merge requires layer_manifest config (not yet supported)")`.
- Unknown strategy: raises `ValueError`.

### 6.4 `graft_checkpoint(source_id, source_parent_id, target_parent_id, checkpoint_store, dest_path)`

```python
def graft_checkpoint(
    self,
    source_id: str,
    source_parent_id: str,
    target_parent_id: str,
    checkpoint_store: Path,
    dest_path: Path,
) -> None:
```

Logic: `delta = source - source_parent`, `grafted = target_parent + delta`. Save to `dest_path`.

---

## 7. CLI Commands

### 7.1 `orchestrator` subgroup

```
autotune orchestrator start    # Foreground daemon; user daemonizes externally
autotune orchestrator stop     # SIGTERM to PID from orchestrator.pid
autotune orchestrator status   # Print PID, GPU state, queue, running experiments
```

Each subcommand: find project root → instantiate `ExperimentOrchestrator(root)` → call `cmd_*()`.

### 7.2 `checkpoints` command

```
autotune checkpoints           # List: node_id | pareto_rank | status | size
autotune checkpoints --prune   # Delete rank >= 2 checkpoint dirs
```

Standalone — does not require orchestrator to be running. Loads DAG, scans `orchestrator/checkpoints/`, cross-references.

---

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| No GPUs detected | Log warning, use virtual GPU `{index: 0}` with no VRAM tracking. Launch without `CUDA_VISIBLE_DEVICES`. |
| torch not installed for merge/graft | `_launch()` catches `ImportError`, calls `_mark_crashed()`. Branch moves unaffected. |
| Training script missing `result.json` | Infer status from log (OOM patterns) or returncode. Write `crash` completion. |
| Orchestrator killed mid-experiment | `_recover_orphans()` on next `run()` marks all stale running as `crash`. |
| Parent checkpoint missing | `_launch()` catches `FileNotFoundError`, calls `_mark_crashed()`. |
| Timeout exceeded | SIGTERM → 5s grace → SIGKILL. Status `"timeout"`. |
| DAG load fails during completion | Log error, still write `completed/*.json` so `autotune reconcile` picks it up later. |

---

## 9. Queue Spec JSON Schema

Written by `autotune submit --node <id>` (already implemented):

```json
{
  "node_id": "node_0002",
  "theta": {"lr": 5e-4, "rank": 16},
  "parent_ids": ["node_0001"],
  "move_type": "branch",
  "files": ["train.py"]
}
```

The orchestrator enriches this at launch time with: `base_commit` (from `git rev-parse HEAD` at launch time), `overlay_dir`, and timing metadata. Missing optional fields default: `estimated_vram_gb` → config default, `timeout_hours` → config default, `priority` → 1.

---

## 10. Completed JSON Schema

Written by `_handle_completion()`:

```json
{
  "node_id": "node_0002",
  "status": "completed",
  "metrics": {
    "task_score": 0.85,
    "retention_score": 0.88,
    "cost": 1.2,
    "composite": 0.85
  },
  "elapsed_seconds": 4170,
  "peak_vram_mb": 18200,
  "gpu": 1,
  "completed_at": "2026-03-28T14:30:00"
}
```

For failures, adds `"error"` (last 20 log lines) and `"log_location"`.

---

## 11. Testing Strategy

| Test File | Scope | Approach |
|-----------|-------|----------|
| `tests/test_orchestrator.py` | GPU parsing, best-fit scheduling, state persistence, orphan recovery, tick cycle, checkpoint pruning | Mock `subprocess.Popen`, `nvidia-smi`, filesystem |
| `tests/test_runner.py` (additions) | `merge_checkpoints` weighted_avg + compose, `graft_checkpoint` arithmetic, torch-not-installed guard, `_load_state_dict` format detection | Small fake state_dict dicts, mock torch |
| `tests/test_cli.py` (additions) | `orchestrator status` output, `checkpoints` listing, `checkpoints --prune` | Mock `ExperimentOrchestrator`, real filesystem |

No real GPU or torch required in CI. All subprocess and GPU interactions are mocked.
