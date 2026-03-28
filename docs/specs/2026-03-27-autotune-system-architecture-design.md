# autoTune System Architecture Design

**Date**: 2026-03-27
**Status**: Approved
**Scope**: Full system architecture for the autoTune finetuning framework

---

## 1. Overview

autoTune is an autonomous experiment framework for neural network and LLM finetuning. It overlays onto existing ML project repos, enabling coding agents to iteratively improve finetuned models through persistent experimentation, knowledge accumulation, and multi-branch exploration.

### What Makes autoTune Novel

autoTune introduces three innovations over existing HPO tools (Optuna, BOHB, Freeze-Thaw):

1. **DAG-structured checkpoint management** — experiments form a directed acyclic graph, not flat trials or a tree. Nodes can have multiple parents via merge operations, enabling knowledge recombination across branches.

2. **Manifold-based hyperparameter navigation** — CMA-ES learns parameter correlations and proposes moves along discovered principal components (ridges), not axis-aligned individual parameters. The system navigates a learned low-dimensional surface rather than the full hyperparameter space.

3. **Forgetting-aware multi-objective Pareto optimization** — every experiment is evaluated on task performance, retention (resistance to catastrophic forgetting), and cost. Forgetting cliffs are detected and marked as danger zones on the manifold. Checkpoint pruning and exploration steering are informed by retention health.

### Design Principles

- **Agent-driven**: coding agents (Claude) make strategic decisions (move type, parent selection, what code to change). The manifold navigator provides numerical guidance, not commands.
- **Method-agnostic**: the framework doesn't know or care about LoRA vs QLoRA vs DoRA vs full finetuning. It manages hyperparameters, checkpoints, and the DAG. The training script handles the method.
- **User-defined retention**: retention evaluation is a contract (script + probe set), not a built-in metric. If not configured, the system operates in 2-objective mode (task x cost).
- **Overlay model**: autoTune drops an `autotune/` directory into an existing repo. It never creates training scripts.

### Architectural Approach

**Sovereign Core + Optuna Substrate**: autoTune owns the DAG, state machine, agent contract, and control flow. Optuna is used as a library dependency for CMA-ES sampling (manifold covariance learning), NSGA-III (Pareto front computation), and optionally RDB storage for trial history. Optuna does not own the loop or the state.

---

## 2. Project Structure & Overlay Model

### Directory Layout

```
user-project/                          # existing finetuning repo
├── train.py                           # user's training script (autoTune never creates this)
├── eval_retention.py                  # user's retention eval script (optional)
├── autotune/                          # overlay created by `autotune init`
│   ├── config.yaml                    # project config
│   ├── program.md                     # agent loop instructions
│   ├── learnings.md                   # accumulated agent insights
│   ├── dag.json                       # experiment DAG (runtime, gitignored)
│   ├── manifold_state.json            # learned covariance + PCs (runtime, gitignored)
│   ├── results.tsv                    # flat log (runtime, gitignored)
│   ├── .gitignore
│   └── orchestrator/
│       ├── queue/                     # pending experiment specs
│       ├── running/                   # in-flight sentinels
│       ├── completed/                 # notification JSONs for reconcile
│       ├── archive/                   # per-experiment: spec.json + run.log + result.json
│       └── checkpoints/              # managed checkpoint storage
│           ├── node_0001/            # saved model weights / LoRA adapters
│           ├── node_0002/
│           └── merged_0001/          # merged checkpoint artifacts
├── .autotune_worktrees/              # git worktrees (ephemeral)
└── .autotune_retention_probe/        # cached retention probe data (optional)
```

### Key Difference from autoMIL

The `checkpoints/` directory is new. autoMIL doesn't manage checkpoints — experiments produce `result.json` and that's it. autoTune must persist checkpoints because children branch from them and merges combine them. This is the fundamental new resource the framework manages.

**Checkpoint storage contract**: training scripts must save their checkpoint to a path provided via the `AUTOTUNE_CHECKPOINT_DIR` environment variable. The framework handles lifecycle (keep, prune, merge).

### Result Contract

Training scripts write `result.json` to their working directory:

```json
{
  "status": "completed",
  "metrics": {
    "task_loss": 0.42,
    "task_accuracy": 0.88,
    "custom_metric": 0.91
  },
  "retention": {
    "score": 0.93,
    "details": {"perplexity": 12.4, "mmlu": 0.82}
  },
  "composite": 0.85,
  "elapsed_seconds": 4098,
  "peak_vram_mb": 18200,
  "checkpoint_saved": true
}
```

The `retention` block is optional. If omitted, the system operates in 2-objective mode (task x cost).

---

## 3. Core Data Structures

### DAGNode

```python
@dataclass(frozen=True)
class DAGNode:
    id: str                                # "node_0042"
    parent_ids: tuple[str, ...]            # () root, (p,) branch, (p1,p2) merge
    move_type: str                         # "branch" | "merge" | "graft" | "seed"

    # Merge metadata (None for branch/graft/seed)
    merge_strategy: str | None             # "weighted_avg" | "compose" | "selective"
    merge_weights: tuple[float, ...] | None

    # Position on the manifold
    theta: dict[str, float]                # {"lr": 2e-4, "rank": 16, ...}
    manifold_coords: tuple[float, ...] | None  # projected PC coordinates (computed)

    # What the agent decided
    strategy_desc: str                     # "increase capacity along PC1"
    technique_tags: tuple[str, ...]        # ("lr_increase", "rank_increase")

    # Outcomes (None while pending/running)
    task_score: float | None
    retention_score: float | None
    cost: float | None                     # vram_gb x elapsed_hours
    composite: float | None

    # Checkpoint
    checkpoint_id: str | None              # reference into checkpoints/ dir

    # Framework-computed
    depth: int                             # longest path from any root
    potential: float                       # DAG-UCB score
    pareto_rank: int | None                # 0 = Pareto-optimal, 1 = near-front, 2+ = dominated

    # Lifecycle
    node_type: str                         # "executed" | "proposed"
    status: str                            # pending | running | keep | discard | oom | crash | timeout | cancelled
    created_at: str
```

Key differences from autoMIL's node:
- `parent_ids` is a tuple (supports merge with multiple parents)
- `theta` stores the full hyperparameter vector for manifold computation
- `retention_score` is a first-class field
- `pareto_rank` replaces binary keep/discard with layered Pareto ranking
- `move_type` records how the node was created

### DAGMeta

```python
@dataclass(frozen=True)
class DAGMeta:
    next_id: int
    total_executed: int
    total_proposed: int
    best_composite: float | None
    best_node_id: str | None
    objective_count: int                   # 2 (task x cost) or 3 (task x retention x cost)
    scoring: ScoringConfig

@dataclass(frozen=True)
class ScoringConfig:
    exploration_weight: float              # UCB exploration term
    diversity_weight: float                # merge diversity bonus
    retention_cliff_threshold: float       # delta_retention that triggers cliff warning
```

### TechniqueStats

```python
@dataclass(frozen=True)
class TechniqueStats:
    times_tried: int
    avg_task_delta: float
    avg_retention_delta: float             # tracks forgetting impact per technique
    best_parent_delta: float
```

### ManifoldState

```python
@dataclass(frozen=True)
class ManifoldState:
    param_names: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]  # d x d covariance matrix
    principal_components: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    effective_dim: int                     # number of significant PCs
    mean: tuple[float, ...]               # centroid of evaluated points
    step_sizes: tuple[float, ...]         # per-PC exploration radius
    generation: int                        # CMA-ES generation count

@dataclass(frozen=True)
class DangerZone:
    center: tuple[float, ...]              # manifold coordinates
    radius: float
    cause_technique: str
    severity: float                        # how steep the retention drop was
```

---

## 4. Module Architecture

### Module Map

```
src/autotune/
├── dag.py              # DAG graph: CRUD, Pareto ranking, reconciliation
├── manifold.py         # CMA-ES covariance learning, PC extraction, proposals
├── forgetting.py       # Retention surface, cliff detection, danger zones
├── navigator.py        # Strategy layer: combines DAG scores + manifold + forgetting
├── orchestrator.py     # GPU scheduler, checkpoint lifecycle, process management
├── runner.py           # Git worktree + checkpoint operations (branch/merge/graft)
├── cli.py              # Agent-facing CLI commands
├── templates/          # Jinja2 templates for init
│   ├── config.yaml.j2
│   ├── program.md.j2
│   ├── learnings.md.j2
│   └── .gitignore.j2
└── viz/
    ├── server.py       # SSE dashboard (aiohttp + watchdog)
    └── static/
        ├── index.html
        └── app.js      # 3D DAG visualization (ForceGraph3D)
```

### Module Responsibilities

#### `dag.py` (~400 lines)

Owns: `dag.json` persistence, node CRUD, Pareto ranking, reconciliation, technique stats.

```python
class ExperimentDAG:
    # Node operations
    def add_seed(node_id, theta, description, techniques) -> DAGNode
    def add_proposed(parent_ids, move_type, description, techniques, ...) -> DAGNode
    def mark_running(node_id) -> DAGNode
    def promote(node_id, metrics) -> DAGNode
    def mark_failed(node_id, status, error) -> DAGNode
    def cancel(node_id) -> DAGNode

    # DAG queries
    def get_node(node_id) -> DAGNode | None
    def parents(node_id) -> tuple[DAGNode, ...]
    def children(node_id) -> list[DAGNode]
    def ancestors(node_id) -> list[DAGNode]
    def lineage(node_id) -> list[list[DAGNode]]    # all root-to-node paths
    def pareto_front(layer=0) -> list[DAGNode]
    def kept_nodes() -> list[DAGNode]              # pareto_rank 0-1

    # Scoring
    def rerank_pareto() -> None                    # NSGA-II non-dominated sorting
    def recalculate_potential() -> None             # DAG-UCB on all nodes
    def rank_proposals(n, max_per_branch) -> list[DAGNode]
    def technique_stats(tag) -> TechniqueStats

    # Reconciliation
    def reconcile(queue, running, completed, archive) -> None

    # Persistence
    def save() -> None                             # atomic write via mkstemp+rename
    @staticmethod
    def load(path) -> ExperimentDAG
```

#### `manifold.py` (~300 lines)

Owns: `manifold_state.json`, covariance estimation, PC extraction. Wraps Optuna's CMA-ES sampler for covariance update logic.

```python
class ManifoldNavigator:
    # Learning
    def update(points: list[dict], objectives: list[tuple]) -> ManifoldState

    # Proposals
    def propose_direction(from_theta, pc_index=None) -> dict[str, float]
    def propose_candidates(from_theta, n) -> list[dict[str, float]]

    # Queries
    def project(theta) -> tuple[float, ...]
    def effective_dimensionality() -> int
    def principal_components() -> list[PC]
    def parameter_importance() -> dict[str, float]

    # State
    def state() -> ManifoldState
    def save() -> None
    @staticmethod
    def load(path) -> ManifoldNavigator
```

Does NOT decide what to explore. Only answers "what are the correlations?" and "given a direction, what's the new theta?"

#### `forgetting.py` (~200 lines)

Owns: danger zone tracking, cliff detection, retention surface analysis.

```python
class ForgettingDetector:
    # Detection
    def check_cliff(node: DAGNode, parents: tuple[DAGNode, ...]) -> CliffReport | None
    def is_in_danger_zone(theta: dict, manifold: ManifoldNavigator) -> bool

    # Analysis
    def retention_velocity(node_id, dag: ExperimentDAG) -> float
    def worst_technique_for_retention() -> str

    # State
    def add_danger_zone(center, radius, cause, severity) -> None
    def danger_zones() -> tuple[DangerZone, ...]
    def save() -> None
    @staticmethod
    def load(path) -> ForgettingDetector
```

If user doesn't provide retention eval, this module is effectively inactive. `ForgettingDetector` is the sole owner of `DangerZone` state — `ManifoldNavigator` queries it via `ForgettingDetector.is_in_danger_zone()` but does not store danger zones itself.

#### `navigator.py` (~350 lines)

**New module with no autoMIL equivalent.** Bridge between agent's high-level strategy and low-level numerical proposals. Combines DAG scores, manifold directions, and forgetting constraints.

```python
class StrategyNavigator:
    def __init__(self, dag, manifold, forgetting): ...

    # Move proposals
    def suggest_moves(n, budget_hours=None) -> list[MoveProposal]
    def score_move(move: MoveProposal) -> float

    # Agent assistance
    def explain_manifold() -> str
    def explain_frontier() -> str
    def recommend_strategy() -> str
```

Agent calls `suggest_moves()` to get recommendations, then uses its own judgment to accept, modify, or override.

#### `orchestrator.py` (~600 lines)

Extends autoMIL's orchestrator with checkpoint management.

```python
class ExperimentOrchestrator:
    # Same as autoMIL
    def run() -> None
    def tick() -> None

    # Extended for autoTune
    def _launch(spec, gpu_id) -> None
        # Sets AUTOTUNE_CHECKPOINT_DIR, AUTOTUNE_PARENT_CHECKPOINT
        # For MERGE: calls runner.merge_checkpoints() first
        # For GRAFT: calls runner.graft_checkpoint() first

    def _handle_completion(node_id, returncode) -> None
        # Same as autoMIL + registers checkpoint
        # Triggers manifold.update() and forgetting.check_cliff()

    # Checkpoint lifecycle
    def _prune_checkpoints() -> None
        # Delete checkpoints for pareto_rank >= 2 nodes
```

#### `runner.py` (~200 lines)

Extends autoMIL's runner with checkpoint operations.

```python
class ExperimentRunner:
    # From autoMIL (unchanged)
    def create_worktree(base_commit, node_id) -> Path
    def apply_overlay(wt_path, overlay_dir, deletions) -> None
    def collect_result(wt_path, archive_dir) -> dict | None
    def cleanup_worktree(wt_path) -> None

    # New for autoTune
    def collect_checkpoint(wt_path, checkpoint_store, node_id) -> str | None
    def prepare_branch(checkpoint_id, wt_path) -> None
    def merge_checkpoints(parent_ids, weights, strategy, dest) -> str
    def graft_checkpoint(source_node, target_parent, dest) -> str
```

#### `cli.py` (~450 lines)

```
autotune init [--task] [--base-model]
autotune submit --node <id> --desc "..." --files f1.py [--parent <id>]
autotune propose --parents <id> [<id2>] --move <branch|merge|graft> --desc "..."
autotune suggest [--n 5]
autotune explain
autotune rank [--n 10]
autotune reconcile
autotune status
autotune check
autotune frontier
autotune checkpoints [--prune]
autotune orchestrator start|stop|status
autotune viz start|stop|status
```

### Module Dependency Graph

```
cli.py
  ├── dag.py            (direct)
  ├── navigator.py      (direct)
  ├── orchestrator.py   (direct)
  └── viz/server.py     (direct)

navigator.py
  ├── dag.py            (reads DAG state)
  ├── manifold.py       (gets directions)
  └── forgetting.py     (checks danger zones)

orchestrator.py
  ├── runner.py          (worktree + checkpoint ops)
  ├── manifold.py        (triggers update on completion)
  └── forgetting.py      (triggers cliff check on completion)

dag.py          → standalone (no deps on other autotune modules)
manifold.py     → standalone (wraps Optuna CMA-ES)
forgetting.py   → standalone
runner.py       → standalone (git + file operations)
```

---

## 5. Key Algorithms

### 5.1 DAG-UCB Scoring

**Branch candidates** (single parent):

```
potential(node) = composite
    + w_e x sqrt(ln(N) / (1 + child_count))
    x retention_health(node)

where:
  retention_health(node) =
    1.0                         if retention_score is None
    retention_score             if retention_score >= cliff_threshold
    retention_score x 0.5       if retention_score < cliff_threshold
```

**Merge candidates** (pair of parents):

```
merge_potential(A, B) =
    0.5 x (A.composite + B.composite)
    + w_e x sqrt(ln(N) / (1 + merge_count(A, B)))
    + w_div x diversity(A, B)
    x complementarity(A, B)

where:
  diversity(A, B) = 1 - cosine_similarity(A.manifold_coords, B.manifold_coords)
  complementarity(A, B) =
    1.0 + |A.task_score - B.task_score| x |A.retention_score - B.retention_score|
```

**Graft candidates**:

```
graft_potential(technique_source, target_parent) =
    target_parent.composite
    + technique_source.parent_delta
    + w_e x sqrt(ln(N) / (1 + graft_count(source_tech, target)))
    x retention_health(target_parent)
```

### 5.2 Pareto Ranking (NSGA-II Non-Dominated Sorting)

Replaces autoMIL's binary parent-dominance check with full non-dominated sorting across ALL executed nodes:

```
Input: all executed nodes with objectives (task, retention, cost)
       (2-objective mode if no retention: (task, cost))

Algorithm:
  remaining = all executed nodes
  rank = 0
  while remaining:
    front = [n for n in remaining
             if no other node in remaining dominates n on ALL objectives]
    for n in front:
      n.pareto_rank = rank
      n.status = "keep" if rank <= 1 else "discard"
    remaining -= front
    rank += 1

Dominance: A dominates B iff
  A.task_score >= B.task_score AND
  A.retention_score >= B.retention_score AND  (skipped in 2-obj mode)
  A.cost <= B.cost AND
  at least one inequality is strict
```

Checkpoint pruning: only layer 0-1 checkpoints are retained. Layer 2+ checkpoints are deleted.

### 5.3 Manifold Covariance Update (CMA-ES Core)

After each batch of completed experiments:

```
1. Normalize theta vectors to [0,1] range per parameter (using config bounds)
2. Rank points by composite score, take top-k (mu best)
3. Compute weighted mean of top-k:
   mu_new = sum(w_i x theta_i)
4. Update covariance estimate:
   sigma_new = (1-c) x sigma_old + c x sum(w_i x (theta_i - mu_old)(theta_i - mu_old)^T)
5. Eigendecompose sigma_new:
   eigenvalues, eigenvectors = eigh(sigma_new)
   PCs = eigenvectors sorted by eigenvalue descending
6. Determine effective dimensionality:
   effective_dim = min k such that sum(eigenvalues[:k]) / sum(eigenvalues) > 0.95
7. Update step sizes per PC:
   If PC_i direction produced keeps: increase step_size_i x 1.2
   If PC_i direction produced discards: decrease step_size_i x 0.8
8. Persist to manifold_state.json
```

For the initial phase (< 10 evaluated points), the manifold is unreliable and proposals fall back to random sampling within bounds.

### 5.4 Forgetting Cliff Detection

```
On each experiment completion:

1. Compute delta_retention:
   BRANCH: delta = node.retention - parent.retention
   MERGE:  delta = node.retention - max(parent_1.retention, parent_2.retention)
   GRAFT:  delta = node.retention - target_parent.retention

2. If delta < cliff_threshold (default -0.10):
   - Create DangerZone at node.manifold_coords
   - Record cause_technique from node.technique_tags
   - Update technique_stats: avg_retention_delta for each tag
   - Log warning for agent

3. Danger zone radius:
   radius = |delta| x step_size_of_direction_taken

4. Future proposals check:
   navigator.suggest_moves() filters out proposals in danger zones
```

### 5.5 Checkpoint Merge Operations

```
weighted_avg(checkpoints, weights):
  Load state_dicts from all parents
  For each parameter key present in all dicts:
    merged[key] = sum(w_i x parent_i[key])
  Save merged state_dict

compose(checkpoints):  # LoRA-specific
  Load base model state_dict
  For each parent checkpoint:
    Load LoRA delta (adapter weights)
    Accumulate: merged_delta += delta_i
  Save base + merged_delta

selective(checkpoints, layer_manifest):
  For each layer range in manifest:
    Copy parameters from specified parent
  Save assembled state_dict

graft(source_node, source_parent, target_parent):
  delta = source_node.checkpoint - source_parent.checkpoint
  grafted = target_parent.checkpoint + delta
  Save grafted state_dict
```

All merge operations are method-agnostic at the state_dict level. For specialized merge logic, the user provides a custom merge hook in config.

---

## 6. Agent Integration

### The Agent Loop

```
1. OBSERVE
   autotune status          -> queue/running/done
   autotune reconcile       -> sync DAG with results
   autotune explain         -> manifold summary, frontier, strategy

2. ANALYZE
   Read learnings.md        -> past insights
   autotune frontier        -> current Pareto front
   autotune suggest --n 5   -> navigator proposals
   Review completed results -> what worked/failed

3. DECIDE (agent's creative judgment)
   Choose: accept a suggestion, modify it, or devise own strategy
   Pick move type: branch / merge / graft
   Select parent(s)
   Decide what to change in the training code

4. IMPLEMENT
   Edit training script, config, data pipeline
   (Agent has full codebase access)

5. SUBMIT
   autotune propose --parents <ids> --move <type> --desc "..." --techniques "..."
   autotune submit --node <id> --files changed.py

6. REFLECT
   Update learnings.md with insights

-> Wait for orchestrator to complete, repeat
```

### How the Agent Uses Navigator Suggestions

The agent calls `autotune suggest` and receives ranked move proposals:

```
1. BRANCH from node_0012 (score: 0.847)
   Direction: PC1 (+capacity) magnitude 0.3
   Suggested theta: lr=4.2e-4, rank=28, alpha=56
   Reasoning: node_0012 has high retention budget (0.94),
     PC1 moves consistently improve task score

2. MERGE node_0008 + node_0015 (score: 0.812)
   Weights: 0.55 / 0.45 (weighted_avg)
   Reasoning: high diversity (cosine_dist=0.72),
     complementary strengths

3. BRANCH from node_0015 (score: 0.798)
   Direction: PC2 (+regularization) magnitude 0.2
   Suggested theta: dropout=0.15, warmup=0.08
```

The agent can:
- **Accept**: follow the suggestion exactly
- **Modify**: follow the direction but also change code (e.g., switch to DoRA)
- **Override**: ignore the suggestion and try its own strategy

### Agent Autonomy Model

| Layer | Who decides | Mechanism |
|-------|-------------|-----------|
| High-level strategy | Agent | "branch vs. merge vs. graft", "which parent", "which technique category" |
| Numerical parameters | Manifold navigator | "lr=4.2e-4, rank=28" along PC1 |
| Code changes | Agent | full codebase access, edits training scripts |
| Keep/discard | Framework | Pareto ranking, automatic |
| Checkpoint lifecycle | Framework | prune rank 2+, keep rank 0-1 |

---

## 7. Environment Variables & Contracts

### Environment Variables Set by Orchestrator

```
CUDA_VISIBLE_DEVICES=2
AUTOTUNE_GPU=0
AUTOTUNE_NODE_ID=node_0042
AUTOTUNE_CHECKPOINT_DIR=/path/to/checkpoints/node_0042/
AUTOTUNE_PARENT_CHECKPOINT=/path/to/checkpoints/node_0038/
AUTOTUNE_MOVE_TYPE=branch
AUTOTUNE_RETENTION_SCRIPT=eval_retention.py
AUTOTUNE_RETENTION_PROBE=retention_probe/
```

### Training Script Responsibilities

Must:
1. Load parent checkpoint from `AUTOTUNE_PARENT_CHECKPOINT` if set
2. Save checkpoint to `AUTOTUNE_CHECKPOINT_DIR`
3. Write `result.json` to working directory

Must NOT:
- Decide keep/discard
- Manage GPU allocation
- Merge checkpoints
- Know about DAG topology

### Checkpoint Format

Framework-agnostic. Supports:
- PyTorch (`.pt`, `.bin`)
- SafeTensors (`.safetensors`)
- HuggingFace format (directory with `adapter_config.json`)
- Custom (user provides merge hook in config)

### Retention Eval Contract

If configured, `eval_retention.py` is called with:
```
--checkpoint-dir <path> --probe-dir <path>
```

Must write `retention_result.json`:
```json
{
  "score": 0.93,
  "details": {"perplexity": 12.4, "mmlu": 0.82}
}
```

---

## 8. Configuration Schema

```yaml
project:
  name: "my-finetune-project"
  base_model: "meta-llama/Llama-3-8B"

task:
  type: "instruction_tuning"

run:
  script: "train.py"
  checkpoint_format: "safetensors"

retention:
  enabled: true
  script: "eval_retention.py"
  probe_set: "retention_probe/"

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
  data_mix_ratio:
    type: float
    range: [0.3, 1.0]

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
    - "data/"
  readonly:
    - "eval_retention.py"
```

The `parameters` section defines the manifold dimensions with types, ranges, and optional log scaling.

---

## 9. Visualization

### Server Architecture

Same pattern as autoMIL: aiohttp + watchdog SSE. Monitors `dag.json` and `manifold_state.json`.

### Dashboard Tabs

**Tab 1: DAG View (3D Force Graph)**
- Nodes: spheres sized by composite, colored by status
- Edges: directed arrows from parent(s) to child
- Merge nodes: diamond shape with two incoming edges
- Graft nodes: dashed incoming edge from technique source
- Pareto spine: highlight paths from roots to Pareto rank-0 nodes
- Danger zones: translucent red spheres at forgetting cliff locations

**Tab 2: Pareto Frontier**
- 2D or 3D scatter of all executed nodes
- Axes: task_score x retention_score (x cost as bubble size if 3-obj)
- Pareto front highlighted as connected line/surface
- Color by pareto_rank

**Tab 3: Manifold Explorer**
- 2D projection onto top 2 PCs
- PC direction arrows labeled with parameter contributions
- Danger zones as red shaded regions
- Step size ellipse around current mean

**Tab 4: Leaderboard**
- Top-N nodes by composite
- Columns: node_id, task_score, retention_score, cost, pareto_rank, depth, move_type, technique_tags

**Tab 5: Technique Effectiveness**
- Table: times_tried, avg_task_delta, avg_retention_delta, best_parent_delta
- Color-coded green (helps) / red (hurts retention)

---

## 10. Node State Machine

```
add_seed()        -> type=executed, status=keep (baseline)
add_proposed()    -> type=proposed, status=pending
mark_running()    -> status=running
promote()         -> type=executed, status determined by Pareto ranking
mark_failed()     -> type=executed, status=oom|crash|timeout
cancel()          -> status=cancelled

Orphan recovery (reconcile): running -> pending if not in orchestrator's active set
Checkpoint pruning: pareto_rank >= 2 -> checkpoint deleted (node retained in DAG)
```

---

## 11. Comparison with autoMIL

| Aspect | autoMIL | autoTune |
|--------|---------|----------|
| Topology | Tree (one parent) | DAG (multiple parents via merge) |
| Exploration | Axis-aligned (one technique at a time) | Manifold-aligned (correlated moves along PCs) |
| Keep/discard | Binary parent-dominance | Layered Pareto ranking (NSGA-II) |
| Forgetting | Not modeled | First-class objective with cliff detection |
| Checkpoints | Not managed | Managed lifecycle (store, prune, merge) |
| Numerical guidance | None (agent picks everything) | Manifold suggests theta, agent decides strategy |
| Move types | Branch only | Branch, merge, graft |
| Search space | Implicit (whatever agent changes) | Explicit (parameters section in config) |
| Knowledge reuse | Code overlay only | Code overlay + checkpoint inheritance |
| Optimizer | None (pure UCB) | CMA-ES for correlations, NSGA-III for Pareto |

---

## 12. Dependencies

```
# Core
click           # CLI framework
pyyaml          # config parsing
jinja2          # template rendering
numpy           # covariance math, eigendecomposition

# Optuna (substrate)
optuna          # CMA-ES sampler, NSGA-III, non-dominated sorting

# Checkpoint operations
torch           # state_dict load/save
safetensors     # safetensors format support (optional)

# Visualization
aiohttp         # SSE server
watchdog        # filesystem monitoring

# Optional
peft            # LoRA adapter composition (for compose merge strategy)
```
