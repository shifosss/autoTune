# autoTune Workflow & Mathematical Reference

Complete system reference covering the experiment lifecycle, all CLI commands, algorithms with mathematical detail, data structures, and runtime file layout.

For interactive flowcharts and visualizations, see [workflow-flowchart.html](workflow-flowchart.html).

---

## Table of Contents

- [1. System Overview](#1-system-overview)
- [2. CLI Command Reference](#2-cli-command-reference)
- [3. DAGNode State Machine](#3-dagnode-state-machine)
- [4. Orchestrator](#4-orchestrator)
  - [4.1 Main Loop](#41-main-loop)
  - [4.2 GPU Scheduling](#42-gpu-scheduling)
  - [4.3 Experiment Launch](#43-experiment-launch)
  - [4.4 Completion Handling](#44-completion-handling)
  - [4.5 Post-Completion Intelligence](#45-post-completion-intelligence)
  - [4.6 Orphan Recovery](#46-orphan-recovery)
- [5. Strategy Navigator — Suggestion Pipeline](#5-strategy-navigator--suggestion-pipeline)
- [6. NSGA-II Pareto Ranking](#6-nsga-ii-pareto-ranking)
- [7. DAG-UCB Potential Scoring](#7-dag-ucb-potential-scoring)
- [8. CMA-ES Manifold Learning](#8-cma-es-manifold-learning)
  - [8.1 Covariance Update](#81-covariance-update)
  - [8.2 Proposal Generation](#82-proposal-generation)
  - [8.3 Projection & Parameter Importance](#83-projection--parameter-importance)
- [9. Forgetting Detection](#9-forgetting-detection)
- [10. Branch & Merge Scoring](#10-branch--merge-scoring)
- [11. Checkpoint Operations](#11-checkpoint-operations)
- [12. Data Structures](#12-data-structures)
- [13. File System Layout](#13-file-system-layout)
- [14. Environment Variable Contract](#14-environment-variable-contract)
- [15. Module Dependencies](#15-module-dependencies)

---

## 1. System Overview

autoTune is an autonomous experiment framework for neural network finetuning. It overlays an `autotune/` directory onto existing ML repos, enabling coding agents to iteratively improve models through:

1. **DAG-structured checkpoint management** — experiments form a DAG (not flat trials). Merge operations combine multiple parents.
2. **Manifold-based hyperparameter navigation** — CMA-ES learns parameter correlations; proposals move along principal components.
3. **Forgetting-aware Pareto optimization** — 3-objective (task, retention, cost) with cliff detection and danger zones on the manifold.

### Experiment Lifecycle

```
autotune init
  → Agent writes training script
  → autotune submit --theta {...}          (seed experiments, 2-3 baselines)
  → autotune suggest --n 5                 (get ranked proposals)
  → Agent analyzes frontier + manifold + technique stats
  → Agent forms hypothesis, edits training code
  → autotune propose --parents ... --move ... --theta {...}
  → autotune submit --node <id>            (queue for execution)
  → Orchestrator: GPU scheduling → launch → training in worktree → completion
  → Post-completion intelligence: Pareto rerank + manifold update + forgetting check
  → Loop: suggest → propose → execute → intelligence update
  → autotune frontier                      (final Pareto front)
  → autotune checkpoints --prune           (cleanup dominated)
```

---

## 2. CLI Command Reference

All 15 Click CLI entry points.

| Command | Category | Triggers | Writes |
|---------|----------|----------|--------|
| `autotune init` | Setup | Creates autotune/ overlay with templates | config.yaml, dag.json, manifold_state.json, forgetting.json |
| `autotune submit --theta {}` | Creation | DAG.add_seed() + rerank + recalculate | dag.json |
| `autotune propose --parents ...` | Creation | DAG.add_proposed() + recalculate_potential() | dag.json |
| `autotune submit --node id` | Creation | Writes experiment spec to queue | queue/\<id\>.json |
| `autotune suggest --n 5` | Intelligence | StrategyNavigator.suggest_moves() | (stdout) |
| `autotune explain` | Intelligence | explain_manifold() + explain_frontier() + recommend_strategy() | (stdout) |
| `autotune frontier` | Intelligence | DAG.pareto_front() sorted by composite | (stdout) |
| `autotune rank --n 10` | Intelligence | DAG.rank_proposals(n) | (stdout) |
| `autotune status` | Intelligence | Reads queue/running/completed counters | (stdout) |
| `autotune check` | Maintenance | Validate config, DAG integrity, orphans | (stdout) |
| `autotune reconcile` | Maintenance | Process completed/*.json, rerank, recalculate | dag.json |
| `autotune checkpoints [--prune]` | Maintenance | List/delete checkpoints by Pareto rank | checkpoint_store/ |
| `autotune orchestrator start\|stop\|status` | Daemon | Spawn/terminate ExperimentOrchestrator | orchestrator.pid, gpu_state.json |
| `autotune viz start\|stop\|status` | Daemon | Spawn/terminate aiohttp SSE dashboard | viz_server.pid |

---

## 3. DAGNode State Machine

### State Transitions

```
[initial] ─── add_seed() ──────────────────────────────────→ EXECUTED (keep, rank 0)

[initial] ─── add_proposed() ──→ PROPOSED (pending)
                                    │
                      mark_running()│            cancel()
                                    ▼                │
                                 RUNNING              ▼
                                 │  │  │         CANCELLED
                                 │  │  │
               promote(metrics) ─┘  │  └─ mark_failed(status)
               pareto_rank ≤ 1      │     status ∈ {oom, crash, timeout}
                    │                │              │
                    ▼                │              ▼
             EXECUTED (keep)         │       EXECUTED (failed)
                    │                │       metrics = None
                    │     promote(metrics)
                    │     pareto_rank ≥ 2
                    │                │
                    │                ▼
                    │       EXECUTED (discard)
                    │                │
                    └──── rerank_pareto() ────┘
                         (bidirectional:
                          keep ↔ discard)
```

### Checkpoint Lifecycle by Status

| Status | Checkpoint | In DAG |
|--------|-----------|--------|
| `keep` (rank ≤ 1) | Retained in checkpoint_store/ | Yes |
| `discard` (rank ≥ 2) | Pruned by _prune_checkpoints() | Yes (metrics preserved) |
| `oom / crash / timeout` | Never collected | Yes (metrics = None) |
| `cancelled` | Never created | Yes |

---

## 4. Orchestrator

### 4.1 Main Loop

`ExperimentOrchestrator.run()` is a daemon that executes queued experiments.

**Startup sequence:**
1. `runner.prune_stale_worktrees()` — clean leftover git worktrees
2. `_recover_orphans()` — crash recovery for running/*.json leftovers
3. Write `orchestrator.pid`
4. Enter tick loop

**`tick()` — runs every `poll_interval` seconds (default 30):**
1. `_check_running()` — poll all `subprocess.Popen` processes
2. For each finished process → `_handle_completion(node_id, returncode)`
3. For each timed-out process → `process.terminate()`, mark `_timed_out[id] = True`
4. If not draining: read `_get_pending()` (sorted queue/*.json), attempt launch for each
5. `_save_state()` → write gpu_state.json

**Shutdown:** Set `draining = True`, wait for all running processes, delete PID, save state.

### 4.2 GPU Scheduling

Best-fit bin packing algorithm.

For each GPU `g` with current allocations `A_g`:

```
free_g = vram_g - safety_margin - sum(est_vram(n) for n in A_g)
```

Select the GPU with the **smallest** `free_g >= needed_gb` (tightest fit).

Constraint: `|A_g| < max_per_gpu` (default 2 concurrent experiments per GPU).

A `_pre_launch_check()` re-queries `nvidia-smi` immediately before launch to confirm the GPU still has capacity.

### 4.3 Experiment Launch

`_launch(spec, gpu_id)` — comprehensive launch sequence:

1. Create `archive/<id>/`, save spec.json
2. Delete `queue/<id>.json` (prevent retry on crash)
3. `base_commit = git rev-parse HEAD`
4. `runner.create_worktree(base_commit, id)` → `git worktree add --detach`
5. **Checkpoint preparation by move type:**
   - **seed / branch:** `runner.prepare_branch()` — copy `checkpoint_store/<parent>/` → `wt/.parent_checkpoint/`
   - **merge:** `runner.merge_checkpoints()` — weighted_avg or compose strategy (see [Checkpoint Operations](#11-checkpoint-operations))
   - **graft:** `runner.graft_checkpoint()` — delta = source - source_parent, grafted = target + delta
6. Build environment variables (see [Environment Variable Contract](#14-environment-variable-contract))
7. `subprocess.Popen(cmd, cwd=wt_path, env=env, stdout/stderr→log_file)`
8. Create `RunningExperiment` record with timeout_at = `time.time() + timeout_hours * 3600`
9. Write `running/<id>.json` (orphan recovery sentinel)
10. `dag.mark_running(id)` — status: pending → running

### 4.4 Completion Handling

`_handle_completion(node_id, returncode)`:

1. Pop from `self.running`, remove GPU allocation, close log file
2. `runner.collect_result(wt, archive)` — copy + parse result.json
3. **Determine status:**
   - `result.status == "completed"` → completed
   - `"CUDA out of memory"` in last 2000 bytes of log → oom
   - `_timed_out[id]` → timeout
   - `returncode != 0` → crash
4. Write `completed/<id>.json` (metrics, status, elapsed, gpu)
5. If checkpoint was saved: `runner.collect_checkpoint()` — copy `wt/.checkpoint_output/` → `checkpoint_store/<id>/`
6. Delete `running/<id>.json`
7. `runner.cleanup_worktree(wt_path)`
8. Run post-completion intelligence (see below)
9. `_append_results_tsv(completion)`

### 4.5 Post-Completion Intelligence

After every experiment completion (inside try/except):

1. Load DAG
2. If completed: `dag.promote(id, metrics)` — type → executed, set scores
3. If failed: `dag.mark_failed(id, status, error)` — metrics remain None
4. `dag.rerank_pareto()` — NSGA-II non-dominated sorting
5. `dag.recalculate_potential()` — DAG-UCB for all executed nodes
6. If `len(executed) >= 10`: `manifold.update(thetas, objectives)` — CMA-ES covariance learning, save to manifold_state.json
7. If completed AND retention_score is not None: `forgetting.check_cliff(node, parents)` — if delta < threshold, create DangerZone, save to forgetting.json
8. `_prune_checkpoints()` — delete checkpoints where pareto_rank >= retention_layers (default 2)
9. `dag.save()`

### 4.6 Orphan Recovery

On startup, `_recover_orphans()` handles experiments that were running when the orchestrator crashed:

For each `running/<id>.json`:
- Write completion with status = "crash", error = "Orchestrator restarted while running"
- Clean up worktree
- Delete sentinel file

---

## 5. Strategy Navigator — Suggestion Pipeline

`StrategyNavigator.suggest_moves(n=5)` generates and ranks candidate next experiments.

**Input:** DAG state, manifold state, forgetting state.

**Process:**

1. Get `kept = dag.kept_nodes()` (pareto_rank ≤ 1)
2. **Generate BRANCH proposals** — one per kept node:
   - `theta_new = manifold.propose_direction(node.theta)`
   - `coords = manifold.project(theta_new)`
   - If `forgetting.is_in_danger_zone_coords(coords)` → skip
   - `score = branch_score(node)` (see [Branch & Merge Scoring](#10-branch--merge-scoring))
   - Add `MoveProposal(move_type="branch", parent_ids=(node.id,), theta=theta_new)`
3. **Generate MERGE proposals** — one per diverse kept-node pair:
   - `diversity = cosine_distance(a.manifold_coords, b.manifold_coords)`
   - If `diversity < 0.1` → skip (not diverse enough)
   - `theta_merged = 0.5 * a.theta + 0.5 * b.theta`
   - If `forgetting.is_in_danger_zone_coords(manifold.project(theta_merged))` → skip
   - `score = merge_score(a, b, diversity)` (see [Branch & Merge Scoring](#10-branch--merge-scoring))
   - Add `MoveProposal(move_type="merge", parent_ids=(a.id, b.id), merge_strategy="weighted_avg")`
4. Sort all proposals by score descending
5. Return top `n`

---

## 6. NSGA-II Pareto Ranking

`dag.rerank_pareto()` — stratifies executed nodes into non-dominated layers.

### Objective Space

Each executed node `n` is evaluated on up to three objectives:

```
f(n) = (task(n), retention(n), -cost(n))     if retention enabled
f(n) = (task(n), -cost(n))                   otherwise
```

All objectives are **maximized** (cost is negated).

### Domination

Node A **dominates** node B (written A ≻ B) if and only if:

```
A ≻ B  ⟺  (∀ i: f_i(A) ≥ f_i(B))  ∧  (∃ j: f_j(A) > f_j(B))
```

A is at least as good as B on every objective, and strictly better on at least one.

### Non-Dominated Sorting Algorithm

```
S₀ ← {all executed nodes with metrics}
r  ← 0
while S_r ≠ ∅:
    F_r ← {n ∈ S_r | ¬∃ m ∈ S_r : m ≻ n}     (non-dominated front)
    ∀ n ∈ F_r:  n.pareto_rank ← r
    S_{r+1} ← S_r \ F_r
    r ← r + 1
```

### Status Assignment

```
n.status = keep      if n.pareto_rank ≤ 1
n.status = discard   if n.pareto_rank ≥ 2
```

- Rank 0 = Pareto-optimal
- Rank 1 = near-front (backup, checkpoints retained)
- Rank 2+ = dominated (checkpoints pruned)

The retention threshold (default 2) is configurable via `orchestrator.checkpoint_retention_layers`.

---

## 7. DAG-UCB Potential Scoring

`dag.recalculate_potential()` — UCB1-inspired scoring that balances exploitation and exploration over a DAG structure.

### Core Equation

```
potential(n) = (c(n) + w_e * sqrt(ln(N) / (1 + k(n)))) * h(n)
```

Where:
- `c(n)` = `n.composite` (or 0 if None) — **exploitation** term
- `w_e` = `scoring.exploration_weight` — hyperparameter
- `N` = `total_executed` — number of completed experiments
- `k(n)` = `|children(n)|` — direct child count in the DAG — **exploration bonus** (fewer children → larger bonus)
- `h(n)` = retention health — **safety** multiplier

### Retention Health Multiplier

```
h(n) = 1.0                            if n.retention_score is None
h(n) = n.retention_score              if n.retention_score ≥ |τ|
h(n) = n.retention_score * 0.5        if n.retention_score < |τ|
```

Where `τ = scoring.retention_cliff_threshold` (e.g., -0.10).

Nodes below the cliff threshold have their potential **halved**, steering exploration away from regions that cause forgetting.

### Intuition

- **Exploitation:** High composite score `c(n)` gives a high base potential.
- **Exploration:** Nodes with fewer children get a larger bonus. The `ln(N) / (1 + k)` term grows as `k` decreases, preventing over-exploitation of a single branch.
- **Safety:** The health multiplier penalizes nodes near retention cliffs, pushing the search away from catastrophic forgetting.

---

## 8. CMA-ES Manifold Learning

`ManifoldNavigator.update()` — learns parameter correlations via covariance matrix adaptation. Triggered after every completion when `|executed| >= 10`.

### 8.1 Covariance Update

**Step 1: Normalization**

All hyperparameter vectors θ ∈ ℝᵈ are normalized to [0,1]ᵈ:

```
θ̂_i = (θ_i - lo_i) / (hi_i - lo_i)    for each parameter i
```

Where `(lo_i, hi_i)` are the configured bounds from config.yaml.

**Step 2: Elite Selection & Weighting**

Rank all N evaluated points by composite score. Take the top μ = ⌊N/2⌋ (elite half).

Log-decay weights:

```
w_i = ln(μ + 0.5) - ln(i)    for i = 1, ..., μ
Normalize: w_i ← w_i / Σ w_j   so that Σ w_i = 1
```

Higher-ranked points get exponentially more weight.

**Step 3: Mean Update**

```
μ_new = Σ_{i=1}^{μ} w_i * θ̂_i
```

The centroid moves toward the best-performing region.

**Step 4: Covariance Update**

Compute weighted sample covariance from deviations relative to the *old* mean:

```
C_sample = Σ_{i=1}^{μ} w_i * (θ̂_i - μ_old)(θ̂_i - μ_old)ᵀ
```

Blend with running estimate using learning rate α = 0.3:

```
C_new = (1 - α) * C_old + α * C_sample
```

The exponential moving average prevents the covariance from changing too rapidly on noisy objectives.

**Step 5: Eigendecomposition**

```
C_new = V Λ Vᵀ
```

Where Λ = diag(λ₁, λ₂, ..., λ_d) with λ₁ ≥ λ₂ ≥ ... ≥ λ_d.

Columns of V are the **principal components** (eigenvectors), sorted by variance contribution.

**Step 6: Effective Dimensionality**

```
d_eff = min{k : (Σ_{j=1}^{k} λ_j) / (Σ_{j=1}^{d} λ_j) ≥ 0.95}
```

Only the top d_eff PCs capture meaningful variance. Proposals and projections use this reduced subspace.

**Step 7: Step Size Adaptation**

```
s_j^{t+1} = s_j^t * 1.2    if λ_j^{t+1} > λ_j^t    (variance grew — productive direction)
s_j^{t+1} = s_j^t * 0.8    if λ_j^{t+1} ≤ λ_j^t    (variance shrank — over-explored)
```

### 8.2 Proposal Generation

**`propose_direction(from_theta, pc_index)`** — step along a principal component:

If untrained (generation = 0): random unit vector with step 0.1.

If trained:
1. Sample PC index j proportional to eigenvalue magnitude (if not specified): `P(j) = λ_j / Σ_k λ_k`
2. Step along the chosen PC:
   ```
   θ̂_new = clip(normalize(θ_from) + s_j * v_j, 0, 1)
   ```
3. Denormalize back to original parameter space.

**`propose_candidates(from_theta, n)`** — Gaussian sampling:

If untrained: n uniform samples from [0,1]ᵈ.

If trained:
```
θ̂_i ~ N(normalize(θ_from), 0.01 * C)    for i = 1, ..., n
```
Clip to [0,1]ᵈ, then denormalize.

### 8.3 Projection & Parameter Importance

**`project(theta)`** — manifold coordinates:

```
z = V_{:d_eff}ᵀ * (θ̂ - μ)
```

Projects a d-dimensional parameter vector onto the d_eff-dimensional manifold subspace. Used for danger zone checks and diversity computation.

**`parameter_importance()`** — variance contribution per parameter:

```
importance(i) = Σ_j λ_j * |V_{ij}|  /  Σ_{i'} Σ_j λ_j * |V_{i'j}|
```

Sums each parameter's absolute loading across all PCs, weighted by eigenvalue. Normalized to sum to 1.

---

## 9. Forgetting Detection

`ForgettingDetector` — retention cliff detection and manifold danger zone tracking. Inactive if no retention eval is configured.

### Cliff Detection: `check_cliff(node, parents)`

Compare node's retention against the best parent:

```
r* = max(retention(p) for p in parents if p has retention)
δ  = retention(n) - r*
```

A cliff is detected when:

```
δ < τ    (τ = cliff_threshold, e.g., -0.10)
```

### Danger Zone Creation

When a cliff is detected at node n:

```
DangerZone = (
    center    = project(θ_n),          # manifold coordinates
    radius    = |δ| * s,               # s = 0.5 (step size multiplier)
    cause     = n.technique_tags[0],   # technique that caused forgetting
    severity  = |δ|                    # magnitude of retention drop
)
```

A larger retention drop creates a larger exclusion zone.

### Zone Check: `is_in_danger_zone_coords(coords)`

For a candidate at manifold coordinates z:

```
in_danger = ∃ dz ∈ D : ‖z - z_dz‖₂ ≤ r_dz
```

Where D is the accumulated set of danger zones. Uses truncated dimensions: `min(dim(z), dim(z_dz))`.

If any danger zone contains the candidate, the proposal is **skipped** by `suggest_moves()`.

---

## 10. Branch & Merge Scoring

Scoring formulas used by `StrategyNavigator.suggest_moves()` to rank proposals.

### Branch Scoring (Single Parent)

```
score_branch(n) = (c(n) + w_e * sqrt(ln(N) / (1 + k(n)))) * h(n)
```

Identical to DAG-UCB potential. Variables:
- `c(n)` = parent's composite score (or 0)
- `w_e` = `scoring.exploration_weight`
- `N` = total executed experiments
- `k(n)` = number of children of the parent node
- `h(n)` = retention health (see [DAG-UCB](#7-dag-ucb-potential-scoring))

### Merge Scoring (Two Parents)

```
score_merge(A, B) = (c̄ + w_e * sqrt(ln(N) / (1 + m_AB)) + w_d * div(A,B)) * κ(A, B)
```

Where:
- `c̄ = (c(A) + c(B)) / 2`
- `m_AB` = count of prior merges between A and B
- `div(A,B) = 1 - (z_A · z_B) / (‖z_A‖ * ‖z_B‖)` — cosine distance in manifold space

### Complementarity Bonus

```
κ(A, B) = 1.0 + |task(A) - task(B)| * |ret(A) - ret(B)|    if both have task and retention
κ(A, B) = 1.0                                                otherwise
```

Pairs where one parent is strong on task and the other on retention get a complementarity boost.

### Diversity Gate

```
div(A, B) < 0.1  →  skip merge proposal
```

Prevents proposing merges between near-identical parents.

---

## 11. Checkpoint Operations

Four checkpoint strategies for different move types.

### Branch (Copy)

```
θ_child ← θ_parent
```

Direct copy of the parent checkpoint. The training script starts from this base.

### Weighted Average Merge

Given parents P₁, P₂, ..., P_m with weights w₁, w₂, ..., w_m where Σ w_i = 1:

```
θ_merged[k] = Σ_{i=1}^{m} w_i * θ_{P_i}[k]    ∀ k ∈ ⋂_i keys(θ_{P_i})
```

Only keys present in **all** parent state dicts are merged. Missing keys are dropped.

### Compose Merge (LoRA)

Given base P₀ and additional parents P₁, ..., P_{m-1}:

```
Δ = Σ_{i=1}^{m-1} (θ_{P_i}[k] - θ_{P₀}[k])
θ_composed[k] = θ_{P₀}[k] + Δ
```

Sums the deltas of all parents relative to a shared base, then applies the total delta. Useful for composing independent LoRA adaptations.

### Graft (Cross-Branch Technique Transfer)

Given source experiment S with source parent S_p, and target parent T_p:

```
Δ_S[k] = θ_S[k] - θ_{S_p}[k]              (technique delta)
θ_grafted[k] = θ_{T_p}[k] + Δ_S[k]        (apply to different branch)
```

For keys not in S or S_p, the target parent's value is kept unchanged.

### State Dict Loading (Framework-Agnostic)

When loading from a directory, `_load_state_dict()` searches in order:

1. `adapter_model.safetensors` (PEFT/LoRA)
2. `pytorch_model.bin` (PyTorch)
3. `model.safetensors` (SafeTensors)

File format detected by extension: `.safetensors` → `safetensors.torch.load_file()`, else `torch.load(weights_only=True)`.

---

## 12. Data Structures

All types in `types.py` are `@dataclass(frozen=True)` for immutability.

### DAGNode

Immutable experiment node — the fundamental unit of the experiment DAG.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | "node_0001", sequential |
| `parent_ids` | `tuple[str, ...]` | () for seed, (p,) for branch, (p1,p2) for merge |
| `move_type` | `str` | "seed" \| "branch" \| "merge" \| "graft" |
| `theta` | `dict[str, float]` | Hyperparameters, e.g., {"lr": 2e-4, "rank": 16} |
| `manifold_coords` | `tuple[float, ...] \| None` | PC projection (computed on complete) |
| `strategy_desc` | `str` | Agent's rationale for this experiment |
| `technique_tags` | `tuple[str, ...]` | e.g., ("lr_increase", "rank_boost") |
| `task_score` | `float \| None` | Primary task metric (None while pending) |
| `retention_score` | `float \| None` | Forgetting resistance (None if not configured) |
| `cost` | `float \| None` | Compute cost metric |
| `composite` | `float \| None` | Scalar combining all objectives |
| `checkpoint_id` | `str \| None` | Reference to checkpoint_store/\<id\>/ |
| `depth` | `int` | Longest path from any root |
| `potential` | `float` | DAG-UCB exploration score |
| `pareto_rank` | `int \| None` | 0 = optimal, 1 = near-front, 2+ = dominated |
| `node_type` | `str` | "executed" \| "proposed" |
| `status` | `str` | pending \| running \| keep \| discard \| oom \| crash \| timeout \| cancelled |
| `created_at` | `str` | ISO-8601 timestamp |

### DAGMeta

| Field | Type | Description |
|-------|------|-------------|
| `next_id` | `int` | Counter for sequential node IDs |
| `total_executed` | `int` | Count of completed experiments |
| `total_proposed` | `int` | Count of proposed experiments |
| `best_composite` | `float` | Highest composite score seen |
| `scoring` | `ScoringConfig` | UCB hyperparameters |

### ScoringConfig

| Field | Type | Description |
|-------|------|-------------|
| `exploration_weight` | `float` | w_e in UCB formula |
| `diversity_weight` | `float` | w_d in merge scoring |
| `retention_cliff_threshold` | `float` | τ for cliff detection (e.g., -0.10) |

### ManifoldState

| Field | Type | Description |
|-------|------|-------------|
| `param_names` | `tuple[str, ...]` | Ordered parameter names |
| `covariance` | `np.ndarray (d×d)` | Learned covariance matrix in [0,1]^d |
| `eigenvalues` | `np.ndarray (d,)` | Sorted descending |
| `eigenvectors` | `np.ndarray (d×d)` | Sorted by eigenvalue |
| `mean` | `np.ndarray (d,)` | Centroid of elite points in [0,1]^d |
| `step_sizes` | `np.ndarray (d,)` | Adapted step per PC |
| `effective_dim` | `int` | PCs capturing ≥95% variance |
| `generation` | `int` | CMA-ES update count |

### DangerZone

| Field | Type | Description |
|-------|------|-------------|
| `center` | `tuple[float, ...]` | Manifold coordinates of the cliff |
| `radius` | `float` | Exclusion radius (severity × step_size) |
| `cause_technique` | `str` | Technique tag that caused forgetting |
| `severity` | `float` | Magnitude of retention drop |

### CliffReport

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Node that triggered the cliff |
| `delta` | `float` | Retention change (negative) |
| `severity` | `float` | \|delta\| |
| `danger_zone` | `DangerZone` | The created exclusion zone |

### MoveProposal

| Field | Type | Description |
|-------|------|-------------|
| `move_type` | `str` | "branch" \| "merge" \| "graft" |
| `parent_ids` | `tuple[str, ...]` | Parent(s) to move from |
| `suggested_theta` | `dict[str, float]` | Proposed hyperparameters |
| `merge_strategy` | `str \| None` | Merge algorithm name |
| `merge_weights` | `tuple[float, ...] \| None` | Per-parent weights |
| `direction_label` | `str` | e.g., "PC1 (+capacity)" |
| `score` | `float` | Estimated composite score |
| `reasoning` | `str` | Explanation of proposal |

### TechniqueStats

| Field | Type | Description |
|-------|------|-------------|
| `times_tried` | `int` | Number of nodes using this technique |
| `avg_task_delta` | `float` | Mean task score improvement vs parent |
| `avg_retention_delta` | `float` | Mean retention change (tracks forgetting risk) |
| `best_parent_delta` | `float` | Best single-step composite improvement |

---

## 13. File System Layout

```
project_root/
├── autotune/                              ← overlay directory
│   ├── config.yaml                        ← project configuration
│   ├── dag.json                           ← experiment DAG state
│   ├── manifold_state.json                ← CMA-ES covariance + PCs
│   ├── forgetting.json                    ← danger zones + cliff threshold
│   ├── results.tsv                        ← flat metrics table (all experiments)
│   │
│   ├── orchestrator/
│   │   ├── orchestrator.pid               ← daemon PID
│   │   ├── gpu_state.json                 ← current GPU allocations + queue
│   │   ├── queue/                         ← pending experiments
│   │   │   └── node_XXXX.json
│   │   ├── running/                       ← sentinel files (orphan recovery)
│   │   │   └── node_XXXX.json
│   │   ├── completed/                     ← result notifications
│   │   │   └── node_XXXX.json
│   │   ├── archive/                       ← per-experiment records
│   │   │   └── node_XXXX/
│   │   │       ├── spec.json
│   │   │       ├── run.log
│   │   │       └── result.json
│   │   └── checkpoints/                   ← persistent checkpoint store
│   │       └── node_XXXX/
│   │           └── pytorch_model.bin      (or .safetensors)
│   │
│   └── viz/
│       └── viz_server.pid                 ← dashboard PID
│
└── .autotune_worktrees/                   ← isolated training worktrees
    └── node_XXXX/                         ← git worktree per experiment
        ├── (project files at detached HEAD)
        ├── .parent_checkpoint/            ← prepared parent checkpoint
        ├── .checkpoint_output/            ← training saves here
        └── result.json                    ← training writes here
```

---

## 14. Environment Variable Contract

Variables set by the orchestrator that training scripts must read.

| Variable | Set For | Value | Required by Script |
|----------|---------|-------|--------------------|
| `CUDA_VISIBLE_DEVICES` | All | GPU index(es) | No (auto) |
| `AUTOTUNE_GPU` | All | 0 (logical after masking) | Optional |
| `AUTOTUNE_NODE_ID` | All | node_XXXX | Optional (logging) |
| `AUTOTUNE_CHECKPOINT_DIR` | All | wt/.checkpoint_output/ | **YES — save here** |
| `AUTOTUNE_PARENT_CHECKPOINT` | branch/merge/graft | wt/.parent_checkpoint/ | **YES — load from here** |
| `AUTOTUNE_MOVE_TYPE` | All | seed \| branch \| merge \| graft | Optional |
| `AUTOTUNE_RETENTION_SCRIPT` | If retention enabled | Path to eval script | Optional |
| `AUTOTUNE_RETENTION_PROBE` | If retention enabled | Path to probe set | Optional |

### Training Script Contract

Every training script **must**:

1. Load parent checkpoint from `$AUTOTUNE_PARENT_CHECKPOINT`
2. Save output checkpoint to `$AUTOTUNE_CHECKPOINT_DIR`
3. Write `result.json` to working directory. Both flat and nested formats are accepted:
   ```json
   {"status": "completed", "task_score": 0.85, "cost": 1.5, "checkpoint_saved": true}
   ```
   ```json
   {"status": "completed", "metrics": {"task_score": 0.85, "cost": 1.5}, "composite": 0.85, "checkpoint_saved": true}
   ```
   If `composite` is omitted, it defaults to `task_score`.

---

## 15. Module Dependencies

```
Layer            Module              Dependencies
─────────────────────────────────────────────────────────────
Entry Point      cli.py              dag + navigator + orchestrator + viz

Bridge           navigator.py        dag + manifold + forgetting
Orchestration    orchestrator.py     dag + manifold + forgetting + runner

Standalone       dag.py              types only
                 manifold.py         types only
                 forgetting.py       types only
                 runner.py           types only (+ git + torch/safetensors)
                 viz/server.py       filesystem only (reads dag.json, manifold_state.json)

Foundation       types.py            (no dependencies)
```

**Dependency direction:** cli → navigator/orchestrator → standalone core → types.

The standalone modules (dag, manifold, forgetting, runner) have no internal cross-dependencies. The navigator bridges dag + manifold + forgetting. The orchestrator combines runner + manifold + forgetting + dag. The viz server is decoupled entirely, communicating via filesystem watch on JSON files.
