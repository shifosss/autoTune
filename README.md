# autoTune

Autonomous experiment framework for neural network and LLM finetuning. Overlays an `autotune/` directory onto existing ML project repos, enabling coding agents to iteratively improve finetuned models through DAG-structured experimentation, manifold-based hyperparameter navigation, and forgetting-aware multi-objective optimization.

## What Makes autoTune Novel

1. **DAG-structured checkpoint management** — experiments form a directed acyclic graph, not flat trials or a tree. Merge operations combine multiple parents, enabling knowledge recombination across branches.

2. **Manifold-based hyperparameter navigation** — CMA-ES learns parameter correlations and proposes moves along discovered principal components, navigating a learned low-dimensional surface rather than the full hyperparameter space.

3. **Forgetting-aware Pareto optimization** — every experiment is evaluated on task performance, retention (resistance to catastrophic forgetting), and cost. Forgetting cliffs are detected and marked as danger zones on the manifold.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Or sync from lockfile
uv sync --extra dev

# Optional extras for checkpoint operations
uv pip install -e ".[dev,torch,safetensors,peft]"
```

## Quick Start

```bash
# Initialize autoTune in an existing ML project repo
cd your-project/
autotune init --base-model "meta-llama/Llama-3-8B"

# Check status
autotune status

# After adding seed experiments to the DAG:
autotune suggest --n 5     # Get move suggestions from the navigator
autotune explain           # Manifold summary, Pareto frontier, strategy
autotune frontier          # Current Pareto front
autotune reconcile         # Sync DAG with completed experiments
```

## Architecture

**Sovereign Core + Optuna Substrate** — autoTune owns the DAG, state machine, agent contract, and control flow. Optuna is used only as a library for CMA-ES sampling and NSGA-III Pareto computation.

### Module Dependency Graph

```
cli.py -> dag.py, navigator.py, orchestrator.py, viz/server.py

navigator.py -> dag.py, manifold.py, forgetting.py

orchestrator.py -> runner.py, manifold.py, forgetting.py

dag.py        -> standalone
manifold.py   -> standalone (wraps Optuna CMA-ES)
forgetting.py -> standalone
runner.py     -> standalone (git + file ops)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `types.py` | Frozen dataclasses: DAGNode, DAGMeta, ScoringConfig, ManifoldState, etc. |
| `dag.py` | ExperimentDAG: node CRUD, NSGA-II Pareto ranking, DAG-UCB scoring, reconciliation |
| `manifold.py` | ManifoldNavigator: CMA-ES covariance learning, PC extraction, theta proposals |
| `forgetting.py` | ForgettingDetector: retention cliff detection, danger zone tracking |
| `navigator.py` | StrategyNavigator: combines DAG scores + manifold + forgetting for move suggestions |
| `runner.py` | ExperimentRunner: git worktree management, checkpoint branch/merge/graft |
| `cli.py` | Click CLI: init, submit, propose, suggest, explain, rank, reconcile, status, frontier |

## Agent Integration

autoTune is designed for coding agents (e.g., Claude) to drive the experiment loop:

```
1. OBSERVE   — autotune status, autotune reconcile
2. ANALYZE   — autotune explain, autotune frontier, autotune suggest
3. DECIDE    — accept, modify, or override a suggestion
4. IMPLEMENT — edit training script, config, data pipeline
5. SUBMIT    — autotune propose, autotune submit
6. REFLECT   — update learnings.md with insights
```

The agent makes strategic decisions (move type, parent selection, code changes). The manifold navigator provides numerical guidance (suggested theta along principal components). The framework handles keep/discard decisions via Pareto ranking and checkpoint lifecycle management.

## Overlay Model

autoTune drops an `autotune/` directory into existing repos. It never creates training scripts. Training scripts must:

1. Load parent checkpoint from `AUTOTUNE_PARENT_CHECKPOINT` env var
2. Save checkpoint to `AUTOTUNE_CHECKPOINT_DIR` env var
3. Write `result.json` to working directory

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/autotune --cov-report=term-missing

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Tech Stack

- **Python 3.11+**, uv, hatchling
- **CLI**: Click
- **Config/templates**: PyYAML, Jinja2
- **Math**: NumPy (covariance, eigendecomposition)
- **Optimization**: Optuna (CMA-ES sampler, NSGA-III)
- **Checkpoint I/O**: PyTorch, SafeTensors (optional), PEFT (optional)
- **Visualization**: aiohttp + watchdog (SSE server), ForceGraph3D (frontend)

## License

AGPL-3.0
