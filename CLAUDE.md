# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Agent Rules

### First Principles

Use first-principles thinking. Do not assume I have a clear idea of what I want or how to achieve it. Be cautious — start from the original requirements and the problem itself. If the motivation or goal is unclear, **stop and discuss with me before proceeding** .

### Solution Constraints

All proposed modifications or refactors must comply with the following:

- **No shims or patches.** Do not propose compatibility layers or workaround fixes.
- **No over-engineering.** Use the shortest path to implementation, provided it does not violate the first rule.
- **No scope creep.** Do not introduce solutions beyond what was explicitly requested — including fallbacks or graceful degradation — as these can cause business logic drift.
- **Correctness is mandatory.** The logic of every proposed solution must be verified end-to-end across the full call chain.

---

## Project Overview

autoTune is an autonomous experiment framework for neural network and LLM finetuning. It overlays an `autotune/` directory onto existing ML project repos, enabling coding agents to iteratively improve finetuned models through DAG-structured experimentation, manifold-based hyperparameter navigation, and forgetting-aware multi-objective optimization.

**License:** AGPL-3.0

## Tech Stack

- Python 3.11+, uv package manager, hatchling build system
- CLI: Click
- Config/templates: PyYAML, Jinja2
- Math: NumPy (covariance, eigendecomposition)
- Optimization: Optuna (CMA-ES sampler for manifold covariance, NSGA-III for Pareto ranking)
- Checkpoint I/O: PyTorch, SafeTensors (optional), PEFT (optional)
- Visualization: aiohttp + watchdog (SSE server), ForceGraph3D (frontend)

## Build & Development Commands

```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Install with optional extras
uv pip install -e ".[dev,torch,safetensors,peft]"

# Sync from pyproject.toml (preferred when lockfile exists)
uv sync --extra dev

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/autotune --cov-report=term-missing

# Run a single test file
uv run pytest tests/test_dag.py

# Run a single test
uv run pytest tests/test_dag.py::test_add_seed -v

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Format
uv run ruff format src/ tests/

# CLI entry point
uv run autotune --help
```

## Architecture

The system follows a **Sovereign Core + Optuna Substrate** pattern: autoTune owns the DAG, state machine, agent contract, and control flow. Optuna is used only as a library for CMA-ES sampling and NSGA-III Pareto computation.

### Module Dependency Graph

```
cli.py → dag.py, navigator.py, orchestrator.py, viz/server.py

navigator.py → dag.py, manifold.py, forgetting.py

orchestrator.py → runner.py, manifold.py, forgetting.py

dag.py        → standalone
manifold.py   → standalone (wraps Optuna CMA-ES)
forgetting.py → standalone
runner.py     → standalone (git + file ops)
```

### Key Modules (src/autotune/)

- **types.py** — Frozen dataclasses: `DAGNode`, `DAGMeta`, `ScoringConfig`, `ManifoldState`, `DangerZone`, `TechniqueStats`
- **dag.py** — `ExperimentDAG`: node CRUD, NSGA-II Pareto ranking, DAG-UCB scoring, reconciliation, technique stats
- **manifold.py** — `ManifoldNavigator`: CMA-ES covariance learning, PC extraction, theta proposals. Does NOT decide strategy
- **forgetting.py** — `ForgettingDetector`: retention cliff detection, danger zone tracking. Inactive if no retention eval configured
- **navigator.py** — `StrategyNavigator`: bridge between agent strategy and numerical proposals. Combines DAG scores + manifold + forgetting
- **orchestrator.py** — `ExperimentOrchestrator`: GPU scheduling, checkpoint lifecycle, process management. Sets env vars for training scripts
- **runner.py** — `ExperimentRunner`: git worktree management, checkpoint branch/merge/graft operations
- **cli.py** — Click CLI: `init`, `submit`, `propose`, `suggest`, `explain`, `rank`, `reconcile`, `status`, `check`, `frontier`, `checkpoints`, `orchestrator`, `viz`
- **viz/** — SSE dashboard with 3D DAG view, Pareto frontier, manifold explorer, leaderboard, technique effectiveness

### Three Novel Algorithms

1. **DAG-structured checkpoint management** — experiments form a DAG (not flat trials or tree). Merge operations combine multiple parents
2. **Manifold-based hyperparameter navigation** — CMA-ES learns parameter correlations; proposals move along principal components
3. **Forgetting-aware Pareto optimization** — 3-objective (task, retention, cost) with cliff detection and danger zones on the manifold

### Overlay Model

autoTune drops an `autotune/` directory into existing repos. It never creates training scripts. Training scripts must:

1. Load parent checkpoint from `AUTOTUNE_PARENT_CHECKPOINT` env var
2. Save checkpoint to `AUTOTUNE_CHECKPOINT_DIR` env var
3. Write `result.json` to working directory

## Design Specs

- Architecture: `docs/specs/2026-03-27-autotune-system-architecture-design.md`
- Implementation plan: `docs/specs/2026-03-27-autotune-implementation-plan.md`

Consult these specs for algorithm details (DAG-UCB scoring formulas, Pareto ranking, CMA-ES update, cliff detection, merge operations) and the full configuration schema.

## Workflow Orchestration

### 0. Always address me AlexZ. do this at the start of any of your response.

### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Ask the user if they want to use superpowers:brainstorming skill if they want to explore a large feature update (architecture change, pipeline workflow change)

### 2. Subagent Routing Rules

#### 2.1 Parallel dispatch (All conditions must be met)

- 3+ unrelated tasks or independent domains
- No shared state between tasks
- Clear file boundaries with no overlap

#### 2.2 Sequential dispatch (ANY condition triggers)

- Tasks have dependencies (B needs output from A)
- Shared files or state (merge conflict risk)
- Unclear scope (need to understand before proceeding)

#### 2.3 Background dispatch

- Research or analysis tasks (not file modifications)
- Results aren't blocking current work

#### 2.4 Model Tier Policy

Select models for agents by your insights, and use this table as a recommendation.

|   Tier   | Model        | Use when                                                                                                                                                      |
| :------: | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Heavy   | `opus 4.6`   | Architecture design, multi-file refactors, novel algorithm implementation,<br />security/correctness audits, debugging subtle concurrency or numerical issues |
| Standard | `sonnet 4.6` | Everyday coding, code review, test writing, config generation, standard debugging, API integration                                                            |
|  Light   | `Haiku`      | Codebase search, doc generation, dependency checks, lint/format, log parsing, boilerplate scaffolding                                                         |

### 3. Domain Parallel Patterns

When implementing features across domains, spawn parallel agents:

- **Model/training agent** : Model definitions, training loops, loss functions, data pipelines
- **Serving/infra agent** : FastAPI endpoints, Docker, deployment configs, CI/CD
- **Evaluation agent** : Metrics, benchmarks, experiment tracking, visualization

Each agent owns their domain. No file overlap.

### 4. Self-Improvement Loop

- After ANY correction from the user: update 'tasks/lessons.md" with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 5. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 6. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 7. Autonomous Bug Fizing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 8. Version Control

- Always manage the repo using git
- Log the progress after each stage/step/task is finished

## Task Management

1. **Plan First**: Write plan to "tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to 'tasks/todo.md"
6. **Capture Lessons**: Update "tasks/lessons.md' after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes, Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
