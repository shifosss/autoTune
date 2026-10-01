@AGENTS.md

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Project agents

Select the subagent type and model that matches the task. When multiple could apply, prefer the more specific type. Use the model column as a recommendation — override by your judgment when warranted.

| Task                                  | Subagent Type                     | Model    | Notes                                                    |
| ------------------------------------- | --------------------------------- | -------- | -------------------------------------------------------- |
| Quick file/symbol lookup              | `explorer`                        | `haiku`  | Read-only. "Where is X defined?"                         |
| Deep multi-round codebase exploration | `Explore`                         | `haiku`  | Broader than `explorer`, use for unfamiliar areas        |
| Architecture design, system tradeoffs | `architect`                       | `opus`   | ML pipelines, multi-service, cross-cutting concerns      |
| Step-by-step implementation planning  | `Plan`                            | `opus`   | Critical files, dependencies, architectural tradeoffs    |
| Feature implementation, refactoring   | `implementer`                     | `sonnet` | General-purpose coding workhorse                         |
| Standard code review                  | `code-reviewer`                   | `sonnet` | Style, error handling, test coverage, type safety        |
| Milestone code review (against plan)  | `superpowers:code-reviewer`       | `sonnet` | After completing a major plan step                       |
| Deep code review (bugs, security)     | `feature-dev:code-reviewer`       | `sonnet` | Logic errors, vulnerabilities, confidence-filtered       |
| Trace execution paths, map layers     | `feature-dev:code-explorer`       | `sonnet` | Understand existing features before new development      |
| Design feature architecture           | `feature-dev:code-architect`      | `opus`   | Implementation blueprints from existing patterns         |
| Simplify/refine changed code          | `code-simplifier:code-simplifier` | `haiku`  | Clarity and maintainability pass on recent changes       |
| Write or expand tests                 | `test-writer`                     | `sonnet` | Unit, integration, fixtures for ML pipelines             |
| Debug subtle/multi-component bugs     | `debugger`                        | `opus`   | Numerical instability, race conditions, shape mismatches |
| Security & correctness audit          | `security-auditor`                | `opus`   | Credentials, user data, prompt injection, pre-deploy     |
| Dependency audit                      | `dep-checker`                     | `haiku`  | Version conflicts, torch/transformers compatibility      |
| Documentation generation              | `doc-writer`                      | `haiku`  | README, docstrings, API docs, changelogs, ADRs           |
| Log/output analysis                   | `log-parser`                      | `haiku`  | Training logs, error logs, CI/CD build outputs           |
| Eval framework & metrics design       | `eval-designer`                   | `sonnet` | Experiment configs, benchmarks, agent scoring            |
| Scaffolding & boilerplate             | `scaffolder`                      | `haiku`  | Project structure, config templates, Dockerfiles         |
| LLM prompt design & debugging         | `prompt-engineer`                 | `sonnet` | Prompt templates, few-shot selection, format specs       |

### Parallel pairings

- **Algorithm/core agent** (`implementer`): DAG operations, manifold math, forgetting detection, Pareto ranking
- **Infrastructure agent** (`implementer`): CLI commands, orchestrator, runner, git worktree ops
- **Evaluation agent** (`eval-designer`): Metrics, experiment tracking, benchmark suites
- **Viz/frontend agent** (`implementer`): SSE server, dashboard components, 3D graph rendering

Each agent owns their domain. No file overlap.

## Codex Cooperation

Use model `gpt-6.1-sol` when dispatching jobs to codex via the user-scope `codex` MCP (`mcp__codex__codex`).
