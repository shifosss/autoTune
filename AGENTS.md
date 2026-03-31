# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `src/autotune/`. Keep CLI wiring in `cli.py`, experiment graph logic in `dag.py`, numerical navigation in `manifold.py` and `navigator.py`, orchestration in `orchestrator.py` and `runner.py`, and dashboard code under `viz/` and `viz/static/`. Jinja templates for `autotune init` live in `src/autotune/templates/`. Tests mirror the package at `tests/test_*.py`. Longer design notes belong in `docs/specs/`, `docs/research/`, or `docs/workflow-*`.

## Build, Test, and Development Commands

Use Python 3.11+ and `uv`.

```bash
uv pip install -e ".[dev]"                    # editable install with pytest + ruff
uv run pytest                                 # full test suite
uv run pytest tests/test_dag.py               # focused test run
uv run pytest --cov=src/autotune --cov-report=term-missing
uv run ruff check src/ tests/                 # lint
uv run ruff format src/ tests/                # format
uv run autotune --help                        # verify CLI entry point
```

## Coding Style & Naming Conventions

Follow Ruff defaults configured in `pyproject.toml`: Python 3.11 target, 100-character line length, and 4-space indentation. Prefer type hints, frozen or explicit dataclass-style models in `types.py`, and small single-purpose functions. Use `snake_case` for modules, functions, variables, and test names; `PascalCase` for classes; and descriptive Click command names that match existing verbs such as `init`, `suggest`, and `reconcile`.

## Testing Guidelines

Write tests with `pytest`. Name files `tests/test_<area>.py` and test functions `test_<behavior>`. Mirror the production module you change and add regression coverage for CLI output, DAG state transitions, or viz behavior as applicable. There is no enforced coverage gate in the repo, so treat `pytest --cov=src/autotune --cov-report=term-missing` as the review baseline.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit-style subjects such as `fix(viz): ...`, `feat(viz): ...`, and `chore(viz): ...`; keep that format and scope commits to one change. PRs should explain the user-visible or algorithmic impact, list verification commands run, and link related issues. Include screenshots or terminal output when changing `src/autotune/viz/` or CLI behavior.

## Contributor Notes

Do not invent new public APIs when an existing module can own the change cleanly. Update docs when behavior or commands change, and validate changes before opening a PR.

## Workflow Orchestration

### 0. Always address me AlexZ. do this at the start of any of your response.

### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Ask the user if they want to use superpowers:brainstorming skill if they want to explore a large feature update (architecture change, pipeline workflow change)

### 2. Subagent Routing Rules

#### 2.1 Dispatch Modes

**Parallel** (all conditions must be met):

- 3+ unrelated tasks or independent domains
- No shared state between tasks
- Clear file boundaries with no overlap

**Sequential** (any condition triggers):

- Tasks have dependencies (B needs output from A)
- Shared files or state (merge conflict risk)
- Unclear scope (need to understand before proceeding)

**Background** (non-blocking):

- Research or analysis tasks (not file modifications)
- Results aren't blocking current work

**In agent teams:** Subagents may spawn their own subagents for parallel, independent tasks. The same dispatch rules above apply — parallel only when no shared state or file overlap.

### 3. Domain Parallel Patterns

When implementing features across domains, spawn parallel agents with specific subagent types:

- **Algorithm/core agent** (`implementer`): DAG operations, manifold math, forgetting detection, Pareto ranking
- **Infrastructure agent** (`implementer`): CLI commands, orchestrator, runner, git worktree ops
- **Evaluation agent** (`eval-designer`): Metrics, experiment tracking, benchmark suites
- **Viz/frontend agent** (`implementer`): SSE server, dashboard components, 3D graph rendering

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
- Create different branch and work within it instead of direct commit into main branch
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
