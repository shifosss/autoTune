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
