"""autoTune CLI entry point.

Provides commands for initializing an autoTune overlay, querying experiment
status, generating move suggestions, explaining the manifold, viewing the
Pareto frontier, and reconciling DAG state with the orchestrator filesystem.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click
import yaml
from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from autotune.forgetting import ForgettingDetector
    from autotune.manifold import ManifoldNavigator

logger = logging.getLogger(__name__)


def _find_project_root() -> Path | None:
    """Walk upward from cwd to find an autoTune project root.

    Returns:
        Path to the project root containing autotune/config.yaml, or None.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "autotune" / "config.yaml").exists():
            return parent
    return None


def _load_config(root: Path) -> dict:
    """Load the autoTune config.yaml from the project root.

    Args:
        root: Project root directory.

    Returns:
        Parsed YAML dict, or empty dict if missing.
    """
    config_path = root / "autotune" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _load_or_create_manifold(root: Path, params: dict) -> "ManifoldNavigator":
    """Load manifold state from disk or create a fresh navigator.

    Args:
        root: Project root directory.
        params: Parameter specifications from config.yaml.

    Returns:
        ManifoldNavigator instance.
    """
    from autotune.manifold import ManifoldNavigator

    manifold_path = root / "autotune" / "manifold_state.json"
    if manifold_path.exists():
        return ManifoldNavigator.load(manifold_path)
    param_names = list(params.keys())
    param_bounds: dict[str, tuple[float, float]] = {}
    for name, spec in params.items():
        r = spec.get("range", [0, 1])
        param_bounds[name] = (r[0], r[1])
    return ManifoldNavigator(param_names=param_names, param_bounds=param_bounds)


def _load_or_create_forgetting(root: Path, config: dict) -> "ForgettingDetector":
    """Load forgetting detector state from disk or create a fresh detector.

    Args:
        root: Project root directory.
        config: Parsed config.yaml dict.

    Returns:
        ForgettingDetector instance.
    """
    from autotune.forgetting import ForgettingDetector

    forgetting_path = root / "autotune" / "forgetting.json"
    if forgetting_path.exists():
        return ForgettingDetector.load(forgetting_path)
    threshold = config.get("scoring", {}).get("retention_cliff_threshold", -0.10)
    return ForgettingDetector(cliff_threshold=threshold)


@click.group()
def main() -> None:
    """autoTune: Autonomous finetuning framework."""


@main.command()
@click.option("--task", "task_type", default="instruction_tuning")
@click.option("--base-model", default="meta-llama/Llama-3-8B")
def init(task_type: str, base_model: str) -> None:
    """Initialize autoTune overlay in the current repo."""
    cwd = Path.cwd()
    if not (cwd / ".git").exists():
        raise click.ClickException("Not a git repository. Run 'git init' first.")
    if (cwd / "autotune" / "config.yaml").exists():
        raise click.ClickException("autoTune already initialized.")

    dirs = [
        "autotune",
        "autotune/orchestrator/queue",
        "autotune/orchestrator/running",
        "autotune/orchestrator/completed",
        "autotune/orchestrator/archive",
        "autotune/orchestrator/checkpoints",
    ]
    for d in dirs:
        (cwd / d).mkdir(parents=True, exist_ok=True)

    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    context = {
        "project_name": cwd.name,
        "task_type": task_type,
        "base_model": base_model,
    }
    templates = {
        "config.yaml.j2": "autotune/config.yaml",
        "program.md.j2": "autotune/program.md",
        "learnings.md.j2": "autotune/learnings.md",
        ".gitignore.j2": "autotune/.gitignore",
    }
    for tmpl_name, out_path in templates.items():
        template = env.get_template(tmpl_name)
        rendered = template.render(context)
        (cwd / out_path).write_text(rendered)

    logger.info("Initialized autoTune overlay in %s", cwd)
    click.echo("autoTune initialized.")


@main.command()
def status() -> None:
    """Show experiment status."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    orch = root / "autotune" / "orchestrator"
    queue_count = len(list((orch / "queue").glob("*.json"))) if (orch / "queue").exists() else 0
    completed_count = (
        len(list((orch / "completed").glob("*.json"))) if (orch / "completed").exists() else 0
    )
    dag_path = root / "autotune" / "dag.json"

    click.echo(f"Queue: {queue_count} pending")
    click.echo(f"Completed: {completed_count} awaiting reconcile")

    if dag_path.exists():
        from autotune.dag import ExperimentDAG

        dag = ExperimentDAG.load(dag_path)
        meta = dag.meta()
        click.echo(f"Executed: {meta.total_executed}")
        click.echo(f"Proposed: {meta.total_proposed}")
        if meta.best_composite is not None:
            click.echo(f"Best composite: {meta.best_composite:.4f} ({meta.best_node_id})")
    else:
        click.echo("No experiments yet.")


@main.command()
@click.option("--n", default=5, help="Number of suggestions")
def suggest(n: int) -> None:
    """Get move suggestions from the navigator."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        raise click.ClickException("No DAG found. Submit seed experiments first.")

    from autotune.dag import ExperimentDAG
    from autotune.navigator import StrategyNavigator

    dag = ExperimentDAG.load(dag_path)
    config = _load_config(root)
    params = config.get("parameters", {})
    manifold = _load_or_create_manifold(root, params)
    forgetting = _load_or_create_forgetting(root, config)
    nav = StrategyNavigator(dag, manifold, forgetting)
    moves = nav.suggest_moves(n=n)

    if not moves:
        click.echo("No moves available.")
        return

    click.echo(f"Move suggestions (top {len(moves)}):\n")
    for i, m in enumerate(moves, 1):
        parents = ", ".join(m.parent_ids)
        click.echo(f"{i}. {m.move_type.upper()} from {parents} (score: {m.score:.3f})")
        click.echo(f"   Direction: {m.direction_label}")
        click.echo(f"   Reasoning: {m.reasoning}")
        click.echo()


@main.command()
def explain() -> None:
    """Explain manifold, frontier, and recommended strategy."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    config = _load_config(root)
    params = config.get("parameters", {})
    manifold = _load_or_create_manifold(root, params)
    forgetting = _load_or_create_forgetting(root, config)
    dag_path = root / "autotune" / "dag.json"

    if not dag_path.exists():
        click.echo("No experiments yet.")
        # Manifold explanation without a DAG: only show manifold state.
        from autotune.navigator import StrategyNavigator

        nav = StrategyNavigator(dag=None, manifold=manifold, forgetting=forgetting)  # type: ignore[arg-type]
        click.echo(nav.explain_manifold())
        return

    from autotune.dag import ExperimentDAG
    from autotune.navigator import StrategyNavigator

    dag = ExperimentDAG.load(dag_path)
    nav = StrategyNavigator(dag, manifold, forgetting)
    click.echo(nav.explain_manifold())
    click.echo()
    click.echo(nav.explain_frontier())
    click.echo()
    click.echo(nav.recommend_strategy())


@main.command()
def frontier() -> None:
    """Show current Pareto front."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        raise click.ClickException("No DAG found.")

    from autotune.dag import ExperimentDAG
    from autotune.navigator import StrategyNavigator

    dag = ExperimentDAG.load(dag_path)
    config = _load_config(root)
    params = config.get("parameters", {})
    manifold = _load_or_create_manifold(root, params)
    forgetting = _load_or_create_forgetting(root, config)
    nav = StrategyNavigator(dag, manifold, forgetting)
    click.echo(nav.explain_frontier())


@main.command()
def check() -> None:
    """Verify project health and consistency."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    issues: list[str] = []

    # 1. Config parseable
    config_path = root / "autotune" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            issues.append(f"config.yaml parse error: {e}")
    else:
        issues.append("config.yaml not found")

    # 2. Orchestrator directories
    orch = root / "autotune" / "orchestrator"
    for subdir in ("queue", "running", "completed", "archive", "checkpoints"):
        if not (orch / subdir).is_dir():
            issues.append(f"orchestrator/{subdir} directory missing")

    # 3. DAG validation (if it exists)
    dag_path = root / "autotune" / "dag.json"
    if dag_path.exists():
        import json

        try:
            with open(dag_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            issues.append(f"dag.json parse error: {e}")
            data = None

        if data is not None:
            if "schema_version" not in data:
                issues.append("dag.json missing schema_version")
            nodes = data.get("nodes", {})
            all_ids = set(nodes.keys())
            for nid, node in nodes.items():
                for pid in node.get("parent_ids", []):
                    if pid not in all_ids:
                        issues.append(f"orphan parent reference: {nid} -> {pid}")

    if issues:
        for issue in issues:
            click.echo(f"  Issue: {issue}")
    else:
        click.echo("All checks passed.")


@main.command()
def reconcile() -> None:
    """Sync DAG with orchestrator state."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        click.echo("No DAG to reconcile.")
        return

    from autotune.dag import ExperimentDAG

    dag = ExperimentDAG.load(dag_path)
    orch = root / "autotune" / "orchestrator"
    dag.reconcile(
        orch / "queue",
        orch / "running",
        orch / "completed",
        orch / "archive",
    )
    dag.save()
    meta = dag.meta()
    click.echo(f"Reconciled. Executed: {meta.total_executed}, Best: {meta.best_composite}")
