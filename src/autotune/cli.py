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
@click.option("--node", default=None, help="Node ID to queue (queue mode)")
@click.option("--desc", default=None, help="Experiment description")
@click.option("--theta", default=None, help="Hyperparameters as JSON")
@click.option("--task-score", type=float, default=None, help="Task metric score")
@click.option("--retention-score", type=float, default=None, help="Retention score")
@click.option("--cost", type=float, default=None, help="Cost in GPU-hours")
@click.option("--composite", type=float, default=None, help="Composite score")
@click.option("--techniques", default=None, help="Comma-separated technique tags")
@click.option("--files", multiple=True, help="Files to include in queue spec")
def submit(
    node: str | None,
    desc: str | None,
    theta: str | None,
    task_score: float | None,
    retention_score: float | None,
    cost: float | None,
    composite: float | None,
    techniques: str | None,
    files: tuple[str, ...],
) -> None:
    """Register a seed experiment or queue a proposed node."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    if node is not None:
        _submit_queue(root, node, files)
    else:
        _submit_seed(root, desc, theta, task_score, retention_score, cost, composite, techniques)


def _submit_seed(
    root: Path,
    desc: str | None,
    theta_str: str | None,
    task_score: float | None,
    retention_score: float | None,
    cost: float | None,
    composite: float | None,
    techniques_str: str | None,
) -> None:
    """Register a completed seed experiment in the DAG."""
    import json as json_mod

    from autotune.dag import ExperimentDAG
    from autotune.types import ScoringConfig

    if theta_str is None:
        raise click.ClickException("--theta is required for seed submission.")
    if task_score is None:
        raise click.ClickException("--task-score is required for seed submission.")
    if cost is None:
        raise click.ClickException("--cost is required for seed submission.")

    try:
        theta = json_mod.loads(theta_str)
    except json_mod.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON for --theta: {e}") from e

    config = _load_config(root)
    scoring_cfg = config.get("scoring", {})
    scoring = ScoringConfig(
        exploration_weight=scoring_cfg.get("exploration_weight", 0.005),
        diversity_weight=scoring_cfg.get("diversity_weight", 0.01),
        retention_cliff_threshold=scoring_cfg.get("retention_cliff_threshold", -0.10),
    )
    retention_enabled = config.get("retention", {}).get("enabled", False)
    objective_count = 3 if retention_enabled or retention_score is not None else 2

    dag_path = root / "autotune" / "dag.json"
    if dag_path.exists():
        dag = ExperimentDAG.load(dag_path)
    else:
        dag = ExperimentDAG.create(dag_path, scoring, objective_count=objective_count)

    technique_tags = tuple(t.strip() for t in techniques_str.split(",")) if techniques_str else ()
    effective_composite = composite if composite is not None else task_score

    seed = dag.add_seed(
        theta=theta,
        description=desc or "",
        techniques=technique_tags,
        composite=effective_composite,
        task_score=task_score,
        retention_score=retention_score,
        cost=cost,
    )
    dag.rerank_pareto()
    dag.recalculate_potential()
    dag.save()
    click.echo(f"Seed registered: {seed.id} (composite={effective_composite:.4f})")


def _submit_queue(root: Path, node_id: str, files: tuple[str, ...]) -> None:
    """Queue a proposed node for orchestrator execution."""
    import json as json_mod

    from autotune.dag import ExperimentDAG

    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        raise click.ClickException("No DAG found.")

    dag = ExperimentDAG.load(dag_path)
    node = dag.get_node(node_id)
    if node is None:
        raise click.ClickException(f"Node {node_id} not found in DAG.")
    if node.status != "pending":
        raise click.ClickException(f"Node {node_id} is not pending (status={node.status}).")

    spec = {
        "node_id": node_id,
        "theta": dict(node.theta),
        "parent_ids": list(node.parent_ids),
        "move_type": node.move_type,
        "files": list(files),
    }
    queue_dir = root / "autotune" / "orchestrator" / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_file = queue_dir / f"{node_id}.json"
    queue_file.write_text(json_mod.dumps(spec, indent=2))
    click.echo(f"Queued {node_id} for execution.")


@main.command()
@click.option("--parents", multiple=True, required=True, help="Parent node IDs")
@click.option("--move", required=True, help="Move type: branch, merge, or graft")
@click.option("--desc", required=True, help="Experiment description")
@click.option("--theta", required=True, help="Proposed hyperparameters as JSON")
@click.option("--techniques", default=None, help="Comma-separated technique tags")
@click.option("--merge-strategy", default=None, help="Merge algorithm name")
@click.option("--merge-weights", default=None, help="Comma-separated merge weights")
def propose(
    parents: tuple[str, ...],
    move: str,
    desc: str,
    theta: str,
    techniques: str | None,
    merge_strategy: str | None,
    merge_weights: str | None,
) -> None:
    """Create a proposed experiment in the DAG."""
    import json as json_mod

    from autotune.dag import ExperimentDAG

    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        raise click.ClickException("No DAG found. Submit seed experiments first.")

    if move not in ("branch", "merge", "graft"):
        raise click.ClickException(f"Invalid move type: {move}. Must be branch, merge, or graft.")

    try:
        theta_dict = json_mod.loads(theta)
    except json_mod.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON for --theta: {e}") from e

    dag = ExperimentDAG.load(dag_path)

    for pid in parents:
        if dag.get_node(pid) is None:
            raise click.ClickException(f"Parent node {pid} not found in DAG.")

    technique_tags = tuple(t.strip() for t in techniques.split(",")) if techniques else ()
    weights = tuple(float(w.strip()) for w in merge_weights.split(",")) if merge_weights else None

    node = dag.add_proposed(
        parent_ids=parents,
        move_type=move,
        description=desc,
        techniques=technique_tags,
        theta=theta_dict,
        merge_strategy=merge_strategy,
        merge_weights=weights,
    )
    dag.recalculate_potential()
    dag.save()
    click.echo(f"Proposed: {node.id} ({move} from {', '.join(parents)})")


@main.command()
@click.option("--n", default=10, help="Number of proposals to show")
def rank(n: int) -> None:
    """Show ranked pending proposals."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    dag_path = root / "autotune" / "dag.json"
    if not dag_path.exists():
        raise click.ClickException("No DAG found.")

    from autotune.dag import ExperimentDAG

    dag = ExperimentDAG.load(dag_path)
    ranked = dag.rank_proposals(n=n)

    if not ranked:
        click.echo("No pending proposals.")
        return

    click.echo(f"Ranked proposals (top {len(ranked)}):\n")
    for i, node in enumerate(ranked, 1):
        parents = ", ".join(node.parent_ids)
        click.echo(
            f"{i}. {node.id} [{node.move_type}] from {parents} (potential={node.potential:.3f})"
        )
        if node.strategy_desc:
            click.echo(f"   {node.strategy_desc}")


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


@main.group()
def orchestrator() -> None:
    """Manage the GPU scheduler daemon."""


@orchestrator.command("start")
def orch_start() -> None:
    """Start the orchestrator daemon (foreground)."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.orchestrator import ExperimentOrchestrator

    orch = ExperimentOrchestrator(root)
    orch.cmd_start()


@orchestrator.command("stop")
def orch_stop() -> None:
    """Stop the running orchestrator."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.orchestrator import ExperimentOrchestrator

    orch = ExperimentOrchestrator(root)
    orch.cmd_stop()


@orchestrator.command("status")
def orch_status() -> None:
    """Show orchestrator status."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.orchestrator import ExperimentOrchestrator

    orch = ExperimentOrchestrator(root)
    orch.cmd_status()


@main.command()
@click.option("--prune", is_flag=True, help="Delete checkpoints for pareto_rank >= 2")
def checkpoints(prune: bool) -> None:
    """List or prune managed checkpoints."""
    import shutil

    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")

    store = root / "autotune" / "orchestrator" / "checkpoints"
    if not store.exists():
        click.echo("No checkpoints directory.")
        return

    dag_path = root / "autotune" / "dag.json"
    node_ranks: dict[str, int | None] = {}
    if dag_path.exists():
        from autotune.dag import ExperimentDAG

        dag = ExperimentDAG.load(dag_path)
        for node in dag.all_executed():
            node_ranks[node.id] = node.pareto_rank

    ckpt_dirs = sorted(d for d in store.iterdir() if d.is_dir())
    if not ckpt_dirs:
        click.echo("No checkpoints found.")
        return

    pruned = 0
    for d in ckpt_dirs:
        node_id = d.name
        rank = node_ranks.get(node_id)
        size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)

        if prune and rank is not None and rank >= 2:
            shutil.rmtree(d)
            click.echo(f"  Pruned: {node_id} (rank={rank}, {size_mb:.1f}MB)")
            pruned += 1
        else:
            click.echo(f"  {node_id}: rank={rank}, {size_mb:.1f}MB")

    if prune:
        click.echo(f"Pruned {pruned} checkpoints.")
