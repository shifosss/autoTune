"""Frozen dataclasses for autoTune's core data model.

All types are frozen (immutable) to prevent hidden side effects and enable
safe concurrent access across the experiment orchestration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for DAG-UCB node scoring.

    Attributes:
        exploration_weight: UCB exploration bonus coefficient.
        diversity_weight: Penalty coefficient for technique repetition.
        retention_cliff_threshold: Retention delta below which a cliff is triggered.
    """

    exploration_weight: float
    diversity_weight: float
    retention_cliff_threshold: float


@dataclass(frozen=True)
class DAGMeta:
    """Top-level metadata for an ExperimentDAG.

    Attributes:
        next_id: Monotonically increasing counter for node ID generation.
        total_executed: Number of nodes with node_type == "executed".
        total_proposed: Number of nodes with node_type == "proposed".
        best_composite: Highest composite score seen so far, or None.
        best_node_id: ID of the node with best_composite, or None.
        objective_count: Number of active Pareto objectives (2 or 3).
        scoring: Scoring hyperparameters for DAG-UCB.
    """

    next_id: int
    total_executed: int
    total_proposed: int
    best_composite: float | None
    best_node_id: str | None
    objective_count: int
    scoring: ScoringConfig


@dataclass(frozen=True)
class DAGNode:
    """A single node in the experiment DAG.

    Attributes:
        id: Unique node identifier (e.g. "node_0001").
        parent_ids: Tuple of parent node IDs. Empty for seed nodes.
        move_type: One of "branch", "merge", "graft", "seed".
        merge_strategy: Merge algorithm name, or None for non-merge nodes.
        merge_weights: Per-parent weights for weighted merges, or None.
        theta: Hyperparameter dict (e.g. {"lr": 2e-4, "rank": 16}).
        manifold_coords: Position in the learned manifold space, or None.
        strategy_desc: Human-readable description of the experiment rationale.
        technique_tags: Ordered tuple of technique labels applied at this node.
        task_score: Primary task metric (higher is better), or None if pending.
        retention_score: Retention metric (higher is better), or None if pending.
        cost: Training cost in GPU-hours, or None if pending.
        composite: Scalar composite of all objectives, or None if pending.
        checkpoint_id: Identifier of the saved checkpoint artifact, or None.
        depth: DAG depth (0 for seed nodes).
        potential: Estimated future improvement from DAG-UCB, initially 0.0.
        pareto_rank: NSGA-II Pareto rank (0 = Pareto-optimal), or None.
        node_type: "executed" once training completes, "proposed" otherwise.
        status: Lifecycle status ("pending", "running", "keep", "discard").
        created_at: ISO-8601 creation timestamp.
    """

    id: str
    parent_ids: tuple[str, ...]
    move_type: str  # "branch" | "merge" | "graft" | "seed"

    merge_strategy: str | None
    merge_weights: tuple[float, ...] | None

    theta: dict[str, float]
    manifold_coords: tuple[float, ...] | None

    strategy_desc: str
    technique_tags: tuple[str, ...]

    task_score: float | None
    retention_score: float | None
    cost: float | None
    composite: float | None

    checkpoint_id: str | None

    depth: int
    potential: float
    pareto_rank: int | None

    node_type: str  # "executed" | "proposed"
    status: str
    created_at: str


@dataclass(frozen=True)
class TechniqueStats:
    """Aggregate performance statistics for a named technique.

    Attributes:
        times_tried: Total number of nodes that applied this technique.
        avg_task_delta: Mean task score improvement over parent.
        avg_retention_delta: Mean retention score change over parent.
        best_parent_delta: Best single-step composite improvement observed.
    """

    times_tried: int
    avg_task_delta: float
    avg_retention_delta: float
    best_parent_delta: float


@dataclass(frozen=True)
class DangerZone:
    """A region of manifold space associated with retention cliffs.

    Attributes:
        center: Manifold coordinates at the zone center.
        radius: Exclusion radius around the center.
        cause_technique: Technique tag that triggered the cliff.
        severity: Magnitude of the retention drop that created this zone.
    """

    center: tuple[float, ...]
    radius: float
    cause_technique: str
    severity: float


@dataclass(frozen=True)
class ManifoldState:
    """Snapshot of the CMA-ES learned hyperparameter manifold.

    Attributes:
        param_names: Ordered parameter names matching all coordinate tuples.
        covariance: Full covariance matrix as a tuple of row tuples.
        principal_components: Eigenvectors (rows) sorted by eigenvalue descending.
        eigenvalues: Eigenvalues corresponding to principal_components rows.
        effective_dim: Number of PCs capturing >= 95% of variance.
        mean: Current CMA-ES distribution mean in parameter space.
        step_sizes: Per-parameter step size (sigma) for proposals.
        generation: Number of CMA-ES update steps applied so far.
    """

    param_names: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]
    principal_components: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    effective_dim: int
    mean: tuple[float, ...]
    step_sizes: tuple[float, ...]
    generation: int


@dataclass(frozen=True)
class MoveProposal:
    """A candidate next experiment proposed by StrategyNavigator.

    Attributes:
        move_type: One of "branch", "merge", "graft".
        parent_ids: Node IDs that would become parents of the new node.
        suggested_theta: Proposed hyperparameter values.
        merge_strategy: Merge algorithm name, or None for non-merge proposals.
        merge_weights: Per-parent weights for weighted merges, or None.
        direction_label: Human-readable manifold direction (e.g. "PC1 (+capacity)").
        score: Estimated composite score for this proposal.
        reasoning: Explanation of why this proposal was selected.
    """

    move_type: str  # "branch" | "merge" | "graft"
    parent_ids: tuple[str, ...]
    suggested_theta: dict[str, float]
    merge_strategy: str | None
    merge_weights: tuple[float, ...] | None
    direction_label: str
    score: float
    reasoning: str


@dataclass(frozen=True)
class CliffReport:
    """A detected retention cliff event for a specific executed node.

    Attributes:
        node_id: The node where the cliff was detected.
        parent_ids: Parents of that node (source of comparison).
        delta_retention: Signed retention change (negative = drop).
        cause_techniques: Technique tags applied at node_id.
        severity: Absolute magnitude of the retention drop.
        danger_zone: The manifold danger zone created from this cliff.
    """

    node_id: str
    parent_ids: tuple[str, ...]
    delta_retention: float
    cause_techniques: tuple[str, ...]
    severity: float
    danger_zone: DangerZone
