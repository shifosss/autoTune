"""Strategy navigator: combines DAG scores, manifold, and forgetting."""

from __future__ import annotations

import logging
import math

from autotune.dag import ExperimentDAG
from autotune.forgetting import ForgettingDetector
from autotune.manifold import ManifoldNavigator
from autotune.types import DAGMeta, DAGNode, MoveProposal

logger = logging.getLogger(__name__)


class StrategyNavigator:
    """Bridge between agent strategy and numerical experiment proposals.

    Combines DAG-UCB scores, manifold principal components, and forgetting
    danger zones to produce ranked MoveProposal objects. Does not manage
    any DAG or manifold state — those objects are owned externally and
    passed in.

    Args:
        dag: The experiment DAG to read kept nodes and metadata from.
        manifold: The manifold navigator for PC-based direction proposals.
        forgetting: The forgetting detector for danger zone filtering.
    """

    def __init__(
        self,
        dag: ExperimentDAG,
        manifold: ManifoldNavigator,
        forgetting: ForgettingDetector,
    ) -> None:
        self._dag = dag
        self._manifold = manifold
        self._forgetting = forgetting

    def suggest_moves(
        self,
        n: int = 5,
        budget_hours: float | None = None,
    ) -> list[MoveProposal]:
        """Generate ranked candidate next experiments.

        Produces branch proposals from each kept node and merge proposals
        from complementary pairs. Proposals in manifold danger zones are
        skipped. Results are sorted by score descending and truncated to n.

        Args:
            n: Maximum number of proposals to return.
            budget_hours: Optional GPU-hour budget filter (unused currently,
                reserved for future cost-gating logic).

        Returns:
            List of MoveProposal objects sorted by score descending.
        """
        proposals: list[MoveProposal] = []
        kept = self._dag.kept_nodes()
        if not kept:
            logger.debug("suggest_moves: no kept nodes, returning empty list")
            return proposals
        meta = self._dag.meta()

        # Branch proposals: one per kept node, along a manifold PC direction
        for node in kept:
            theta = self._manifold.propose_direction(node.theta)
            coords = self._manifold.project(theta)
            if self._forgetting.is_in_danger_zone_coords(coords):
                logger.debug("Branch from %s skipped: proposed theta in danger zone", node.id)
                continue
            score = self._score_branch(node, meta)
            proposals.append(
                MoveProposal(
                    move_type="branch",
                    parent_ids=(node.id,),
                    suggested_theta=theta,
                    merge_strategy=None,
                    merge_weights=None,
                    direction_label=self._direction_label(node.theta, theta),
                    score=score,
                    reasoning=self._branch_reasoning(node),
                )
            )

        # Merge proposals: one per sufficiently diverse kept-node pair
        for i, a in enumerate(kept):
            for b in kept[i + 1 :]:
                div = self._diversity(a, b)
                if div < 0.1:
                    continue
                score = self._score_merge(a, b, div, meta)
                merged_theta = {
                    k: 0.5 * a.theta.get(k, 0) + 0.5 * b.theta.get(k, 0)
                    for k in set(a.theta) | set(b.theta)
                }
                proposals.append(
                    MoveProposal(
                        move_type="merge",
                        parent_ids=(a.id, b.id),
                        suggested_theta=merged_theta,
                        merge_strategy="weighted_avg",
                        merge_weights=(0.5, 0.5),
                        direction_label=f"merge {a.id}+{b.id}",
                        score=score,
                        reasoning=self._merge_reasoning(a, b, div),
                    )
                )

        proposals.sort(key=lambda p: p.score, reverse=True)
        logger.debug("suggest_moves: %d proposals generated, returning top %d", len(proposals), n)
        return proposals[:n]

    def score_move(self, move: MoveProposal) -> float:
        """Return the pre-computed score for a MoveProposal.

        Args:
            move: The proposal to score.

        Returns:
            The proposal's score field.
        """
        return move.score

    def explain_manifold(self) -> str:
        """Produce a human-readable summary of the learned manifold structure.

        Returns a brief message if the manifold has not yet been trained
        (fewer than 10 evaluated points). Otherwise describes effective
        dimensionality, parameter importances, and the top principal components.

        Returns:
            Multi-line string describing the current manifold state.
        """
        state = self._manifold.state()
        if state.generation == 0:
            return "Manifold not yet learned (need 10+ evaluated points)."
        pcs = self._manifold.principal_components()
        importance = self._manifold.parameter_importance()
        lines = [
            f"Manifold: {state.effective_dim} effective dimensions (generation {state.generation})",
            "",
            "Parameter importance:",
        ]
        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {imp:.1%}")
        lines.append("")
        for i, pc in enumerate(pcs[:3]):
            top = sorted(pc.items(), key=lambda x: -abs(x[1]))[:3]
            desc = ", ".join(f"{n}{'↑' if v > 0 else '↓'}" for n, v in top)
            lines.append(f"  PC{i + 1}: {desc}")
        return "\n".join(lines)

    def explain_frontier(self) -> str:
        """Produce a human-readable summary of the current Pareto front.

        Returns a short message if no kept nodes exist. Otherwise lists
        each kept node with its task score, retention score, cost, and rank,
        sorted by composite score descending.

        Returns:
            Multi-line string describing the Pareto frontier.
        """
        front = self._dag.kept_nodes()
        if not front:
            return "No Pareto front yet."
        lines = ["Pareto front (rank 0-1 nodes):", ""]
        for node in sorted(front, key=lambda n: -(n.composite or 0)):
            ret = f", ret={node.retention_score:.3f}" if node.retention_score is not None else ""
            lines.append(
                f"  {node.id}: task={node.task_score:.3f}{ret}, "
                f"cost={node.cost:.2f}, rank={node.pareto_rank}"
            )
        return "\n".join(lines)

    def recommend_strategy(self) -> str:
        """Return a one-line strategy recommendation based on the top-scored move.

        Returns:
            String describing the top-ranked move, or an instruction to add
            seed experiments if no kept nodes are available.
        """
        moves = self.suggest_moves(n=1)
        if not moves:
            return "No moves available. Add seed experiments first."
        m = moves[0]
        return f"Recommend: {m.move_type} {m.direction_label} (score={m.score:.3f}). {m.reasoning}"

    # ── scoring helpers ────────────────────────────────────────────────────────

    def _score_branch(self, parent: DAGNode, meta: DAGMeta) -> float:
        """Compute the branch proposal score for a single parent node.

        Score = (composite + exploration_bonus) * retention_health.

        Args:
            parent: The kept node to branch from.
            meta: Current DAG metadata snapshot.

        Returns:
            Float score for this branch proposal.
        """
        composite = parent.composite or 0.0
        w_e = meta.scoring.exploration_weight
        total = max(meta.total_executed, 1)
        child_count = len(self._dag.children(parent.id))
        exploration = w_e * math.sqrt(math.log(total) / (1 + child_count))
        health = self._retention_health(parent, meta.scoring.retention_cliff_threshold)
        return (composite + exploration) * health

    def _score_merge(
        self,
        a: DAGNode,
        b: DAGNode,
        diversity: float,
        meta: DAGMeta,
    ) -> float:
        """Compute the merge proposal score for a pair of kept nodes.

        Score = (avg_composite + exploration + diversity_bonus) * complementarity.
        Complementarity rewards pairing nodes with differing task/retention profiles.

        Args:
            a: First parent node.
            b: Second parent node.
            diversity: Cosine-distance diversity measure between a and b.
            meta: Current DAG metadata snapshot.

        Returns:
            Float score for this merge proposal.
        """
        avg_composite = 0.5 * ((a.composite or 0) + (b.composite or 0))
        w_e = meta.scoring.exploration_weight
        w_div = meta.scoring.diversity_weight
        total = max(meta.total_executed, 1)
        exploration = w_e * math.sqrt(math.log(total) / 1)
        complementarity = 1.0
        if (
            a.task_score is not None
            and b.task_score is not None
            and a.retention_score is not None
            and b.retention_score is not None
        ):
            complementarity = 1.0 + (
                abs(a.task_score - b.task_score) * abs(a.retention_score - b.retention_score)
            )
        return (avg_composite + exploration + w_div * diversity) * complementarity

    @staticmethod
    def _retention_health(node: DAGNode, cliff_threshold: float) -> float:
        """Return a retention health multiplier for scoring.

        Nodes with retention above the cliff threshold return their retention
        score as the multiplier (reward high retention). Nodes below the
        threshold are penalised with a 0.5 multiplier.

        Args:
            node: The node to evaluate.
            cliff_threshold: Signed threshold (negative, e.g. -0.10); the
                absolute value is used as the cutoff.

        Returns:
            Float in (0, 1].
        """
        if node.retention_score is None:
            return 1.0
        if node.retention_score >= abs(cliff_threshold):
            return node.retention_score
        return node.retention_score * 0.5

    def _diversity(self, a: DAGNode, b: DAGNode) -> float:
        """Compute cosine-distance diversity between two nodes in manifold space.

        Returns 0.5 (neutral) when either node has no manifold projection.

        Args:
            a: First node.
            b: Second node.

        Returns:
            Float in [0, 2] where 0 means identical direction, 1 orthogonal,
            2 opposite.
        """
        coords_a = self._manifold.project(a.theta)
        coords_b = self._manifold.project(b.theta)
        if not coords_a or not coords_b:
            return 0.5
        dot = sum(x * y for x, y in zip(coords_a, coords_b))
        mag_a = math.sqrt(sum(x**2 for x in coords_a)) or 1e-10
        mag_b = math.sqrt(sum(x**2 for x in coords_b)) or 1e-10
        cosine = dot / (mag_a * mag_b)
        return 1.0 - max(-1.0, min(1.0, cosine))

    def _direction_label(self, from_theta: dict[str, float], to_theta: dict[str, float]) -> str:
        """Generate a compact human-readable label for the direction of change.

        Args:
            from_theta: Starting hyperparameter point.
            to_theta: Proposed hyperparameter point.

        Returns:
            Comma-separated parameter change indicators (up to 3), e.g.
            "lr↑, rank↓". Returns "small step" when no parameter changes
            exceed numerical noise.
        """
        changes: list[str] = []
        for k in from_theta:
            if k in to_theta:
                delta = to_theta[k] - from_theta[k]
                if abs(delta) > 1e-10:
                    direction = "↑" if delta > 0 else "↓"
                    changes.append(f"{k}{direction}")
        return ", ".join(changes[:3]) if changes else "small step"

    def _branch_reasoning(self, parent: DAGNode) -> str:
        """Format a branch proposal reasoning string.

        Args:
            parent: The parent node being branched from.

        Returns:
            Human-readable reasoning string.
        """
        ret_str = ""
        if parent.retention_score is not None:
            ret_str = f", retention budget={parent.retention_score:.2f}"
        return f"parent composite={parent.composite:.3f}{ret_str}"

    def _merge_reasoning(self, a: DAGNode, b: DAGNode, div: float) -> str:
        """Format a merge proposal reasoning string.

        Args:
            a: First parent node.
            b: Second parent node.
            div: Diversity score between a and b.

        Returns:
            Human-readable reasoning string.
        """
        return (
            f"diversity={div:.2f}, "
            f"{a.id} task={a.task_score:.3f} ret={a.retention_score}, "
            f"{b.id} task={b.task_score:.3f} ret={b.retention_score}"
        )
