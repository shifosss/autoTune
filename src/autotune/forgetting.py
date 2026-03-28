"""Forgetting detector: retention cliff detection and danger zones."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

from autotune.types import CliffReport, DAGNode, DangerZone


class ForgettingDetector:
    """Detects catastrophic forgetting and tracks manifold danger zones.

    Compares retention scores between parent and child nodes to identify
    retention cliffs. When a cliff is detected, a danger zone is recorded
    at the child node's manifold coordinates to guide future proposals
    away from that region.

    Attributes:
        _cliff_threshold: Signed retention delta below which a cliff is flagged.
        _danger_zones: Accumulated danger zones from detected cliffs and manual adds.
    """

    def __init__(self, cliff_threshold: float = -0.10) -> None:
        """Initialize the detector with a configurable cliff threshold.

        Args:
            cliff_threshold: Signed retention delta (e.g. -0.10) below which
                a retention drop is classified as a cliff. Must be negative.
        """
        self._cliff_threshold = cliff_threshold
        self._danger_zones: list[DangerZone] = []

    def check_cliff(
        self,
        node: DAGNode,
        parents: tuple[DAGNode, ...],
        step_size: float = 0.5,
    ) -> CliffReport | None:
        """Check whether a node represents a retention cliff relative to its parents.

        For merge nodes, the best (maximum) parent retention is used as the
        baseline, matching the intuition that merging should not regress below
        the strongest parent.

        A detected cliff automatically appends a DangerZone to the internal
        zone list at the node's manifold coordinates.

        Args:
            node: The executed node to evaluate.
            parents: All parent nodes of `node`.
            step_size: Manifold step size of the direction taken (for danger zone radius).

        Returns:
            A CliffReport if a cliff is detected, or None if retention is
            stable, missing, or no parents have retention scores.
        """
        if node.retention_score is None:
            return None

        parent_retentions = [p.retention_score for p in parents if p.retention_score is not None]
        if not parent_retentions:
            return None

        best_parent_retention = max(parent_retentions)
        delta = node.retention_score - best_parent_retention

        if delta >= self._cliff_threshold:
            return None

        severity = abs(delta)
        coords = node.manifold_coords or ()
        dz = DangerZone(
            center=coords,
            radius=severity * step_size,
            cause_technique=(node.technique_tags[0] if node.technique_tags else "unknown"),
            severity=severity,
        )
        self._danger_zones.append(dz)

        return CliffReport(
            node_id=node.id,
            parent_ids=node.parent_ids,
            delta_retention=delta,
            cause_techniques=node.technique_tags,
            severity=severity,
            danger_zone=dz,
        )

    def add_danger_zone(
        self,
        center: tuple[float, ...],
        radius: float,
        cause: str,
        severity: float,
    ) -> None:
        """Manually register a danger zone on the manifold.

        Args:
            center: Manifold coordinates at the zone center.
            radius: Exclusion radius around the center.
            cause: Technique tag that triggered the cliff.
            severity: Magnitude of the retention drop.
        """
        self._danger_zones.append(
            DangerZone(
                center=center,
                radius=radius,
                cause_technique=cause,
                severity=severity,
            )
        )

    def danger_zones(self) -> tuple[DangerZone, ...]:
        """Return all accumulated danger zones as an immutable tuple.

        Returns:
            Tuple of DangerZone instances, ordered by insertion time.
        """
        return tuple(self._danger_zones)

    def is_in_danger_zone_coords(self, coords: tuple[float, ...]) -> bool:
        """Check whether manifold coordinates fall inside any danger zone.

        Distance is computed over the minimum shared dimensionality between
        the query point and each zone center.

        Args:
            coords: Manifold coordinates to test.

        Returns:
            True if `coords` lies within the radius of at least one zone.
        """
        for dz in self._danger_zones:
            if not dz.center:
                continue
            min_len = min(len(coords), len(dz.center))
            dist = math.sqrt(sum((coords[i] - dz.center[i]) ** 2 for i in range(min_len)))
            if dist <= dz.radius:
                return True
        return False

    def worst_technique_for_retention(self) -> str | None:
        """Return the technique most frequently associated with cliffs.

        Returns:
            Technique tag string, or None if no danger zones have been recorded.
        """
        if not self._danger_zones:
            return None
        counts = Counter(dz.cause_technique for dz in self._danger_zones)
        return counts.most_common(1)[0][0]

    def save(self, path: Path) -> None:
        """Persist detector state to a JSON file atomically.

        Writes to a temporary file in the same directory then renames,
        ensuring the target path is never left in a partial state.

        Args:
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written or renamed.
        """
        data = {
            "cliff_threshold": self._cliff_threshold,
            "danger_zones": [
                {
                    "center": list(dz.center),
                    "radius": dz.radius,
                    "cause_technique": dz.cause_technique,
                    "severity": dz.severity,
                }
                for dz in self._danger_zones
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp, str(path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def load(path: Path) -> ForgettingDetector:
        """Restore a ForgettingDetector from a previously saved JSON file.

        Args:
            path: Source file path produced by :meth:`save`.

        Returns:
            A ForgettingDetector instance with all danger zones restored.

        Raises:
            FileNotFoundError: If `path` does not exist.
            json.JSONDecodeError: If the file is malformed.
            KeyError: If required fields are missing from the JSON.
        """
        with open(path) as f:
            data = json.load(f)

        detector = ForgettingDetector(cliff_threshold=data["cliff_threshold"])
        for dz_data in data.get("danger_zones", []):
            detector._danger_zones.append(
                DangerZone(
                    center=tuple(dz_data["center"]),
                    radius=dz_data["radius"],
                    cause_technique=dz_data["cause_technique"],
                    severity=dz_data["severity"],
                )
            )
        return detector
