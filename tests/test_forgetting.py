import pytest
from autotune.forgetting import ForgettingDetector
from autotune.types import DAGNode


def _make_node(
    node_id: str,
    retention: float | None = None,
    parent_ids: tuple[str, ...] = (),
    move_type: str = "branch",
    techniques: tuple[str, ...] = (),
) -> DAGNode:
    return DAGNode(
        id=node_id,
        parent_ids=parent_ids,
        move_type=move_type,
        merge_strategy=None,
        merge_weights=None,
        theta={},
        manifold_coords=(0.0,),
        strategy_desc="",
        technique_tags=techniques,
        task_score=0.8,
        retention_score=retention,
        cost=1.0,
        composite=0.8,
        checkpoint_id=None,
        depth=1,
        potential=0.0,
        pareto_rank=0,
        node_type="executed",
        status="keep",
        created_at="2026-03-27",
    )


class TestCliffDetection:
    def test_detects_cliff(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        parent = _make_node("p", retention=0.95)
        child = _make_node("c", retention=0.80, parent_ids=("p",), techniques=("lr_increase",))
        report = detector.check_cliff(child, (parent,))
        assert report is not None
        assert report.delta_retention == pytest.approx(-0.15)
        assert report.cause_techniques == ("lr_increase",)

    def test_no_cliff_when_retention_stable(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        parent = _make_node("p", retention=0.95)
        child = _make_node("c", retention=0.92, parent_ids=("p",))
        report = detector.check_cliff(child, (parent,))
        assert report is None

    def test_merge_uses_max_parent_retention(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        p1 = _make_node("p1", retention=0.90)
        p2 = _make_node("p2", retention=0.95)
        child = _make_node(
            "c",
            retention=0.80,
            parent_ids=("p1", "p2"),
            move_type="merge",
            techniques=("merge_weighted",),
        )
        report = detector.check_cliff(child, (p1, p2))
        assert report is not None
        assert report.delta_retention == pytest.approx(-0.15)

    def test_no_cliff_when_no_retention(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        parent = _make_node("p", retention=None)
        child = _make_node("c", retention=None, parent_ids=("p",))
        assert detector.check_cliff(child, (parent,)) is None


class TestDangerZones:
    def test_add_and_query(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        detector.add_danger_zone(
            center=(0.5, 0.3),
            radius=0.15,
            cause="lr_increase",
            severity=0.15,
        )
        zones = detector.danger_zones()
        assert len(zones) == 1
        assert zones[0].cause_technique == "lr_increase"

    def test_is_in_danger_zone(self):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        detector.add_danger_zone(
            center=(0.5, 0.3),
            radius=0.15,
            cause="lr_increase",
            severity=0.15,
        )
        assert detector.is_in_danger_zone_coords((0.5, 0.35))
        assert not detector.is_in_danger_zone_coords((0.9, 0.9))


class TestForgettingPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        detector = ForgettingDetector(cliff_threshold=-0.10)
        detector.add_danger_zone(
            center=(0.5,),
            radius=0.2,
            cause="dropout",
            severity=0.12,
        )
        path = tmp_path / "forgetting.json"
        detector.save(path)
        loaded = ForgettingDetector.load(path)
        assert len(loaded.danger_zones()) == 1
        assert loaded.danger_zones()[0].cause_technique == "dropout"
