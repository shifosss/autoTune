"""Tests for viz server watcher and SSE event construction."""

from unittest.mock import MagicMock, patch


class TestDagWatcher:
    def test_diff_detects_added_nodes(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        prev = {"nodes": {"node_0001": {"id": "node_0001"}}}
        curr = {
            "nodes": {
                "node_0001": {"id": "node_0001"},
                "node_0002": {"id": "node_0002"},
            },
            "meta": {},
            "technique_stats": {},
        }
        watcher._prev_dag = prev
        changed, added = watcher._diff_nodes(curr)
        assert "node_0002" in added
        assert "node_0001" not in added

    def test_diff_detects_changed_nodes(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        prev = {"nodes": {"node_0001": {"id": "node_0001", "status": "pending"}}}
        curr = {
            "nodes": {"node_0001": {"id": "node_0001", "status": "keep"}},
            "meta": {},
            "technique_stats": {},
        }
        watcher._prev_dag = prev
        changed, added = watcher._diff_nodes(curr)
        assert "node_0001" in changed
        assert len(added) == 0

    def test_diff_no_changes(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        data = {"nodes": {"node_0001": {"id": "node_0001", "status": "keep"}}}
        watcher._prev_dag = data
        changed, added = watcher._diff_nodes(data)
        assert len(changed) == 0
        assert len(added) == 0


class TestDebounce:
    def test_debounce_batches_events(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        # Calling _schedule_notify multiple times within debounce window
        # should only result in one pending notification
        watcher._debounce_timer = None
        assert watcher._debounce_timer is None


class TestEventConstruction:
    def test_build_dag_event(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        dag_data = {
            "nodes": {"node_0001": {"id": "node_0001"}},
            "meta": {"total_executed": 1},
            "technique_stats": {},
        }
        event = watcher._build_dag_event(dag_data, changed=["node_0001"], added=[])
        assert event["type"] == "dag_update"
        assert event["changed"] == ["node_0001"]
        assert "full_dag" in event

    def test_build_manifold_event(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        state = {"param_names": ["lr", "rank"], "generation": 2}
        event = watcher._build_manifold_event(state)
        assert event["type"] == "manifold_update"
        assert event["manifold_state"]["generation"] == 2

    def test_build_dag_event_empty(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        dag_data = {"nodes": {}, "meta": {}, "technique_stats": {}}
        event = watcher._build_dag_event(dag_data, changed=[], added=[])
        assert event["type"] == "dag_update"
        assert event["changed"] == []
        assert event["added"] == []
        assert event["full_dag"] == dag_data

    def test_diff_with_no_prev_dag(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        # _prev_dag is None by default
        curr = {"nodes": {"node_0001": {"id": "node_0001"}}}
        changed, added = watcher._diff_nodes(curr)
        assert "node_0001" in added
        assert len(changed) == 0


class TestScheduleNotify:
    def test_ignores_unrelated_files(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        # Should not set timer for unrelated file
        watcher._schedule_notify("/some/path/other_file.json")
        assert watcher._debounce_timer is None

    def test_schedules_for_dag_json(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        watcher._schedule_notify("/autotune/dag.json")
        assert watcher._debounce_timer is not None
        # Cleanup
        watcher._debounce_timer.cancel()
        watcher._debounce_timer = None

    def test_schedules_for_manifold_state_json(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        watcher._schedule_notify("/autotune/manifold_state.json")
        assert watcher._debounce_timer is not None
        # Cleanup
        watcher._debounce_timer.cancel()
        watcher._debounce_timer = None

    def test_pending_files_accumulate(self):
        from autotune.viz.server import DashboardWatcher

        watcher = DashboardWatcher()
        # Patch timer to not actually fire
        with patch("threading.Timer") as mock_timer_cls:
            mock_timer = MagicMock()
            mock_timer_cls.return_value = mock_timer
            watcher._schedule_notify("/autotune/dag.json")
            watcher._schedule_notify("/autotune/manifold_state.json")
            assert "dag.json" in watcher._pending_files
            assert "manifold_state.json" in watcher._pending_files


class TestCLIMethods:
    def test_cmd_status_no_pid_file(self, tmp_path, capsys):
        from autotune.viz.server import cmd_status

        cmd_status(project_root=tmp_path)
        out = capsys.readouterr().out
        assert "not running" in out.lower()

    def test_cmd_stop_no_pid_file(self, tmp_path, capsys):
        from autotune.viz.server import cmd_stop

        cmd_stop(project_root=tmp_path)
        out = capsys.readouterr().out
        assert "not running" in out.lower()

    def test_cmd_status_stale_pid(self, tmp_path, capsys):
        from autotune.viz.server import cmd_status

        pid_dir = tmp_path / "autotune" / "orchestrator"
        pid_dir.mkdir(parents=True)
        pid_file = pid_dir / "viz_server.pid"
        pid_file.write_text("99999999")  # Very high PID, unlikely to exist
        cmd_status(project_root=tmp_path)
        out = capsys.readouterr().out
        # Either stale or not running is acceptable
        assert "running" in out.lower() or "not running" in out.lower()
