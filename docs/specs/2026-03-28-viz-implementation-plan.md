# Viz Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only 5-tab browser dashboard for human researchers to explore autoTune experiment campaigns via real-time SSE updates.

**Architecture:** aiohttp SSE server watches `dag.json` + `manifold_state.json` via watchdog. Frontend is modular vanilla ES modules (no build step): one file per tab, shared state store, dark glassmorphism theme. CDN deps: 3d-force-graph, d3.js, Plotly.js, three.js.

**Tech Stack:** Python (aiohttp, watchdog), HTML/CSS/JS (vanilla ES modules), CDN (3d-force-graph, d3 v7, Plotly.js basic, three.js)

**Spec:** `docs/specs/2026-03-28-viz-dashboard-design.md`
**Reference:** `autoMIL/src/automil/viz/` (server.py pattern to port)

---

## File Structure

```
src/autotune/viz/
├── __init__.py              # empty
├── server.py                # aiohttp SSE + watchdog (~200 lines)
└── static/
    ├── index.html           # shell: header, tabs, containers (~130 lines)
    ├── style.css            # dark glassmorphism theme (~400 lines)
    ├── main.js              # SSE client, tabs, state, detail panel (~200 lines)
    ├── dag-view.js          # 2D DAG + 3D toggle (~250 lines)
    ├── pareto.js            # Plotly scatter + projection selector (~150 lines)
    ├── manifold.js          # PC scatter + overlays (~200 lines)
    ├── leaderboard.js       # sortable table (~100 lines)
    └── techniques.js        # color-coded stats table (~100 lines)
src/autotune/cli.py          # add viz group (~30 lines added)
tests/test_viz.py            # server unit tests (~120 lines)
```

---

## Task 1: SSE Server and Watcher

**Files:**
- Create: `src/autotune/viz/__init__.py`
- Create: `src/autotune/viz/server.py`
- Create: `tests/test_viz.py`

- [ ] **Step 1: Create the viz package**

```bash
mkdir -p src/autotune/viz/static
```

Create `src/autotune/viz/__init__.py`:
```python
"""autoTune visualization dashboard."""
```

- [ ] **Step 2: Write failing tests for the watcher**

Create `tests/test_viz.py`:

```python
"""Tests for viz server watcher and SSE event construction."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_viz.py -v`
Expected: FAIL (no module)

- [ ] **Step 4: Implement server.py**

Create `src/autotune/viz/server.py`:

```python
"""Visualization dashboard: aiohttp SSE server with watchdog filesystem watcher."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from aiohttp import web
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8420
DEBOUNCE_SEC = 0.2

# ── globals set by cmd_start ─────────────────────────────────────────────────

_dag_path: Path | None = None
_manifold_path: Path | None = None
_pid_file: Path | None = None
_static_dir: Path = Path(__file__).parent / "static"


class DashboardWatcher(FileSystemEventHandler):
    """Watches dag.json and manifold_state.json, broadcasts SSE events."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._prev_dag: dict | None = None
        self._debounce_timer: threading.Timer | None = None
        self._pending_files: set[str] = set()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    # ── watchdog hooks ──

    def on_modified(self, event: Any) -> None:
        self._schedule_notify(event.src_path)

    def on_moved(self, event: Any) -> None:
        self._schedule_notify(event.dest_path)

    def on_created(self, event: Any) -> None:
        self._schedule_notify(event.src_path)

    def _schedule_notify(self, path: str) -> None:
        filename = Path(path).name
        if filename not in ("dag.json", "manifold_state.json"):
            return
        self._pending_files.add(filename)
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
        self._debounce_timer = threading.Timer(DEBOUNCE_SEC, self._flush_pending)
        self._debounce_timer.start()

    def _flush_pending(self) -> None:
        files = self._pending_files.copy()
        self._pending_files.clear()
        self._debounce_timer = None
        for filename in files:
            if filename == "dag.json" and _dag_path and _dag_path.exists():
                self._notify_dag()
            elif filename == "manifold_state.json" and _manifold_path and _manifold_path.exists():
                self._notify_manifold()

    def _notify_dag(self) -> None:
        try:
            data = json.loads(_dag_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read dag.json: %s", e)
            return
        changed, added = self._diff_nodes(data)
        self._prev_dag = data
        event = self._build_dag_event(data, changed, added)
        self._broadcast(event)

    def _notify_manifold(self) -> None:
        try:
            data = json.loads(_manifold_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read manifold_state.json: %s", e)
            return
        event = self._build_manifold_event(data)
        self._broadcast(event)

    def _broadcast(self, event: dict) -> None:
        if not self._loop:
            return
        for q in self._subscribers:
            self._loop.call_soon_threadsafe(q.put_nowait, event)

    # ── diffing ──

    def _diff_nodes(self, current: dict) -> tuple[list[str], list[str]]:
        changed: list[str] = []
        added: list[str] = []
        prev_nodes = (self._prev_dag or {}).get("nodes", {})
        curr_nodes = current.get("nodes", {})
        for nid, node in curr_nodes.items():
            if nid not in prev_nodes:
                added.append(nid)
            elif node != prev_nodes[nid]:
                changed.append(nid)
        return changed, added

    # ── event construction ──

    def _build_dag_event(
        self, data: dict, changed: list[str], added: list[str]
    ) -> dict:
        return {
            "type": "dag_update",
            "changed": changed,
            "added": added,
            "full_dag": data,
        }

    def _build_manifold_event(self, data: dict) -> dict:
        return {"type": "manifold_update", "manifold_state": data}

    # ── initial state ──

    def get_initial_dag(self) -> dict | None:
        if _dag_path and _dag_path.exists():
            try:
                data = json.loads(_dag_path.read_text())
                self._prev_dag = data
                return self._build_dag_event(data, [], list(data.get("nodes", {}).keys()))
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def get_initial_manifold(self) -> dict | None:
        if _manifold_path and _manifold_path.exists():
            try:
                data = json.loads(_manifold_path.read_text())
                return self._build_manifold_event(data)
            except (json.JSONDecodeError, OSError):
                pass
        return None


watcher = DashboardWatcher()


# ── aiohttp handlers ─────────────────────────────────────────────────────────


async def sse_handler(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse()
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    await response.prepare(request)

    # Send initial state
    for event in (watcher.get_initial_dag(), watcher.get_initial_manifold()):
        if event:
            await response.write(f"data: {json.dumps(event)}\n\n".encode())

    queue = watcher.subscribe()
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25.0)
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
            except asyncio.TimeoutError:
                await response.write(b": heartbeat\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        watcher.unsubscribe(queue)
    return response


async def index_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(_static_dir / "index.html")


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/events", sse_handler)
    app.router.add_static("/static", _static_dir)
    return app


# ── CLI entry points ─────────────────────────────────────────────────────────


def cmd_start(port: int = DEFAULT_PORT, project_root: Path | None = None) -> None:
    global _dag_path, _manifold_path, _pid_file

    if project_root is None:
        project_root = Path.cwd()

    autotune_dir = project_root / "autotune"
    _dag_path = autotune_dir / "dag.json"
    _manifold_path = autotune_dir / "manifold_state.json"
    _pid_file = autotune_dir / "orchestrator" / "viz_server.pid"

    # Check stale PID
    if _pid_file.exists():
        try:
            pid = int(_pid_file.read_text().strip())
            os.kill(pid, 0)
            print(f"Viz server already running (pid={pid})")
            return
        except (OSError, ValueError):
            _pid_file.unlink(missing_ok=True)

    # Setup logging
    log_path = autotune_dir / "orchestrator" / "viz_server.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    # Write PID
    _pid_file.parent.mkdir(parents=True, exist_ok=True)
    _pid_file.write_text(str(os.getpid()))

    # Start watchdog
    loop = asyncio.new_event_loop()
    watcher.set_loop(loop)
    observer = Observer()
    observer.schedule(watcher, str(autotune_dir), recursive=False)
    observer.start()

    # Start aiohttp
    stop_event = asyncio.Event()

    def _signal_handler(sig: int, frame: Any) -> None:
        logger.info("Received signal %d, stopping viz server...", sig)
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    async def _run() -> None:
        app = create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", port)
        await site.start()
        logger.info("Viz dashboard running at http://localhost:%d", port)
        await stop_event.wait()
        await runner.cleanup()

    try:
        loop.run_until_complete(_run())
    finally:
        observer.stop()
        observer.join()
        if _pid_file and _pid_file.exists():
            _pid_file.unlink()
        logger.info("Viz server stopped.")


def cmd_status(project_root: Path | None = None) -> None:
    if project_root is None:
        project_root = Path.cwd()
    pid_file = project_root / "autotune" / "orchestrator" / "viz_server.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            print(f"Viz server running (pid={pid})")
        except (OSError, ValueError):
            print("Viz server not running (stale PID file)")
    else:
        print("Viz server not running (no PID file)")


def cmd_stop(project_root: Path | None = None) -> None:
    if project_root is None:
        project_root = Path.cwd()
    pid_file = project_root / "autotune" / "orchestrator" / "viz_server.pid"
    if not pid_file.exists():
        print("Viz server not running (no PID file)")
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to viz server (pid={pid})")
    except (OSError, ValueError) as e:
        print(f"Failed to stop viz server: {e}")
        pid_file.unlink(missing_ok=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_viz.py -v`
Expected: PASS

- [ ] **Step 6: Run full test suite for regression**

Run: `uv run pytest -v --tb=short`
Expected: All pass (124+ existing + new viz tests)

- [ ] **Step 7: Commit**

```bash
git add src/autotune/viz/ tests/test_viz.py
git commit -m "feat(viz): add SSE server with dual-file watcher and debounce"
```

---

## Task 2: CLI Viz Commands

**Files:**
- Modify: `src/autotune/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cli.py`:

```python
class TestVizCLI:
    def test_viz_status_not_running(self, project_dir, monkeypatch):
        monkeypatch.chdir(project_dir)
        runner = CliRunner()
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["viz", "status"])
        assert result.exit_code == 0
        assert "not running" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::TestVizCLI -v`
Expected: FAIL (no viz command)

- [ ] **Step 3: Add viz group to cli.py**

Append to the end of `src/autotune/cli.py` (after the `checkpoints` command):

```python
@main.group()
def viz() -> None:
    """Manage the visualization dashboard server."""


@viz.command("start")
@click.option("--port", default=8420, help="HTTP port")
def viz_start(port: int) -> None:
    """Start the visualization dashboard (foreground)."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.viz.server import cmd_start

    cmd_start(port=port, project_root=root)


@viz.command("stop")
def viz_stop() -> None:
    """Stop the visualization dashboard."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.viz.server import cmd_stop

    cmd_stop(project_root=root)


@viz.command("status")
def viz_status() -> None:
    """Show visualization server status."""
    root = _find_project_root()
    if not root:
        raise click.ClickException("Not in an autoTune project.")
    from autotune.viz.server import cmd_status

    cmd_status(project_root=root)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py::TestVizCLI -v`
Expected: PASS

- [ ] **Step 5: Verify CLI shows viz group**

Run: `uv run autotune --help`
Expected: `viz` appears in command list

- [ ] **Step 6: Commit**

```bash
git add src/autotune/cli.py tests/test_cli.py
git commit -m "feat(cli): add viz start|stop|status commands"
```

---

## Task 3: HTML Shell and CSS Theme

**Files:**
- Create: `src/autotune/viz/static/index.html`
- Create: `src/autotune/viz/static/style.css`

- [ ] **Step 1: Create index.html**

Create `src/autotune/viz/static/index.html` — the dashboard shell with header, tab bar, content containers, detail panel, and CDN script imports. Must use `<script type="module">` for the ES module system. Include all 5 tab containers (`#view-dag`, `#view-pareto`, `#view-manifold`, `#view-leaderboard`, `#view-techniques`), the detail panel, and the stats bar. Load CDN scripts: d3 v7, three.js, three-spritetext, 3d-force-graph, plotly basic. Load `/static/main.js` as entry module.

- [ ] **Step 2: Create style.css**

Create `src/autotune/viz/static/style.css` with the full dark glassmorphism theme per spec section 5. Include:
- `:root` CSS custom properties for all color tokens (status, pareto rank, move type, danger, composite, delta, text)
- Body: `background: linear-gradient(135deg, #0a0e1a, #0f1529, #0a0e1a)`, `overflow: hidden`, flex column
- Top bar: `height: 40px`, glass background (`rgba(15,21,41,0.8)`, `backdrop-filter: blur(20px)`)
- Tab bar: `height: 36px`, active tab with accent background
- Content area: `height: calc(100vh - 76px)`
- Detail panel: `width: 360px`, `position: fixed`, `right: -380px` → `right: 0` transition
- Glass cards: `rgba(15,21,41,0.6)`, `backdrop-filter: blur(20px)`, `border: 1px solid rgba(255,255,255,0.06)`
- Table styles: sticky header, hover row, colored status dots, delta tint cells
- Status dot animations: running pulse (1.5s), connection indicator pulse
- `@media (prefers-reduced-motion: reduce)` reset

- [ ] **Step 3: Verify the shell loads**

Run: `uv run python -c "from aiohttp import web; print('aiohttp ok')"` to verify dep.

Then start server briefly to confirm HTML serves:
```bash
timeout 3 uv run python -c "
from autotune.viz.server import create_app
from aiohttp import web
web.run_app(create_app(), port=8421, print=lambda *a: print('Server started'))
" || true
```

- [ ] **Step 4: Commit**

```bash
git add src/autotune/viz/static/index.html src/autotune/viz/static/style.css
git commit -m "feat(viz): add HTML shell and dark glassmorphism CSS theme"
```

---

## Task 4: Main Module — SSE Client, Tabs, State, Detail Panel

**Files:**
- Create: `src/autotune/viz/static/main.js`

- [ ] **Step 1: Create main.js**

Create `src/autotune/viz/static/main.js` with:
- **State store**: `window.autotuneState = { dag: null, manifold: null, selectedNode: null, activeTab: 'dag' }`
- **STATUS_COLORS** constant object (keep=#3fb950, discard=#f85149, pending=#8b949e, running=#388bfd, oom/crash/timeout=#f85149, cancelled=#484f58)
- **SSE client**: `connectSSE()` with EventSource, exponential backoff reconnect (1s→30s). Routes `dag_update` to dag/pareto/leaderboard/techniques modules, `manifold_update` to manifold module. Updates stats bar on each dag event.
- **Tab switching**: `initTabs()` — click handlers on `.tab` elements, show/hide `.view` containers. Lazy init: calls `module.init(container)` on first activation. Reads `window.location.hash` for initial tab.
- **Module registry**: `const TABS = { dag: null, pareto: null, manifold: null, leaderboard: null, techniques: null }` — populated by dynamic `import()` on first tab activation.
- **Detail panel**: `showDetail(nodeId)` reads node from `autotuneState.dag.nodes[nodeId]`, populates panel HTML (composite large, status badge, metrics grid, parents, techniques, theta, depth, rank, created_at). `hideDetail()` slides panel out. Parent IDs are clickable (call `showDetail` recursively).
- **Stats bar**: `updateStats()` reads `dag.meta` — executed count, proposed count, best composite + node ID.
- **Connection indicator**: `setConnectionStatus(connected)` — green/amber dot, seconds counter updated via `setInterval`.

Export: `showDetail`, `hideDetail`, `autotuneState`, `STATUS_COLORS` as globals on `window` for tab modules to use.

- [ ] **Step 2: Commit**

```bash
git add src/autotune/viz/static/main.js
git commit -m "feat(viz): add main.js with SSE client, tab switching, detail panel"
```

---

## Task 5: Leaderboard Tab

**Files:**
- Create: `src/autotune/viz/static/leaderboard.js`

- [ ] **Step 1: Create leaderboard.js**

Create `src/autotune/viz/static/leaderboard.js` with exports `init(container)`, `render(state)`:
- **init**: Creates table structure with header row (Rank, Node, Task, Retention, Cost, Composite, P.Rank, Type, Status). Retention column has class `retention-col` for hiding in 2-obj mode.
- **render**: Reads `state.dag.nodes`, filters `node_type === "executed"`, sorts by composite descending (default), builds `<tr>` rows. Status cell has colored dot span. Row click calls `window.showDetail(nodeId)`.
- **Sorting**: Click header → toggle `data-sort-dir`, re-sort and re-render. Current sort column highlighted.
- **Show all/limit**: Initially shows 20 rows. "Show all (N)" button at bottom expands.
- **2-objective mode**: If `state.dag.meta.objective_count === 2`, hides `.retention-col` elements.

- [ ] **Step 2: Commit**

```bash
git add src/autotune/viz/static/leaderboard.js
git commit -m "feat(viz): add leaderboard tab with sortable table"
```

---

## Task 6: Techniques Tab

**Files:**
- Create: `src/autotune/viz/static/techniques.js`

- [ ] **Step 1: Create techniques.js**

Create `src/autotune/viz/static/techniques.js` with exports `init(container)`, `render(state)`:
- **init**: Creates table with headers (Technique, Tried, Avg Task Delta, Avg Ret. Delta, Best Par. Delta, Verdict). Retention delta column has class `retention-col`.
- **render**: Reads `state.dag.technique_stats`, sorts by `times_tried` descending. For each technique:
  - Delta cells: green text + `var(--delta-positive)` tint if > 0, red + `var(--delta-negative)` if < 0. Format: `+0.034` / `-0.012`.
  - Verdict logic: `avg_task_delta > 0 && avg_retention_delta >= -0.05` → "Helps" (green badge), `avg_task_delta > 0 && avg_retention_delta < -0.05` → "Mixed" (amber), else → "Hurts" (red).
  - 2-obj mode: hide retention column, verdict = `avg_task_delta > 0`.

- [ ] **Step 2: Commit**

```bash
git add src/autotune/viz/static/techniques.js
git commit -m "feat(viz): add technique effectiveness tab with color-coded verdicts"
```

---

## Task 7: Pareto Frontier Tab

**Files:**
- Create: `src/autotune/viz/static/pareto.js`

- [ ] **Step 1: Create pareto.js**

Create `src/autotune/viz/static/pareto.js` with exports `init(container)`, `render(state)`:
- **init**: Creates controls bar with projection dropdown (`<select>`: "Task vs Retention", "Task vs Cost", "Retention vs Cost") and frontier line checkbox. Creates `<div id="pareto-chart">` for Plotly. In 2-obj mode, dropdown has only "Task vs Cost".
- **render**: Reads executed nodes from `state.dag.nodes`. Based on selected projection:
  - Extracts X, Y arrays from the two selected axes
  - Bubble size = third axis value (e.g., `20 / cost` when task vs retention). Uniform in 2-obj mode.
  - Color = pareto rank mapped to colorscale: `[[0, '#e3b341'], [0.5, '#a5d6ff'], [1, '#6e7681']]`
  - Hover template: `<b>%{text}</b><br>X=%{x:.4f}<br>Y=%{y:.4f}`
  - Click handler: `Plotly.on('plotly_click')` → `window.showDetail(nodeId)`
- **Frontier line**: When checked, filters rank-0 nodes, sorts by X ascending, adds a second trace (scatter mode `lines`, gold dashed).
- **Layout**: Dark background (`#0a0e1a`), white gridlines at 0.05 opacity, axis labels from selected projection.
- Dropdown `onchange` and checkbox `onchange` both call `render(window.autotuneState)`.

- [ ] **Step 2: Commit**

```bash
git add src/autotune/viz/static/pareto.js
git commit -m "feat(viz): add Pareto frontier tab with projection selector"
```

---

## Task 8: Manifold Explorer Tab

**Files:**
- Create: `src/autotune/viz/static/manifold.js`

- [ ] **Step 1: Create manifold.js**

Create `src/autotune/viz/static/manifold.js` with exports `init(container)`, `render(state)`:
- **init**: Creates controls bar (checkbox "Eigenvectors", checkbox "Step ellipse", generation label). Creates layout: left side `<div id="manifold-chart">` (Plotly), right sidebar `<div class="pc-summary">`.
- **render**: If `state.manifold === null` or `state.manifold.generation === 0`, shows centered message "Manifold not yet learned (need 10+ evaluated experiments)" and blank sidebar. Otherwise:
  - **Scatter**: Project each executed node onto top 2 PCs. Color = composite (continuous scale `#1a3a2a` → `#3fb950`). Uniform size. Hover: node ID, composite, theta. Click → detail panel.
  - **Danger zones**: For each zone from forgetting state (if available in dag), draw a red circle shape via Plotly `shapes` array. Fill: `rgba(248,81,73,0.15)`, stroke: `rgba(248,81,73,0.4)`.
  - **Eigenvectors toggle**: When checked, add annotation arrows from mean point along top 2-3 PC directions. Arrow length = eigenvalue. Label = top 2 parameter names with arrows (e.g., "lr↑ rank↑").
  - **Step ellipse toggle**: When checked, draw ellipse shape centered at mean, semi-axes = step_sizes[0] and step_sizes[1]. Dashed white stroke.
  - **Sidebar**: Generation count, top 3 PCs with parameter loadings, parameter importance percentages.
- **PC projection helper**: `projectToPC(theta, manifold)` — normalizes theta, multiplies by PC matrix, returns [pc1, pc2] coordinates.

- [ ] **Step 2: Commit**

```bash
git add src/autotune/viz/static/manifold.js
git commit -m "feat(viz): add manifold explorer tab with PC projection and overlays"
```

---

## Task 9: DAG View Tab

**Files:**
- Create: `src/autotune/viz/static/dag-view.js`

- [ ] **Step 1: Create dag-view.js**

Create `src/autotune/viz/static/dag-view.js` with exports `init(container)`, `render(state)`, `destroy()`:

**2D Mode (default):**
- Creates SVG via d3. Nodes and links extracted from `state.dag.nodes` (parent_ids → links).
- d3-force simulation with: `forceY(d => d.depth * 120)` for layered layout, `forceX(0)` centering, `forceManyBody(-100)`, `forceLink().distance(80)`.
- Node shapes: circle for branch/seed/graft (`r = 4 + composite * 12`), rotated rect for merge. Color = `STATUS_COLORS[status]`.
- Edges: SVG `<line>` with arrowhead `<marker>`. Graft edges: `stroke-dasharray: 4,4`. Pareto spine edges: gold, 3px.
- **Pareto spine**: compute all ancestor node IDs for every rank-0 node. Mark edges connecting spine nodes.
- d3-zoom for pan/zoom.
- Click node → `showDetail(nodeId)`.
- New node animation: `transform: scale(0)` → `scale(1)` over 0.3s.

**3D Mode (toggle):**
- Toggle button: "Switch to 3D" / "Switch to 2D" in top-right.
- On toggle to 3D: remove SVG, create ForceGraph3D instance.
  - `dagMode('td')`, `dagLevelDistance(80)`, orbit controls.
  - `nodeVal(n => 5 + (n.composite || 0) * 20)`, `nodeColor(n => STATUS_COLORS[n.status])`.
  - Merge nodes: custom `nodeThreeObject` with `THREE.OctahedronGeometry`.
  - Graft links: dashed (`linkLineDash([2,2])`).
  - Pareto spine: `linkWidth(l => isSpine ? 3 : 1)`, `linkColor(l => isSpine ? '#e3b341' : '#30363d')`.
  - Labels on keep nodes: SpriteText, truncated 18 chars.
  - Auto-rotate 0.3 speed.
  - Background: `#0a0e1a`.
  - Click → `showDetail`.
- On toggle to 2D: destroy ForceGraph3D, recreate SVG.

**destroy()**: Cleans up ForceGraph3D instance (if 3D mode active) or d3 simulation (if 2D).

- [ ] **Step 2: Commit**

```bash
git add src/autotune/viz/static/dag-view.js
git commit -m "feat(viz): add DAG view tab with 2D layout and 3D ForceGraph toggle"
```

---

## Task 10: Final Verification

**Files:** All

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All pass

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/ tests/`
Expected: All checks passed

- [ ] **Step 3: Run formatter**

Run: `uv run ruff format src/ tests/ && uv run ruff format --check src/ tests/`
Expected: All formatted

- [ ] **Step 4: Verify CLI shows all commands**

Run: `uv run autotune --help`
Expected: 13 commands including `viz` group

- [ ] **Step 5: Verify file sizes**

Run: `wc -l src/autotune/viz/server.py src/autotune/cli.py`
Expected: server.py under 250, cli.py under 800

- [ ] **Step 6: Verify all static files exist**

Run: `ls -la src/autotune/viz/static/`
Expected: index.html, style.css, main.js, dag-view.js, pareto.js, manifold.js, leaderboard.js, techniques.js

- [ ] **Step 7: Commit any fixes**

```bash
git add -A
git commit -m "chore: final verification fixes for viz dashboard"
```

---

## Summary

| Task | What It Delivers | Files |
|------|-----------------|-------|
| 1 | SSE server with dual-file watcher + debounce | viz/server.py, test_viz.py |
| 2 | CLI viz start/stop/status | cli.py, test_cli.py |
| 3 | HTML shell + CSS glassmorphism theme | index.html, style.css |
| 4 | SSE client, tab switching, detail panel, state | main.js |
| 5 | Leaderboard tab (sortable table) | leaderboard.js |
| 6 | Techniques tab (color-coded verdicts) | techniques.js |
| 7 | Pareto frontier (Plotly scatter + projections) | pareto.js |
| 8 | Manifold explorer (PC scatter + overlays) | manifold.js |
| 9 | DAG view (2D default + 3D toggle) | dag-view.js |
| 10 | Final verification (tests, lint, format) | All |
