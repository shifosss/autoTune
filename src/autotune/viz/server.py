"""Visualization dashboard: aiohttp SSE server with watchdog filesystem watcher."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
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
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the asyncio event loop for thread-safe callbacks."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE subscriber queue and return it."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
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
        """Debounce filesystem events: batch rapid writes into one notification."""
        filename = Path(path).name
        if filename not in ("dag.json", "manifold_state.json"):
            return
        with self._lock:
            self._pending_files.add(filename)
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(DEBOUNCE_SEC, self._flush_pending)
            self._debounce_timer.start()

    def _flush_pending(self) -> None:
        """Fire after debounce window: process all pending file notifications."""
        with self._lock:
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
            data = json.loads(_dag_path.read_text())  # type: ignore[union-attr]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read dag.json: %s", e)
            return
        with self._lock:
            changed, added = self._diff_nodes(data)
            self._prev_dag = data
        event = self._build_dag_event(data, changed, added)
        self._broadcast(event)

    def _notify_manifold(self) -> None:
        try:
            data = json.loads(_manifold_path.read_text())  # type: ignore[union-attr]
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
        """Compare current DAG nodes against previous snapshot.

        Returns:
            Tuple of (changed_node_ids, added_node_ids).
        """
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

    def _build_dag_event(self, data: dict, changed: list[str], added: list[str]) -> dict:
        """Build a dag_update SSE event payload."""
        return {
            "type": "dag_update",
            "changed": changed,
            "added": added,
            "full_dag": data,
        }

    def _build_manifold_event(self, data: dict) -> dict:
        """Build a manifold_update SSE event payload."""
        return {"type": "manifold_update", "manifold_state": data}

    # ── initial state ──

    def get_initial_dag(self) -> dict | None:
        """Read current dag.json and return initial dag_update event."""
        if _dag_path and _dag_path.exists():
            try:
                data = json.loads(_dag_path.read_text())
                with self._lock:
                    self._prev_dag = data
                return self._build_dag_event(data, [], list(data.get("nodes", {}).keys()))
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def get_initial_manifold(self) -> dict | None:
        """Read current manifold_state.json and return initial manifold_update event."""
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
    """Handle GET /events — SSE stream with heartbeat."""
    response = web.StreamResponse()
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    await response.prepare(request)

    # Send initial state immediately on connect
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
                # Heartbeat keeps proxies from closing idle connections
                await response.write(b": heartbeat\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        watcher.unsubscribe(queue)
    return response


async def index_handler(request: web.Request) -> web.FileResponse:
    """Serve the dashboard shell HTML."""
    return web.FileResponse(_static_dir / "index.html")


def create_app() -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/events", sse_handler)
    app.router.add_static("/static", _static_dir)
    return app


# ── CLI entry points ─────────────────────────────────────────────────────────


def cmd_start(port: int = DEFAULT_PORT, project_root: Path | None = None) -> None:
    """Start the viz server in foreground. Writes PID file, blocks until SIGTERM.

    Args:
        port: HTTP port to listen on.
        project_root: Path to the ML project root (contains autotune/ dir).
    """
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    """Print whether the viz server is running.

    Args:
        project_root: Path to the ML project root.
    """
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
    """Send SIGTERM to the running viz server.

    Args:
        project_root: Path to the ML project root.
    """
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
