# autoTune Visualization Dashboard Design

**Date**: 2026-03-28
**Status**: Draft
**Scope**: SSE server, 5-tab read-only dashboard, CLI commands
**Depends on**: Architecture spec (2026-03-27), orchestrator implementation (complete)

---

## 1. Overview

A read-only browser dashboard for human researchers to understand the experiment landscape, inspect trade-offs, and build intuition about the autoTune campaign. The human observes and decides what to tell the coding agent — the dashboard has no control surfaces (no run/prune/cancel buttons).

### Design Decisions

- **2D default DAG + 3D toggle** — immediately readable for most campaigns, toggle to 3D for large DAGs (200+ nodes)
- **Modular vanilla ES modules** — no build step, no framework, each tab is its own JS file
- **Dark glassmorphism** — deep navy background, frosted glass panels, colored node accents
- **Plotly.js for charts** — Pareto scatter and manifold explorer, with interactive hover/click
- **Dual-file SSE** — watches both `dag.json` and `manifold_state.json`, sends separate event types

### Primary Consumer

Human researcher in a supervisory role. Optimized for understanding and exploration, not for direct experiment control. The CLI (`autotune explain`, `autotune suggest`) serves the coding agent.

---

## 2. File Structure

```
src/autotune/viz/
├── __init__.py              # empty
├── server.py                # aiohttp SSE server + watchdog (~200 lines)
└── static/
    ├── index.html           # shell: header, tab bar, content containers (~120 lines)
    ├── style.css            # dark glassmorphism theme (~400 lines)
    ├── main.js              # SSE client, tab switching, shared state, detail panel (~200 lines)
    ├── dag-view.js          # 2D layered DAG (default) + 3D ForceGraph toggle (~250 lines)
    ├── pareto.js            # 2D scatter with projection selector dropdown (~150 lines)
    ├── manifold.js          # PC scatter + danger zones + optional eigenvector overlay (~200 lines)
    ├── leaderboard.js       # sortable top-N table (~100 lines)
    └── techniques.js        # technique stats table with green/red coding (~100 lines)

src/autotune/cli.py          # add viz start|stop|status commands (~30 lines added)
tests/test_viz.py            # NEW: server unit tests (~150 lines)
```

### CDN Dependencies (loaded in `index.html`, no npm)

- `3d-force-graph` + `three.js` + `three-spritetext` — 3D DAG mode
- `d3.js` v7 — 2D DAG layout (force simulation with depth constraints)
- `plotly.js` basic bundle — Pareto scatter + Manifold scatter

---

## 3. Server (`server.py`)

### Architecture

aiohttp web server + watchdog filesystem observer. Watches the `autotune/` directory for changes to `dag.json` and `manifold_state.json`.

### Routes

| Route | Handler | Purpose |
|-------|---------|---------|
| `GET /` | Serve `index.html` | Dashboard shell |
| `GET /events` | SSE stream | Real-time updates |
| `GET /static/{path}` | Static files | JS, CSS, assets |

### SSE Event Types

**`dag_update`** — fired when `dag.json` changes:
```json
{
  "type": "dag_update",
  "changed": ["node_0003"],
  "added": ["node_0004"],
  "full_dag": { "nodes": {}, "meta": {}, "technique_stats": {} }
}
```

**`manifold_update`** — fired when `manifold_state.json` changes:
```json
{
  "type": "manifold_update",
  "manifold_state": { "param_names": [], "principal_components": [], ... }
}
```

### Behaviors

- **Debounce**: 200ms window to batch rapid filesystem events (atomic writes via `os.rename` trigger multiple watchdog events)
- **Heartbeat**: `:heartbeat\n\n` comment every 25 seconds to keep connections alive through proxies
- **Initial load**: On SSE connect, immediately sends both `dag_update` and `manifold_update` with full state
- **Diff tracking**: Compares node IDs between consecutive dag events to populate `changed` and `added` arrays
- **Atomic write detection**: Handles `on_moved` events (catches `os.rename` pattern used by `ExperimentDAG.save()`)

### CLI Methods

- `cmd_start(port, project_root)` — PID file at `orchestrator/viz_server.pid`, log to `viz_server.log`, default port 8420
- `cmd_stop(project_root)` — SIGTERM to PID
- `cmd_status(project_root)` — check PID liveness, print port

---

## 4. Frontend Shell (`index.html` + `main.js`)

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ◉ autoTune    Executed: 23  Proposed: 5  Best: 0.8742     │  header (40px)
│                                          ● Connected  12s  │
├─────────────────────────────────────────────────────────────┤
│  [DAG View]  [Pareto]  [Manifold]  [Leaderboard]  [Techs]  │  tab bar (36px)
├──────────────────────────────────────────────┬──────────────┤
│                                              │  Detail      │
│              TAB CONTENT                     │  Panel       │
│              (fills viewport)                │  (360px)     │
│                                              │  slides in   │
│                                              │  on click    │
└──────────────────────────────────────────────┴──────────────┘
```

### `main.js` Responsibilities

- **SSE client**: `EventSource('/events')` with exponential backoff reconnect (1s → 30s max)
- **State store**: `window.autotuneState = { dag: null, manifold: null, selectedNode: null, activeTab: 'dag' }` — simple object, each tab module reads from it
- **Tab switching**: Show/hide content containers, call each module's `render()` on activation. Lazy initialization — a tab's `init()` is called only on first selection.
- **Detail panel**: Shared right-slide panel (360px, `transition: 0.3s cubic-bezier`). Any tab can call `showDetail(nodeId)`. Shows: node ID, status badge, move type, composite (large), metrics grid (task/retention/cost), parent IDs (clickable), technique tags, theta values, depth, pareto rank, created_at.
- **Stats bar**: Reads `dag.meta` on each `dag_update` — shows executed count, proposed count, best composite + best node ID.
- **Connection indicator**: Green dot = connected, amber pulsing = reconnecting, seconds-since-last-update counter.

### Module Contract

Each tab JS module exports:
```js
export function init(container)  // called once when tab first activated
export function render(state)    // called on each relevant SSE update
export function destroy()        // called when tab deactivated (optional cleanup)
```

### SSE Routing to Tabs

- `dag_update` → calls `render()` on dag-view, pareto, leaderboard, techniques (all read DAG data)
- `manifold_update` → calls `render()` on manifold only

---

## 5. Color System

### Base Theme (CSS Custom Properties)

```css
:root {
  /* Backgrounds */
  --bg-base:      #0a0e1a;
  --bg-surface:   rgba(15, 21, 41, 0.6);
  --bg-elevated:  rgba(25, 32, 56, 0.8);
  --bg-border:    rgba(255, 255, 255, 0.06);

  /* Glass effect */
  --glass-blur:   blur(20px);

  /* Status colors */
  --status-keep:      #3fb950;
  --status-discard:   #f85149;
  --status-pending:   #8b949e;
  --status-running:   #388bfd;
  --status-failure:   #f85149;   /* oom, crash, timeout share this */
  --status-cancelled: #484f58;

  /* Pareto rank */
  --pareto-rank-0:  #e3b341;   /* gold */
  --pareto-rank-1:  #a5d6ff;   /* pale blue */
  --pareto-rank-2:  #6e7681;   /* muted gray */

  /* Move type accents */
  --move-branch:  #d2a8ff;   /* purple */
  --move-merge:   #ffa657;   /* orange */
  --move-graft:   #79c0ff;   /* light blue */
  --move-seed:    #56d364;   /* bright green */

  /* Danger zone */
  --danger-fill:   rgba(248, 81, 73, 0.15);
  --danger-stroke: rgba(248, 81, 73, 0.4);

  /* Composite gradient (for manifold scatter) */
  --composite-low:  #1a3a2a;
  --composite-high: #3fb950;

  /* Text */
  --text-primary:   #e6edf3;
  --text-secondary: #8b949e;
  --text-muted:     #484f58;

  /* Delta colors (for technique table cells) */
  --delta-positive: rgba(63, 185, 80, 0.15);
  --delta-negative: rgba(248, 81, 73, 0.15);
}
```

### Status Color Map (shared JS constant in `main.js`)

```js
const STATUS_COLORS = {
  keep:      '#3fb950',
  discard:   '#f85149',
  pending:   '#8b949e',
  running:   '#388bfd',
  oom:       '#f85149',
  crash:     '#f85149',
  timeout:   '#f85149',
  cancelled: '#484f58',
};
```

---

## 6. Tab 1: DAG View (`dag-view.js`)

### 2D Mode (Default)

- Layered top-down layout: nodes pinned to `y = depth * 120px`, horizontal position by d3-force simulation
- Nodes are SVG circles: size = `8 + composite * 20` pixels, color = status
- Merge nodes: diamond shape (rotated square SVG `<rect transform="rotate(45)">`)
- Graft edges: dashed stroke (`stroke-dasharray: 4,4`)
- Normal edges: solid with arrowhead markers (`<marker>`)
- **Pareto spine**: edges from roots to rank-0 nodes highlighted in gold (`#e3b341`), 3px width. Computed by tracing all ancestors of every rank-0 node.
- **Danger zones**: translucent red circles centered on the node that triggered the cliff (matched by `DangerZone.cause_technique` to the triggering node's technique tags). Radius is visual-only (fixed 40px), not proportional to manifold radius (which has no meaningful 2D mapping).
- Click node → opens detail panel
- Zoom/pan via d3-zoom

### 3D Mode (Toggle)

- Toggle button in top-right corner: "Switch to 3D" / "Switch to 2D"
- ForceGraph3D with `dagMode: 'td'`, orbit controls (preserves vertical axis)
- Same encoding: size=composite, color=status
- Merge nodes: `THREE.OctahedronGeometry` (diamond silhouette)
- Graft edges: `linkLineDash([2,2])`
- Pareto spine: thicker links (3px) with directional particles
- Danger zones: translucent red `THREE.SphereGeometry` added to scene
- Keep-node labels: SpriteText (truncated to 18 chars)
- Auto-rotate at 0.3 speed when idle, stops on user interaction
- Background: `#0a0e1a`

### Toggle Behavior

On switch, the current view is destroyed and the other is initialized with the same data. No split-screen. State (selected node, zoom level) is not preserved across toggles.

---

## 7. Tab 2: Pareto Frontier (`pareto.js`)

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Projection: [task vs retention ▼]      ☐ Show frontier line │  controls bar
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    Plotly 2D scatter                          │
│                    (fills remaining height)                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Projection Dropdown

3-objective mode (retention enabled):
- Task vs Retention (default)
- Task vs Cost
- Retention vs Cost

2-objective mode (retention disabled):
- Task vs Cost (only option)

### Scatter Encoding

- Each executed node is a bubble
- X/Y = selected projection axes
- Bubble size = third objective (e.g., `1/cost` when showing task vs retention, so cheaper experiments appear larger). Uniform size in 2-objective mode.
- Color = pareto rank: gold (`#e3b341`) rank 0, pale blue (`#a5d6ff`) rank 1, muted gray (`#6e7681`) rank 2+
- Hover tooltip: node ID, all three scores, move type, technique tags

### Frontier Line Toggle

When checked: connects rank-0 nodes in ascending X-axis order with a gold dashed polyline. Shows the Pareto trade-off boundary.

### Click Behavior

Click a bubble → opens shared detail panel with that node.

---

## 8. Tab 3: Manifold Explorer (`manifold.js`)

### Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  ☐ Eigenvectors  ☐ Step ellipse              Generation: 3      │  controls
├──────────────────────────────────────────┬───────────────────────┤
│                                          │  PC Summary           │
│         Plotly 2D scatter                │                       │
│         (PC1 x PC2)                      │  PC1: lr↑ rank↑      │
│                                          │  PC2: dropout↑ warm↓  │
│                                          │                       │
│                                          │  Importance:          │
│                                          │  lr: 42%              │
│                                          │  rank: 31%            │
│                                          │  dropout: 18%         │
│                                          │  warmup: 9%           │
└──────────────────────────────────────────┴───────────────────────┘
```

### Scatter (always shown)

- Each executed node projected onto top 2 PCs
- Color = composite score (continuous green gradient: `--composite-low` to `--composite-high`)
- Size = uniform
- Hover: node ID, composite, theta values
- Click → detail panel

### Danger Zones (always shown)

- Red translucent circles at each `DangerZone.center` (first 2 PC dimensions)
- Radius proportional to `DangerZone.radius`
- Fill: `--danger-fill`, stroke: `--danger-stroke`
- Hover tooltip: cause technique, severity

### Toggle: Eigenvectors (research-grade overlay)

- Arrows from the manifold mean point along top 2-3 PC directions
- Arrow length proportional to eigenvalue (explained variance)
- Labels on arrows: top 2 parameter contributions (e.g., "lr↑ rank↑")

### Toggle: Step Ellipse (research-grade overlay)

- Ellipse centered at manifold mean
- Semi-axes = `step_size[0]` along PC1, `step_size[1]` along PC2
- Shows the current exploration radius — where the next proposals will sample
- Stroke: dashed white, 1px

### Right Sidebar — PC Summary (always shown)

- Generation count
- Top 3 PCs with dominant parameter loadings (name + direction arrow)
- Parameter importance percentages from eigenvalue ratios

### Pre-Manifold State (generation 0)

Centered message: "Manifold not yet learned (need 10+ evaluated experiments)". Sidebar shows parameter names with "—" for importance values.

---

## 9. Tab 4: Leaderboard (`leaderboard.js`)

Sortable table of all executed nodes.

### Columns

| Column | Width | Sort | Notes |
|--------|-------|------|-------|
| Rank | 50px | — | Sequential by current sort |
| Node | 90px | asc/desc | node_id |
| Task | 70px | asc/desc | task_score, 4 decimal places |
| Retention | 70px | asc/desc | Hidden in 2-objective mode |
| Cost | 60px | asc/desc | 2 decimal places |
| Composite | 80px | asc/desc | Default sort (descending) |
| P.Rank | 60px | asc/desc | pareto_rank |
| Type | 70px | — | move_type |
| Status | 80px | — | Colored dot + text |

### Behaviors

- Default: sorted by composite descending, top 20 shown
- "Show all" button to expand
- Clickable rows → detail panel
- Click column header → sort by that column (toggle asc/desc)
- Sticky header row with glass blur backdrop
- Hover row highlight

---

## 10. Tab 5: Technique Effectiveness (`techniques.js`)

Table of technique stats with retention-aware verdict.

### Columns

| Column | Width | Notes |
|--------|-------|-------|
| Technique | 120px | technique tag name |
| Tried | 50px | times_tried |
| Avg Task Delta | 90px | Green text if > 0, red if < 0, cell tint |
| Avg Retention Delta | 100px | Same color coding. Hidden in 2-obj mode |
| Best Parent Delta | 100px | Same color coding |
| Verdict | 80px | Computed badge |

### Verdict Logic

- `avg_task_delta > 0 AND avg_retention_delta >= -0.05` → "Helps" (green badge)
- `avg_task_delta > 0 AND avg_retention_delta < -0.05` → "Mixed" (amber badge) — improves task but hurts retention
- `avg_task_delta <= 0` → "Hurts" (red badge)

In 2-objective mode (no retention): verdict is simply `avg_task_delta > 0` → "Helps", else "Hurts".

### Sort

Default: sorted by times_tried descending. Column headers sortable.

---

## 11. Detail Panel

Shared across all tabs. Slides in from right (360px wide, `transition: right 0.3s cubic-bezier(0.4, 0, 0.2, 1)`).

### Content

```
┌─────────────────────────┐
│  ✕                      │  close button
│                         │
│  0.8742                 │  composite (large, 32px)
│  node_0012              │  node ID
│  ● keep    branch       │  status badge + move type
│                         │
│  ─────────────────────  │
│  Task      0.8910       │
│  Retention 0.9420       │  metrics grid
│  Cost      1.20         │
│  P.Rank    0            │
│  ─────────────────────  │
│  Parents: node_0008     │  clickable → navigates
│  Depth: 3               │
│  ─────────────────────  │
│  Techniques:            │
│  [lr_increase] [rank↑]  │  tag chips
│  ─────────────────────  │
│  Theta:                 │
│  lr: 4.2e-4             │
│  rank: 28               │  hyperparameter values
│  alpha: 56              │
│  ─────────────────────  │
│  Created: 2026-03-28    │
│  14:30:00               │
└─────────────────────────┘
```

### Behaviors

- Click parent ID → switches detail to that node (stays open)
- Click ✕ or click outside → closes panel
- Adapts to 2-objective mode: hides retention row

---

## 12. Animations

| Element | Animation | Duration | Trigger |
|---------|-----------|----------|---------|
| Running nodes | Pulsing glow (`box-shadow` or `opacity` cycle) | 1.5s ease-in-out loop | status = running |
| Detail panel | Slide in from right | 0.3s cubic-bezier | Node click |
| New node (2D) | Scale 0→1 | 0.3s ease-out | Node added via SSE |
| Status change | Color interpolation | 0.5s | Node status change |
| Pareto rank change | Brief gold flash then settle | 0.3s | Rank 0 achieved |
| Tab switch | Instant (no animation) | — | Tab click |
| Connection indicator | Amber pulse when reconnecting | 1s loop | SSE disconnect |

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
```

---

## 13. CLI Commands

Added to `src/autotune/cli.py`:

```
autotune viz start [--port 8420]   # Start dashboard server (foreground)
autotune viz stop                  # SIGTERM to PID
autotune viz status                # Check if running, print URL
```

Click group pattern matching `orchestrator` subgroup. Each command: find project root → lazy import `from autotune.viz.server import cmd_start/cmd_stop/cmd_status` → call.

---

## 14. Error Handling

| Scenario | Behavior |
|----------|----------|
| `dag.json` missing on startup | SSE sends empty initial state, dashboard shows "No experiments yet" |
| `manifold_state.json` missing | Manifold tab shows "not yet learned" message |
| SSE connection lost | Client shows amber pulsing indicator, EventSource auto-reconnects |
| Malformed JSON in watched files | Server logs warning, skips event, does not crash |
| Port already in use | `cmd_start` prints clear error with port number |
| Browser < ES2020 | Graceful degradation: no ES modules support → show "upgrade browser" message |

---

## 15. Testing Strategy

| Test File | Scope | Approach |
|-----------|-------|----------|
| `tests/test_viz.py` | Server routes, SSE events, watchdog handler, debounce, CLI methods | Mock filesystem events, aiohttp test client, no real browser |

Frontend modules are not unit tested (vanilla JS without a test runner). Correctness is verified via the SSE contract — if the server sends correct events, the frontend renders correctly. Manual testing during development.

---

## 16. Files Changed Summary

| File | Change | Projected Lines |
|------|--------|----------------|
| `src/autotune/viz/__init__.py` | **New** | 1 |
| `src/autotune/viz/server.py` | **New** | ~200 |
| `src/autotune/viz/static/index.html` | **New** | ~120 |
| `src/autotune/viz/static/style.css` | **New** | ~400 |
| `src/autotune/viz/static/main.js` | **New** | ~200 |
| `src/autotune/viz/static/dag-view.js` | **New** | ~250 |
| `src/autotune/viz/static/pareto.js` | **New** | ~150 |
| `src/autotune/viz/static/manifold.js` | **New** | ~200 |
| `src/autotune/viz/static/leaderboard.js` | **New** | ~100 |
| `src/autotune/viz/static/techniques.js` | **New** | ~100 |
| `src/autotune/cli.py` | Modify | ~684 → ~714 |
| `tests/test_viz.py` | **New** | ~150 |

All files well under 800 lines.
