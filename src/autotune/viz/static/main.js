/**
 * main.js — SSE client, tab switching, shared state, detail panel.
 *
 * Exports (on window): showDetail, hideDetail, autotuneState, STATUS_COLORS
 */

// ── Shared constants ────────────────────────────────────────────────────────

export const STATUS_COLORS = {
  keep:      '#3fb950',
  discard:   '#f85149',
  pending:   '#8b949e',
  running:   '#388bfd',
  oom:       '#f85149',
  crash:     '#f85149',
  timeout:   '#f85149',
  cancelled: '#484f58',
};

// ── Shared state store ──────────────────────────────────────────────────────

export const autotuneState = {
  dag:          null,   // full_dag payload from last dag_update
  manifold:     null,   // manifold_state from last manifold_update
  selectedNode: null,   // node ID currently shown in detail panel
  activeTab:    'dag',  // current active tab key
};

// Expose globally so tab modules can access without import
window.autotuneState  = autotuneState;
window.STATUS_COLORS  = STATUS_COLORS;

// ── Tab module registry ─────────────────────────────────────────────────────

/** Lazy-loaded module references, keyed by tab name. */
const TAB_MODULES = {
  dag:         null,
  pareto:      null,
  manifold:    null,
  leaderboard: null,
  techniques:  null,
};

/** Track which tabs have been initialized. */
const TAB_INITIALIZED = {
  dag: false, pareto: false, manifold: false, leaderboard: false, techniques: false,
};

const TAB_MODULE_URLS = {
  dag:         '/static/dag-view.js',
  pareto:      '/static/pareto.js',
  manifold:    '/static/manifold.js',
  leaderboard: '/static/leaderboard.js',
  techniques:  '/static/techniques.js',
};

// ── Tab switching ───────────────────────────────────────────────────────────

async function activateTab(tabKey) {
  autotuneState.activeTab = tabKey;

  // Update tab button active state
  document.querySelectorAll('.tab').forEach(btn => {
    btn.classList.toggle('tab-active', btn.dataset.tab === tabKey);
  });

  // Show/hide content views
  document.querySelectorAll('.view').forEach(view => {
    view.classList.remove('view-active');
  });
  const container = document.getElementById(`view-${tabKey}`);
  if (container) container.classList.add('view-active');

  // Lazy load module and init on first activation
  if (!TAB_MODULES[tabKey]) {
    try {
      TAB_MODULES[tabKey] = await import(TAB_MODULE_URLS[tabKey]);
    } catch (err) {
      console.error(`Failed to load module for tab "${tabKey}":`, err);
      return;
    }
  }

  const mod = TAB_MODULES[tabKey];
  if (!TAB_INITIALIZED[tabKey] && mod.init) {
    mod.init(container);
    TAB_INITIALIZED[tabKey] = true;
  }

  if (mod.render) {
    mod.render(autotuneState);
  }
}

function initTabs() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
  });

  // Honor URL hash for initial tab
  const hash = window.location.hash.replace('#', '');
  const validTabs = Object.keys(TAB_MODULES);
  const initialTab = validTabs.includes(hash) ? hash : 'dag';
  activateTab(initialTab);
}

// ── SSE routing helpers ─────────────────────────────────────────────────────

const DAG_UPDATE_TABS  = ['dag', 'pareto', 'leaderboard', 'techniques'];
const MANIFOLD_TABS    = ['manifold'];

function renderTabs(tabKeys) {
  for (const key of tabKeys) {
    const mod = TAB_MODULES[key];
    if (mod && TAB_INITIALIZED[key] && mod.render) {
      mod.render(autotuneState);
    }
  }
}

// ── Stats bar ───────────────────────────────────────────────────────────────

function updateStats() {
  const meta = autotuneState.dag?.meta ?? {};
  const nodes = autotuneState.dag?.nodes ?? {};

  document.getElementById('stat-executed').textContent =
    meta.total_executed ?? '—';
  document.getElementById('stat-proposed').textContent =
    meta.total_proposed ?? '—';

  // Find best composite among executed+keep nodes
  let bestComposite = null;
  let bestNodeId = null;
  for (const [nid, node] of Object.entries(nodes)) {
    const c = node.composite ?? null;
    if (c !== null && (bestComposite === null || c > bestComposite)) {
      bestComposite = c;
      bestNodeId = nid;
    }
  }

  document.getElementById('stat-best').textContent =
    bestComposite !== null ? bestComposite.toFixed(4) : '—';
  document.getElementById('stat-best-node').textContent =
    bestNodeId ? `(${bestNodeId})` : '';
}

// ── Connection indicator ────────────────────────────────────────────────────

let _lastEventTime   = null;
let _connAgeInterval = null;
let _reconnecting    = false;

function setConnectionStatus(connected) {
  const dot   = document.getElementById('conn-dot');
  const label = document.getElementById('conn-label');
  if (!dot || !label) return;

  _reconnecting = !connected;
  dot.className = 'conn-dot ' + (
    connected     ? 'conn-connected' :
    _reconnecting ? 'conn-reconnecting' : 'conn-disconnected'
  );
  label.textContent = connected ? 'Connected' : 'Reconnecting…';

  if (connected) {
    _lastEventTime = Date.now();
    if (!_connAgeInterval) {
      _connAgeInterval = setInterval(() => {
        if (_lastEventTime) {
          const sec = Math.floor((Date.now() - _lastEventTime) / 1000);
          const el  = document.getElementById('conn-age');
          if (el) el.textContent = `${sec}s`;
        }
      }, 1000);
    }
  }
}

// ── SSE client ──────────────────────────────────────────────────────────────

let _sse     = null;
let _backoff = 1000; // ms

function connectSSE() {
  _sse = new EventSource('/events');

  _sse.onopen = () => {
    _backoff = 1000;
    setConnectionStatus(true);
  };

  _sse.onmessage = (evt) => {
    _lastEventTime = Date.now();
    let payload;
    try {
      payload = JSON.parse(evt.data);
    } catch {
      return;
    }

    if (payload.type === 'dag_update') {
      autotuneState.dag = payload.full_dag;
      updateStats();
      renderTabs(DAG_UPDATE_TABS);
    } else if (payload.type === 'manifold_update') {
      autotuneState.manifold = payload.manifold_state;
      renderTabs(MANIFOLD_TABS);
    }
  };

  _sse.onerror = () => {
    _sse.close();
    setConnectionStatus(false);
    // Exponential backoff, cap at 30s
    setTimeout(connectSSE, _backoff);
    _backoff = Math.min(_backoff * 2, 30_000);
  };
}

// ── Detail panel ────────────────────────────────────────────────────────────

export function showDetail(nodeId) {
  const dag = autotuneState.dag;
  if (!dag || !dag.nodes) return;
  const node = dag.nodes[nodeId];
  if (!node) return;

  autotuneState.selectedNode = nodeId;

  // Composite score (large)
  const comp = node.composite ?? null;
  document.getElementById('detail-composite').textContent =
    comp !== null ? comp.toFixed(4) : '—';

  // Node ID
  document.getElementById('detail-node-id').textContent = nodeId;

  // Status badge
  const badge = document.getElementById('detail-status-badge');
  const status = node.status ?? 'pending';
  const color  = STATUS_COLORS[status] ?? '#8b949e';
  badge.textContent = status;
  badge.style.color = color;
  badge.style.borderColor = `${color}40`;
  badge.style.backgroundColor = `${color}18`;

  // Move type
  document.getElementById('detail-move-type').textContent =
    node.move_type ?? '';

  // Metrics grid
  document.getElementById('dm-task').textContent =
    node.task_score != null ? node.task_score.toFixed(4) : '—';
  document.getElementById('dm-cost').textContent =
    node.cost      != null ? node.cost.toFixed(2)        : '—';
  document.getElementById('dm-prank').textContent =
    node.pareto_rank != null ? node.pareto_rank : '—';

  const retRow = document.getElementById('dm-retention-row');
  const retVal = document.getElementById('dm-retention');
  const hasRetention = node.retention_score != null;
  retRow.style.display = hasRetention ? '' : 'none';
  retVal.textContent = hasRetention ? node.retention_score.toFixed(4) : '—';

  // Parents (clickable)
  const parentsEl = document.getElementById('dm-parents');
  const parents   = node.parent_ids ?? [];
  parentsEl.innerHTML = '';
  if (parents.length === 0) {
    parentsEl.textContent = 'none';
  } else {
    parents.forEach(pid => {
      const a = document.createElement('span');
      a.className = 'parent-link';
      a.textContent = pid;
      a.addEventListener('click', () => showDetail(pid));
      parentsEl.appendChild(a);
    });
  }

  // Depth
  document.getElementById('dm-depth').textContent = node.depth ?? '—';

  // Technique tags
  const techEl = document.getElementById('dm-techniques');
  techEl.innerHTML = '';
  const techs = node.technique_tags ?? [];
  techs.forEach(t => {
    const chip = document.createElement('span');
    chip.className = 'tag-chip';
    chip.textContent = t;
    techEl.appendChild(chip);
  });

  // Theta
  const thetaEl = document.getElementById('dm-theta');
  thetaEl.innerHTML = '';
  const theta = node.theta ?? {};
  Object.entries(theta).forEach(([k, v]) => {
    const key = document.createElement('span');
    key.className = 'theta-key';
    key.textContent = k + ':';
    const val = document.createElement('span');
    val.className = 'theta-val';
    val.textContent = typeof v === 'number' && !Number.isInteger(v)
      ? v.toExponential(2) : String(v);
    thetaEl.appendChild(key);
    thetaEl.appendChild(val);
  });

  // Created at
  document.getElementById('dm-created').textContent =
    node.created_at
      ? new Date(node.created_at).toLocaleString()
      : '—';

  // Slide panel in
  const panel = document.getElementById('detail-panel');
  panel.classList.add('panel-open');
}

export function hideDetail() {
  const panel = document.getElementById('detail-panel');
  panel.classList.remove('panel-open');
  autotuneState.selectedNode = null;
}

// Expose globally for tab modules
window.showDetail = showDetail;
window.hideDetail = hideDetail;

// ── Bootstrap ───────────────────────────────────────────────────────────────

function init() {
  // Close panel on button click or backdrop click
  document.getElementById('detail-close')?.addEventListener('click', hideDetail);
  document.getElementById('content-area')?.addEventListener('click', (evt) => {
    // Close only if clicking the content area itself, not a tab view
    if (evt.target === evt.currentTarget) hideDetail();
  });

  initTabs();
  connectSSE();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
