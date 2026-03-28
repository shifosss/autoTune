/**
 * pareto.js — Pareto frontier tab with Plotly scatter and projection selector.
 *
 * Module contract:
 *   init(container)  — called once when tab first activated
 *   render(state)    — called on each dag_update SSE event
 */

const PROJECTIONS_3OBJ = [
  { label: 'Task vs Retention',   xKey: 'task_score',      yKey: 'retention_score',  sizeKey: 'cost'      },
  { label: 'Task vs Cost',        xKey: 'task_score',      yKey: 'cost',             sizeKey: 'retention_score' },
  { label: 'Retention vs Cost',   xKey: 'retention_score', yKey: 'cost',             sizeKey: 'task_score' },
];
const PROJECTIONS_2OBJ = [
  { label: 'Task vs Cost',        xKey: 'task_score',      yKey: 'cost',             sizeKey: null },
];

const PARETO_COLORSCALE = [
  [0,   '#e3b341'],  // rank 0 — gold
  [0.5, '#a5d6ff'],  // rank 1 — pale blue
  [1,   '#6e7681'],  // rank 2+ — muted gray
];

const PLOT_LAYOUT_BASE = {
  paper_bgcolor: '#0a0e1a',
  plot_bgcolor:  '#0a0e1a',
  font:          { color: '#8b949e', family: 'monospace', size: 11 },
  margin:        { t: 20, r: 20, b: 50, l: 60 },
  xaxis: {
    gridcolor:   'rgba(255,255,255,0.05)',
    zerolinecolor: 'rgba(255,255,255,0.1)',
    tickfont:    { color: '#8b949e' },
  },
  yaxis: {
    gridcolor:   'rgba(255,255,255,0.05)',
    zerolinecolor: 'rgba(255,255,255,0.1)',
    tickfont:    { color: '#8b949e' },
  },
  showlegend: false,
  hovermode:  'closest',
};

const PLOTLY_CONFIG = {
  displayModeBar: false,
  responsive:     true,
};

let _container      = null;
let _chartDiv       = null;
let _select         = null;
let _frontierCheck  = null;
let _is2Obj         = false;
let _initialized    = false;

export function init(container) {
  _container = container;
  _container.innerHTML = '';

  // Controls bar
  const controls = document.createElement('div');
  controls.className = 'pareto-controls';

  const selLabel = document.createElement('label');
  selLabel.textContent = 'Projection: ';
  selLabel.style.color = 'var(--text-secondary)';
  selLabel.style.fontSize = '12px';

  _select = document.createElement('select');
  _select.className = 'controls-select';
  _select.addEventListener('change', () => render(window.autotuneState));
  selLabel.appendChild(_select);
  controls.appendChild(selLabel);

  const checkLabel = document.createElement('label');
  checkLabel.className = 'controls-label';
  _frontierCheck = document.createElement('input');
  _frontierCheck.type = 'checkbox';
  _frontierCheck.addEventListener('change', () => render(window.autotuneState));
  checkLabel.appendChild(_frontierCheck);
  checkLabel.appendChild(document.createTextNode(' Show frontier line'));
  controls.appendChild(checkLabel);

  _container.appendChild(controls);

  // Chart area
  _chartDiv = document.createElement('div');
  _chartDiv.id = 'pareto-chart';
  _chartDiv.style.cssText = 'flex:1;min-height:0;';
  _container.appendChild(_chartDiv);
  _initialized = true;
}

function _buildProjections(is2Obj) {
  const projs = is2Obj ? PROJECTIONS_2OBJ : PROJECTIONS_3OBJ;
  _select.innerHTML = '';
  projs.forEach((p, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = p.label;
    _select.appendChild(opt);
  });
}

export function render(state) {
  if (!_initialized || !_chartDiv) return;

  const dag = state?.dag;
  if (!dag?.nodes) {
    Plotly.purge(_chartDiv);
    _chartDiv.innerHTML = '<div class="empty-state"><div class="empty-state-title">No experiments yet</div><div class="empty-state-sub">Run some experiments to see the Pareto frontier.</div></div>';
    return;
  }

  const is2Obj = dag.meta?.objective_count === 2;

  // Rebuild dropdown if mode changed
  if (is2Obj !== _is2Obj) {
    _is2Obj = is2Obj;
    _buildProjections(is2Obj);
  }

  const projIdx  = parseInt(_select.value || '0', 10);
  const projs    = is2Obj ? PROJECTIONS_2OBJ : PROJECTIONS_3OBJ;
  const proj     = projs[projIdx] ?? projs[0];

  // Collect executed nodes
  const executed = Object.entries(dag.nodes).filter(
    ([, n]) => n.status && n.status !== 'pending' && n.status !== 'running'
  );

  const xs       = [];
  const ys       = [];
  const sizes    = [];
  const colors   = [];
  const texts    = [];
  const nodeIds  = [];
  const hoverParts = [];

  executed.forEach(([id, n]) => {
    const x = n[proj.xKey]    ?? null;
    const y = n[proj.yKey]    ?? null;
    if (x == null || y == null) return;

    const sizeRaw = proj.sizeKey ? (n[proj.sizeKey] ?? 1) : 1;
    // Normalize rank to [0,1] for colorscale (cap at rank 2)
    const rank      = n.pareto_rank ?? 3;
    const rankNorm  = Math.min(rank, 2) / 2;

    xs.push(x);
    ys.push(y);
    sizes.push(is2Obj ? 10 : Math.max(4, 20 / (sizeRaw || 1)));
    colors.push(rankNorm);
    texts.push(id);
    nodeIds.push(id);

    hoverParts.push(
      `<b>${id}</b><br>` +
      `${proj.xKey}: ${x.toFixed(4)}<br>` +
      `${proj.yKey}: ${y.toFixed(4)}<br>` +
      (proj.sizeKey ? `${proj.sizeKey}: ${sizeRaw.toFixed(4)}<br>` : '') +
      `rank: ${rank}<br>` +
      `type: ${n.move_type ?? '—'}`
    );
  });

  const scatterTrace = {
    type:         'scatter',
    mode:         'markers',
    x:            xs,
    y:            ys,
    text:         nodeIds,
    hovertext:    hoverParts,
    hoverinfo:    'text',
    marker: {
      size:         sizes,
      color:        colors,
      colorscale:   PARETO_COLORSCALE,
      cmin:         0,
      cmax:         1,
      line:         { color: 'rgba(255,255,255,0.15)', width: 0.5 },
    },
    customdata: nodeIds,
  };

  const traces = [scatterTrace];

  // Frontier line
  if (_frontierCheck?.checked) {
    const rank0 = executed
      .filter(([, n]) => n.pareto_rank === 0)
      .map(([, n]) => ({ x: n[proj.xKey], y: n[proj.yKey] }))
      .filter(p => p.x != null && p.y != null)
      .sort((a, b) => a.x - b.x);

    if (rank0.length > 1) {
      traces.push({
        type:  'scatter',
        mode:  'lines',
        x:     rank0.map(p => p.x),
        y:     rank0.map(p => p.y),
        hoverinfo: 'none',
        line:  { color: '#e3b341', dash: 'dot', width: 1.5 },
      });
    }
  }

  const layout = {
    ...PLOT_LAYOUT_BASE,
    xaxis: { ...PLOT_LAYOUT_BASE.xaxis, title: { text: proj.xKey, font: { color: '#8b949e', size: 11 } } },
    yaxis: { ...PLOT_LAYOUT_BASE.yaxis, title: { text: proj.yKey, font: { color: '#8b949e', size: 11 } } },
  };

  Plotly.react(_chartDiv, traces, layout, PLOTLY_CONFIG);

  // Click handler — remove previous before reattaching
  _chartDiv.removeAllListeners?.('plotly_click');
  _chartDiv.on('plotly_click', _clickHandler);
}

function _clickHandler(data) {
  const pt = data?.points?.[0];
  if (pt?.customdata) {
    window.showDetail(pt.customdata);
  }
}
