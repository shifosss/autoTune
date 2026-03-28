/**
 * manifold.js — Manifold explorer: PC scatter, danger zones, overlays.
 *
 * Module contract:
 *   init(container)  — called once when tab first activated
 *   render(state)    — called on each manifold_update SSE event
 */

const PLOT_LAYOUT_BASE = {
  paper_bgcolor: '#0a0e1a',
  plot_bgcolor:  '#0a0e1a',
  font:          { color: '#8b949e', family: 'monospace', size: 11 },
  margin:        { t: 20, r: 20, b: 50, l: 60 },
  xaxis: {
    title:       { text: 'PC1', font: { color: '#8b949e', size: 11 } },
    gridcolor:   'rgba(255,255,255,0.05)',
    zerolinecolor: 'rgba(255,255,255,0.1)',
  },
  yaxis: {
    title:       { text: 'PC2', font: { color: '#8b949e', size: 11 } },
    gridcolor:   'rgba(255,255,255,0.05)',
    zerolinecolor: 'rgba(255,255,255,0.1)',
  },
  showlegend:  false,
  hovermode:   'closest',
};

const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

// Composite color gradient: low → high
const COMPOSITE_COLORSCALE = [
  [0, '#1a3a2a'],
  [1, '#3fb950'],
];

let _container      = null;
let _chartDiv       = null;
let _sidebar        = null;
let _eigenvecCheck  = null;
let _ellipseCheck   = null;
let _genLabel       = null;
let _initialized    = false;

export function init(container) {
  _container = container;
  _container.innerHTML = '';

  // Controls bar
  const controls = document.createElement('div');
  controls.className = 'manifold-controls';

  _eigenvecCheck = _makeCheckbox('Eigenvectors', controls);
  _ellipseCheck  = _makeCheckbox('Step ellipse', controls);

  _genLabel = document.createElement('span');
  _genLabel.style.cssText = 'margin-left:auto;color:var(--text-secondary);font-size:11px;';
  controls.appendChild(_genLabel);

  _container.appendChild(controls);

  // Body: chart + sidebar
  const body = document.createElement('div');
  body.className = 'manifold-body';

  const chartArea = document.createElement('div');
  chartArea.className = 'manifold-chart-area';

  _chartDiv = document.createElement('div');
  _chartDiv.style.cssText = 'width:100%;height:100%;';
  chartArea.appendChild(_chartDiv);
  body.appendChild(chartArea);

  _sidebar = document.createElement('div');
  _sidebar.className = 'manifold-sidebar';
  body.appendChild(_sidebar);

  _container.appendChild(body);
  _initialized = true;
}

function _makeCheckbox(label, parent) {
  const lbl = document.createElement('label');
  lbl.className = 'controls-label';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.addEventListener('change', () => render(window.autotuneState));
  lbl.appendChild(cb);
  lbl.appendChild(document.createTextNode(' ' + label));
  parent.appendChild(lbl);
  return cb;
}

/**
 * Project a theta dict onto the top-2 principal components.
 *
 * @param {Object} theta - hyperparameter key→value map
 * @param {Object} manifold - manifold_state payload
 * @returns {[number, number]} [pc1, pc2] coordinates
 */
function projectToPC(theta, manifold) {
  const paramNames = manifold.param_names ?? [];
  const pcs        = manifold.principal_components ?? [];
  const mean       = manifold.mean ?? paramNames.map(() => 0);

  if (pcs.length < 2 || paramNames.length === 0) return [0, 0];

  // Build normalized parameter vector
  const vec = paramNames.map((name, i) => (theta[name] ?? 0) - mean[i]);

  const dot = (row) => row.reduce((s, w, i) => s + w * (vec[i] ?? 0), 0);

  return [dot(pcs[0]), dot(pcs[1])];
}

export function render(state) {
  if (!_initialized || !_chartDiv) return;

  const manifold = state?.manifold;
  const dag      = state?.dag;

  // Pre-manifold: not yet learned
  if (!manifold || !manifold.generation || manifold.generation === 0) {
    Plotly.purge(_chartDiv);
    _chartDiv.innerHTML =
      '<div class="manifold-not-learned">Manifold not yet learned<br><span style="font-size:11px;margin-top:6px;display:block">(need 10+ evaluated experiments)</span></div>';
    _renderSidebar(null);
    if (_genLabel) _genLabel.textContent = 'Generation: 0';
    return;
  }

  if (_genLabel) {
    _genLabel.textContent = `Generation: ${manifold.generation}`;
  }

  // Collect executed nodes and project
  const nodes    = dag?.nodes ?? {};
  const executed = Object.entries(nodes).filter(
    ([, n]) => n.status && n.status !== 'pending' && n.status !== 'running'
  );

  const xs         = [];
  const ys         = [];
  const composites = [];
  const hoverTexts = [];
  const nodeIds    = [];

  executed.forEach(([id, n]) => {
    if (!n.theta) return;
    const [pc1, pc2] = projectToPC(n.theta, manifold);
    xs.push(pc1);
    ys.push(pc2);
    composites.push(n.composite ?? 0);
    nodeIds.push(id);
    const parts = [`<b>${id}</b>`, `composite: ${(n.composite ?? 0).toFixed(4)}`];
    Object.entries(n.theta).forEach(([k, v]) => parts.push(`${k}: ${v}`));
    hoverTexts.push(parts.join('<br>'));
  });

  const scatterTrace = {
    type:      'scatter',
    mode:      'markers',
    x:         xs,
    y:         ys,
    hovertext: hoverTexts,
    hoverinfo: 'text',
    customdata: nodeIds,
    marker: {
      size:       8,
      color:      composites,
      colorscale: COMPOSITE_COLORSCALE,
      cmin:       0,
      cmax:       1,
      line:       { color: 'rgba(255,255,255,0.15)', width: 0.5 },
    },
  };

  const traces = [scatterTrace];
  const shapes = [];
  const annotations = [];

  // Danger zones (drawn as Plotly shapes)
  const dangerZones = manifold.danger_zones ?? [];
  dangerZones.forEach(zone => {
    const cx = zone.center?.[0] ?? 0;
    const cy = zone.center?.[1] ?? 0;
    const r  = zone.radius ?? 0.5;
    shapes.push({
      type:      'circle',
      xref:      'x',
      yref:      'y',
      x0:        cx - r,
      y0:        cy - r,
      x1:        cx + r,
      y1:        cy + r,
      fillcolor: 'rgba(248,81,73,0.15)',
      line:      { color: 'rgba(248,81,73,0.4)', width: 1 },
    });
  });

  const mean     = manifold.mean ?? [];
  const pcs      = manifold.principal_components ?? [];
  const eigvals  = manifold.eigenvalues ?? [];
  const stepSizes = manifold.step_sizes ?? [];

  // Eigenvectors overlay
  if (_eigenvecCheck?.checked && pcs.length >= 2) {
    const mean0 = mean[0] ?? 0;
    const mean1 = mean[1] ?? 0;
    const [mx, my] = [
      projectToPC({ [manifold.param_names?.[0]]: mean0 }, manifold)[0] ?? 0,
      projectToPC({ [manifold.param_names?.[1]]: mean1 }, manifold)[1] ?? 0,
    ];

    pcs.slice(0, 3).forEach((pc, i) => {
      const scale  = (eigvals[i] ?? 0.5) * 2;
      const ax     = (pc[0] ?? 0) * scale;
      const ay     = (pc[1] ?? 0) * scale;
      const params = manifold.param_names ?? [];
      const topTwo = [...params.keys()]
        .sort((a, b) => Math.abs(pc[b] ?? 0) - Math.abs(pc[a] ?? 0))
        .slice(0, 2)
        .map(idx => `${params[idx]}${(pc[idx] ?? 0) >= 0 ? '↑' : '↓'}`)
        .join(' ');
      annotations.push({
        x: mx + ax,
        y: my + ay,
        ax: mx,
        ay: my,
        xref: 'x', yref: 'y',
        axref: 'x', ayref: 'y',
        text: `PC${i + 1}: ${topTwo}`,
        showarrow: true,
        arrowhead: 2,
        arrowcolor: '#a5d6ff',
        font: { color: '#a5d6ff', size: 10 },
      });
    });
  }

  // Step ellipse overlay
  if (_ellipseCheck?.checked && stepSizes.length >= 2) {
    const cx = mean[0] ?? 0;
    const cy = mean[1] ?? 0;
    shapes.push({
      type:      'circle',
      xref:      'x',
      yref:      'y',
      x0:        cx - stepSizes[0],
      y0:        cy - stepSizes[1],
      x1:        cx + stepSizes[0],
      y1:        cy + stepSizes[1],
      fillcolor: 'rgba(0,0,0,0)',
      line:      { color: 'rgba(255,255,255,0.5)', width: 1, dash: 'dot' },
    });
  }

  const layout = {
    ...PLOT_LAYOUT_BASE,
    shapes,
    annotations,
  };

  Plotly.react(_chartDiv, traces, layout, PLOTLY_CONFIG);
  _chartDiv.removeAllListeners?.('plotly_click');
  _chartDiv.on('plotly_click', (data) => {
    const pt = data?.points?.[0];
    if (pt?.customdata) window.showDetail(pt.customdata);
  });

  _renderSidebar(manifold);
}

function _renderSidebar(manifold) {
  if (!_sidebar) return;
  _sidebar.innerHTML = '';

  const h3 = document.createElement('h3');
  h3.textContent = 'PC Summary';
  _sidebar.appendChild(h3);

  if (!manifold) {
    _sidebar.appendChild(_text('Manifold not learned', 'var(--text-secondary)'));
    return;
  }

  const paramNames = manifold.param_names ?? [];
  const pcs        = manifold.principal_components ?? [];
  const eigvals    = manifold.eigenvalues ?? [];
  const totalEig   = eigvals.reduce((s, v) => s + v, 0) || 1;

  // Top PCs
  pcs.slice(0, 3).forEach((pc, i) => {
    const div = document.createElement('div');
    div.className = 'pc-item';

    const loadings = [...paramNames.keys()]
      .sort((a, b) => Math.abs(pc[b] ?? 0) - Math.abs(pc[a] ?? 0))
      .slice(0, 2)
      .map(idx => `${paramNames[idx]}${(pc[idx] ?? 0) >= 0 ? '↑' : '↓'}`)
      .join(' ');

    const pct = eigvals[i] != null
      ? ((eigvals[i] / totalEig) * 100).toFixed(0) + '%'
      : '—';

    div.innerHTML = `<span class="pc-label">PC${i + 1}</span>: ${loadings} <span style="color:var(--text-muted)">(${pct})</span>`;
    _sidebar.appendChild(div);
  });

  // Parameter importance
  const impH = document.createElement('h3');
  impH.textContent = 'Importance';
  impH.style.marginTop = '16px';
  _sidebar.appendChild(impH);

  // Compute total absolute weight per param across top PCs
  const importance = {};
  paramNames.forEach(name => { importance[name] = 0; });
  pcs.slice(0, 3).forEach((pc, i) => {
    const weight = eigvals[i] ?? 0;
    pc.forEach((v, j) => {
      if (paramNames[j]) importance[paramNames[j]] += Math.abs(v) * weight;
    });
  });

  const totalImp = Object.values(importance).reduce((s, v) => s + v, 0) || 1;
  const sorted   = Object.entries(importance).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([name, val]) => {
    const pct  = ((val / totalImp) * 100).toFixed(0);
    const item = document.createElement('div');
    item.className = 'pc-item';
    item.innerHTML = `<span style="color:var(--text-primary)">${name}</span>: ${pct}%`;
    const bar  = document.createElement('div');
    bar.className = 'importance-bar';
    const fill = document.createElement('div');
    fill.className = 'importance-fill';
    fill.style.width = `${pct}%`;
    bar.appendChild(fill);
    item.appendChild(bar);
    _sidebar.appendChild(item);
  });
}

function _text(str, color) {
  const span = document.createElement('span');
  span.textContent = str;
  span.style.color = color ?? 'var(--text-primary)';
  return span;
}
