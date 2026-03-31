/**
 * dag-view.js — DAG View tab: 2D layered d3 layout + 3D ForceGraph toggle.
 *
 * Module contract:
 *   init(container)  — called once when tab first activated
 *   render(state)    — called on each dag_update SSE event
 *   destroy()        — cleanup simulation/graph on tab deactivate
 */

let _container    = null;
let _is3D         = false;
let _simulation   = null;   // d3 force simulation (2D mode)
let _graph3D      = null;   // ForceGraph3D instance (3D mode)
let _svgEl        = null;
let _div3D        = null;
let _toggleBtn    = null;
let _initialized  = false;

// ── init ─────────────────────────────────────────────────────────────────────

export function init(container) {
  _container = container;
  _container.style.position = 'relative';
  _container.innerHTML = '';

  // Controls overlay
  const controls = document.createElement('div');
  controls.className = 'dag-controls';
  _toggleBtn = document.createElement('button');
  _toggleBtn.className = 'dag-toggle-btn';
  _toggleBtn.textContent = 'Switch to 3D';
  _toggleBtn.addEventListener('click', _onToggle);
  controls.appendChild(_toggleBtn);
  _container.appendChild(controls);

  // 2D SVG container
  _svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  _svgEl.id = 'dag-svg-container';
  _svgEl.style.cssText = 'width:100%;height:100%;display:block;';
  _container.appendChild(_svgEl);

  // 3D div (hidden by default)
  _div3D = document.createElement('div');
  _div3D.id = 'dag-3d-container';
  _div3D.style.cssText = 'width:100%;height:100%;display:none;';
  _container.appendChild(_div3D);

  _initialized = true;
}

// ── render ────────────────────────────────────────────────────────────────────

export function render(state) {
  if (!_initialized) return;

  const dag = state?.dag;
  if (!dag?.nodes || Object.keys(dag.nodes).length === 0) {
    _clearView();
    _showEmptyState();
    return;
  }

  const { nodes, links, spineEdges } = _buildGraphData(dag);

  if (_is3D) {
    _render3D(nodes, links, spineEdges);
  } else {
    _render2D(nodes, links, spineEdges, dag);
  }
}

// ── destroy ───────────────────────────────────────────────────────────────────

export function destroy() {
  _clearView();
}

// ── toggle 2D / 3D ──────────────────────────────────────────────────────────

function _onToggle() {
  _is3D = !_is3D;
  _toggleBtn.textContent = _is3D ? 'Switch to 2D' : 'Switch to 3D';
  _clearView();
  render(window.autotuneState);
}

function _clearView() {
  if (_simulation) {
    _simulation.stop();
    _simulation = null;
  }
  if (_graph3D) {
    _graph3D._destructor?.();
    _graph3D = null;
  }
  if (_svgEl) {
    while (_svgEl.firstChild) _svgEl.removeChild(_svgEl.firstChild);
  }
  if (_div3D) {
    _div3D.innerHTML = '';
  }
  _svgEl.style.display  = _is3D ? 'none' : 'block';
  _div3D.style.display  = _is3D ? 'block' : 'none';
}

// ── data prep ─────────────────────────────────────────────────────────────────

function _buildGraphData(dag) {
  const nodes = Object.entries(dag.nodes).map(([id, n]) => ({
    id,
    depth:     n.depth       ?? 0,
    composite: n.composite   ?? 0,
    status:    n.status      ?? 'pending',
    move_type: n.move_type   ?? 'branch',
    parent_ids: n.parent_ids ?? [],
    technique_tags: n.technique_tags ?? [],
    pareto_rank: n.pareto_rank ?? null,
  }));

  // Build edges
  const links = [];
  nodes.forEach(node => {
    node.parent_ids.forEach(pid => {
      links.push({
        source: pid,
        target: node.id,
        is_graft: node.move_type === 'graft',
      });
    });
  });

  // Compute Pareto spine: ancestors of all rank-0 nodes
  const rank0Ids = new Set(nodes.filter(n => n.pareto_rank === 0).map(n => n.id));
  const spineNodes = new Set(rank0Ids);

  // BFS upward from rank-0 nodes
  const parentMap = {};
  nodes.forEach(n => { parentMap[n.id] = n.parent_ids; });
  const queue = [...rank0Ids];
  while (queue.length) {
    const nid = queue.shift();
    (parentMap[nid] ?? []).forEach(pid => {
      if (!spineNodes.has(pid)) {
        spineNodes.add(pid);
        queue.push(pid);
      }
    });
  }

  const spineEdges = new Set(
    links
      .filter(l => spineNodes.has(l.source) && spineNodes.has(l.target))
      .map(l => `${l.source}→${l.target}`)
  );

  return { nodes, links, spineEdges };
}

// ── 2D rendering ─────────────────────────────────────────────────────────────

function _render2D(nodes, links, spineEdges, dag) {
  const svg    = d3.select(_svgEl);
  const width  = _container.clientWidth  || 800;
  const height = _container.clientHeight || 600;

  svg.attr('viewBox', `0 0 ${width} ${height}`);

  // Arrowhead marker
  let defs = svg.select('defs');
  if (defs.empty()) defs = svg.append('defs');
  defs.html(`
    <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
      <polygon points="0 0, 6 2, 0 4" fill="#30363d"/>
    </marker>
    <marker id="arrowhead-spine" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
      <polygon points="0 0, 6 2, 0 4" fill="#e3b341"/>
    </marker>
  `);

  const g = svg.append('g');

  // Zoom/pan
  svg.call(
    d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (evt) => g.attr('transform', evt.transform))
  );

  // Assign initial positions
  const nodeMap = {};
  const depthCounts = {};
  nodes.forEach(n => {
    depthCounts[n.depth] = (depthCounts[n.depth] ?? 0) + 1;
  });
  const depthPos = {};
  nodes.forEach(n => {
    depthPos[n.depth] = (depthPos[n.depth] ?? 0);
  });
  nodes.forEach(n => {
    n.x = width / 2 + (Math.random() - 0.5) * 200;
    n.y = 60 + n.depth * 120;
    nodeMap[n.id] = n;
  });

  // Force simulation
  if (_simulation) _simulation.stop();

  const simLinks = links.map(l => ({
    source: nodeMap[l.source] ?? l.source,
    target: nodeMap[l.target] ?? l.target,
    ...l,
  }));

  _simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(simLinks).id(d => d.id).distance(80).strength(0.5))
    .force('charge', d3.forceManyBody().strength(-100))
    .force('x', d3.forceX(width / 2).strength(0.05))
    .force('y', d3.forceY(d => 60 + d.depth * 120).strength(1.5))
    .force('collision', d3.forceCollide(20))
    .alphaDecay(0.02);

  // Edges
  const linkEls = g.append('g').selectAll('line')
    .data(simLinks)
    .enter().append('line')
    .attr('stroke', l => spineEdges.has(`${l.source.id}→${l.target.id}`) ? '#e3b341' : '#30363d')
    .attr('stroke-width', l => spineEdges.has(`${l.source.id}→${l.target.id}`) ? 2.5 : 1)
    .attr('stroke-dasharray', l => l.is_graft ? '4,4' : null)
    .attr('marker-end', l =>
      spineEdges.has(`${l.source.id}→${l.target.id}`) ? 'url(#arrowhead-spine)' : 'url(#arrowhead)'
    );

  // Node shapes
  const nodeEls = g.append('g').selectAll('g')
    .data(nodes)
    .enter().append('g')
    .style('cursor', 'pointer')
    .on('click', (evt, d) => { evt.stopPropagation(); window.showDetail(d.id); });

  // Merge nodes: rotated rect; others: circle
  nodeEls.each(function(d) {
    const el = d3.select(this);
    const r = Math.max(4, 6 + (d.composite ?? 0) * 12);
    const color = window.STATUS_COLORS?.[d.status] ?? '#8b949e';

    if (d.move_type === 'merge') {
      el.append('rect')
        .attr('width', r * 1.5)
        .attr('height', r * 1.5)
        .attr('x', -(r * 0.75))
        .attr('y', -(r * 0.75))
        .attr('transform', 'rotate(45)')
        .attr('fill', color)
        .attr('stroke', '#0a0e1a')
        .attr('stroke-width', 1.5);
    } else {
      el.append('circle')
        .attr('r', r)
        .attr('fill', color)
        .attr('stroke', '#0a0e1a')
        .attr('stroke-width', 1.5)
        .classed('status-running-anim', d.status === 'running');
    }
  });

  // Simulation tick
  _simulation.on('tick', () => {
    linkEls
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    nodeEls.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}

// ── 3D rendering ──────────────────────────────────────────────────────────────

function _render3D(nodes, links, spineEdges) {
  if (!window.ForceGraph3D) {
    _div3D.innerHTML = '<div class="empty-state"><div class="empty-state-title">ForceGraph3D not loaded</div></div>';
    return;
  }

  if (_graph3D) {
    _graph3D.graphData({ nodes, links });
    return;
  }

  _graph3D = ForceGraph3D()(_div3D)
    .backgroundColor('#0a0e1a')
    .dagMode('td')
    .dagLevelDistance(80)
    .nodeId('id')
    .nodeVal(n => 5 + (n.composite ?? 0) * 20)
    .nodeColor(n => window.STATUS_COLORS?.[n.status] ?? '#8b949e')
    .nodeThreeObject(n => {
      if (n.move_type === 'merge') {
        const geo  = new THREE.OctahedronGeometry(6 + (n.composite ?? 0) * 12);
        const mat  = new THREE.MeshLambertMaterial({ color: window.STATUS_COLORS?.[n.status] ?? '#8b949e' });
        return new THREE.Mesh(geo, mat);
      }
      if (n.status === 'keep' && window.SpriteText) {
        const sprite = new SpriteText(n.id.slice(-6));
        sprite.color = '#e6edf3';
        sprite.textHeight = 6;
        return sprite;
      }
      return null;
    })
    .nodeThreeObjectExtend(true)
    .linkColor(l => spineEdges.has(`${l.source?.id ?? l.source}→${l.target?.id ?? l.target}`) ? '#e3b341' : '#30363d')
    .linkWidth(l => spineEdges.has(`${l.source?.id ?? l.source}→${l.target?.id ?? l.target}`) ? 3 : 1)
    .linkDirectionalParticles(l => spineEdges.has(`${l.source?.id ?? l.source}→${l.target?.id ?? l.target}`) ? 2 : 0)
    .linkDirectionalParticleSpeed(0.006)
    .linkLineDash(l => l.is_graft ? [2, 2] : null)
    .onNodeClick(n => window.showDetail(n.id))
    .graphData({ nodes, links });

  // Auto-rotate
  let autoRotate = true;
  _div3D.addEventListener('click', () => { autoRotate = false; });
  const animate = () => {
    if (_graph3D && autoRotate) {
      _graph3D.cameraPosition(
        { x: _graph3D.cameraPosition().x },
        undefined,
        0
      );
    }
  };
  setInterval(animate, 33);
}

// ── helpers ───────────────────────────────────────────────────────────────────

function _showEmptyState() {
  const el = document.createElement('div');
  el.className = 'empty-state';
  el.innerHTML = '<div class="empty-state-title">No experiments yet</div><div class="empty-state-sub">The DAG will appear here as experiments are run.</div>';
  _container.appendChild(el);
}
