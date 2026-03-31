/**
 * leaderboard.js — Sortable top-N table of all executed experiments.
 *
 * Module contract:
 *   init(container)  — called once when tab first activated
 *   render(state)    — called on each dag_update SSE event
 */

const DEFAULT_LIMIT = 20;

let _container    = null;
let _tbody        = null;
let _sortKey      = 'composite';
let _sortDir      = 'desc';
let _showAll      = false;
let _showAllBtn   = null;

// Column definitions
const COLUMNS = [
  { key: 'rank',        label: 'Rank',      width: '50px',  sortable: false, retention: false },
  { key: 'id',          label: 'Node',      width: '90px',  sortable: true,  retention: false },
  { key: 'task_score',  label: 'Task',      width: '70px',  sortable: true,  retention: false },
  { key: 'retention',   label: 'Retention', width: '70px',  sortable: true,  retention: true  },
  { key: 'cost',        label: 'Cost',      width: '60px',  sortable: true,  retention: false },
  { key: 'composite',   label: 'Composite', width: '80px',  sortable: true,  retention: false },
  { key: 'pareto_rank', label: 'P.Rank',    width: '60px',  sortable: true,  retention: false },
  { key: 'move_type',   label: 'Type',      width: '70px',  sortable: false, retention: false },
  { key: 'status',      label: 'Status',    width: '80px',  sortable: false, retention: false },
];

export function init(container) {
  _container = container;
  _container.innerHTML = '';

  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'display:flex;flex-direction:column;height:100%;';

  const controls = document.createElement('div');
  controls.className = 'table-controls';

  _showAllBtn = document.createElement('button');
  _showAllBtn.className = 'show-all-btn';
  _showAllBtn.textContent = `Show all`;
  _showAllBtn.addEventListener('click', () => {
    _showAll = !_showAll;
    render(window.autotuneState);
  });
  controls.appendChild(_showAllBtn);
  wrapper.appendChild(controls);

  const scrollWrap = document.createElement('div');
  scrollWrap.style.cssText = 'flex:1;overflow-y:auto;';

  const table = document.createElement('table');
  table.className = 'data-table';

  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  COLUMNS.forEach(col => {
    const th = document.createElement('th');
    th.textContent = col.label;
    th.style.width = col.width;
    th.dataset.key = col.key;
    if (col.retention) th.classList.add('retention-col');
    if (col.sortable) {
      th.addEventListener('click', () => {
        if (_sortKey === col.key) {
          _sortDir = _sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          _sortKey = col.key;
          _sortDir = 'desc';
        }
        render(window.autotuneState);
      });
    }
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  _tbody = document.createElement('tbody');
  table.appendChild(_tbody);
  scrollWrap.appendChild(table);
  wrapper.appendChild(scrollWrap);
  _container.appendChild(wrapper);
}

export function render(state) {
  if (!_tbody) return;

  const dag = state?.dag;
  if (!dag?.nodes) {
    _tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--text-secondary);padding:40px">No experiments yet</td></tr>';
    return;
  }

  const is2Obj = dag.meta?.objective_count === 2;

  // Toggle retention columns
  _container.querySelectorAll('.retention-col').forEach(el => {
    el.style.display = is2Obj ? 'none' : '';
  });

  // Collect executed nodes
  const rows = Object.entries(dag.nodes)
    .filter(([, n]) => n.status && n.status !== 'pending')
    .map(([id, n]) => ({
      id,
      task_score:  n.task_score  ?? null,
      retention:   n.retention_score ?? null,
      cost:        n.cost        ?? null,
      composite:   n.composite   ?? null,
      pareto_rank: n.pareto_rank ?? null,
      move_type:   n.move_type   ?? '—',
      status:      n.status      ?? 'pending',
    }));

  // Sort
  rows.sort((a, b) => {
    const av = a[_sortKey] ?? -Infinity;
    const bv = b[_sortKey] ?? -Infinity;
    if (av < bv) return _sortDir === 'asc' ? -1 :  1;
    if (av > bv) return _sortDir === 'asc' ?  1 : -1;
    return 0;
  });

  // Update sort indicators on headers
  _container.querySelectorAll('th[data-key]').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.key === _sortKey) {
      th.classList.add(_sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
    }
  });

  const limit   = _showAll ? rows.length : DEFAULT_LIMIT;
  const visible = rows.slice(0, limit);

  // Render rows
  _tbody.innerHTML = '';
  visible.forEach((row, idx) => {
    const tr = document.createElement('tr');
    tr.addEventListener('click', () => window.showDetail(row.id));

    const cells = [
      String(idx + 1),
      row.id,
      row.task_score  != null ? row.task_score.toFixed(4) : '—',
      row.retention   != null ? row.retention.toFixed(4)  : '—',
      row.cost        != null ? row.cost.toFixed(2)        : '—',
      row.composite   != null ? row.composite.toFixed(4)   : '—',
      row.pareto_rank != null ? String(row.pareto_rank)    : '—',
      row.move_type,
      null, // status — special rendering
    ];

    COLUMNS.forEach((col, ci) => {
      const td = document.createElement('td');
      if (col.retention) td.classList.add('retention-col');

      if (col.key === 'status') {
        const dot = document.createElement('span');
        dot.className = 'status-dot';
        const color = window.STATUS_COLORS?.[row.status] ?? '#8b949e';
        dot.style.background = color;
        if (row.status === 'running') dot.classList.add('status-running-anim');
        td.appendChild(dot);
        td.appendChild(document.createTextNode(row.status));
      } else {
        td.textContent = cells[ci];
      }
      tr.appendChild(td);
    });

    _tbody.appendChild(tr);
  });

  // Update show-all button
  if (_showAllBtn) {
    if (rows.length <= DEFAULT_LIMIT) {
      _showAllBtn.style.display = 'none';
    } else {
      _showAllBtn.style.display = '';
      _showAllBtn.textContent = _showAll
        ? `Show top ${DEFAULT_LIMIT}`
        : `Show all (${rows.length})`;
    }
  }
}
