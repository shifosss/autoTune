/**
 * techniques.js — Technique effectiveness table with color-coded verdicts.
 *
 * Module contract:
 *   init(container)  — called once when tab first activated
 *   render(state)    — called on each dag_update SSE event
 */

let _container = null;
let _tbody     = null;
let _sortKey   = 'times_tried';
let _sortDir   = 'desc';

const COLUMNS = [
  { key: 'technique',            label: 'Technique',          width: '120px', sortable: true,  retention: false },
  { key: 'times_tried',          label: 'Tried',              width: '50px',  sortable: true,  retention: false },
  { key: 'avg_task_delta',       label: 'Avg Task Δ',         width: '90px',  sortable: true,  retention: false },
  { key: 'avg_retention_delta',  label: 'Avg Retention Δ',   width: '100px', sortable: true,  retention: true  },
  { key: 'best_parent_delta',    label: 'Best Parent Δ',     width: '100px', sortable: true,  retention: false },
  { key: 'verdict',              label: 'Verdict',            width: '80px',  sortable: false, retention: false },
];

function _fmtDelta(v) {
  if (v == null) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(3);
}

function _deltaClass(v) {
  if (v == null) return '';
  return v > 0 ? 'delta-pos' : 'delta-neg';
}

function _deltaTextClass(v) {
  if (v == null) return '';
  return v > 0 ? 'delta-pos-text' : 'delta-neg-text';
}

function _verdictBadge(avgTask, avgRetention, is2Obj) {
  const badge = document.createElement('span');
  if (is2Obj) {
    if (avgTask > 0) {
      badge.className = 'verdict-helps';
      badge.textContent = 'Helps';
    } else {
      badge.className = 'verdict-hurts';
      badge.textContent = 'Hurts';
    }
  } else {
    if (avgTask > 0 && (avgRetention == null || avgRetention >= -0.05)) {
      badge.className = 'verdict-helps';
      badge.textContent = 'Helps';
    } else if (avgTask > 0 && avgRetention < -0.05) {
      badge.className = 'verdict-mixed';
      badge.textContent = 'Mixed';
    } else {
      badge.className = 'verdict-hurts';
      badge.textContent = 'Hurts';
    }
  }
  return badge;
}

export function init(container) {
  _container = container;
  _container.innerHTML = '';

  const scrollWrap = document.createElement('div');
  scrollWrap.style.cssText = 'height:100%;overflow-y:auto;padding:12px 16px;';

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
  _container.appendChild(scrollWrap);
}

export function render(state) {
  if (!_tbody) return;

  const dag = state?.dag;
  if (!dag?.technique_stats) {
    _tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-secondary);padding:40px">No technique data yet</td></tr>';
    return;
  }

  const is2Obj = dag.meta?.objective_count === 2;

  // Toggle retention columns
  _container.querySelectorAll('.retention-col').forEach(el => {
    el.style.display = is2Obj ? 'none' : '';
  });

  // Build rows from technique_stats
  const rows = Object.entries(dag.technique_stats).map(([name, stats]) => ({
    technique:           name,
    times_tried:         stats.times_tried       ?? 0,
    avg_task_delta:      stats.avg_task_delta     ?? null,
    avg_retention_delta: stats.avg_retention_delta ?? null,
    best_parent_delta:   stats.best_parent_delta  ?? null,
  }));

  // Sort
  rows.sort((a, b) => {
    const av = a[_sortKey] ?? -Infinity;
    const bv = b[_sortKey] ?? -Infinity;
    if (av < bv) return _sortDir === 'asc' ? -1 :  1;
    if (av > bv) return _sortDir === 'asc' ?  1 : -1;
    return 0;
  });

  // Update sort indicators
  _container.querySelectorAll('th[data-key]').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.key === _sortKey) {
      th.classList.add(_sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
    }
  });

  _tbody.innerHTML = '';
  rows.forEach(row => {
    const tr = document.createElement('tr');

    // Technique name
    const tdName = document.createElement('td');
    tdName.textContent = row.technique;
    tdName.style.fontFamily = 'monospace';
    tr.appendChild(tdName);

    // Tried
    const tdTried = document.createElement('td');
    tdTried.textContent = String(row.times_tried);
    tr.appendChild(tdTried);

    // Avg Task Delta
    const tdTask = document.createElement('td');
    tdTask.className = _deltaClass(row.avg_task_delta);
    const taskSpan = document.createElement('span');
    taskSpan.className = _deltaTextClass(row.avg_task_delta);
    taskSpan.textContent = _fmtDelta(row.avg_task_delta);
    tdTask.appendChild(taskSpan);
    tr.appendChild(tdTask);

    // Avg Retention Delta
    const tdRet = document.createElement('td');
    tdRet.className = 'retention-col ' + _deltaClass(row.avg_retention_delta);
    if (is2Obj) tdRet.style.display = 'none';
    const retSpan = document.createElement('span');
    retSpan.className = _deltaTextClass(row.avg_retention_delta);
    retSpan.textContent = _fmtDelta(row.avg_retention_delta);
    tdRet.appendChild(retSpan);
    tr.appendChild(tdRet);

    // Best Parent Delta
    const tdBest = document.createElement('td');
    tdBest.className = _deltaClass(row.best_parent_delta);
    const bestSpan = document.createElement('span');
    bestSpan.className = _deltaTextClass(row.best_parent_delta);
    bestSpan.textContent = _fmtDelta(row.best_parent_delta);
    tdBest.appendChild(bestSpan);
    tr.appendChild(tdBest);

    // Verdict badge
    const tdVerdict = document.createElement('td');
    tdVerdict.appendChild(
      _verdictBadge(row.avg_task_delta, row.avg_retention_delta, is2Obj)
    );
    tr.appendChild(tdVerdict);

    _tbody.appendChild(tr);
  });
}
