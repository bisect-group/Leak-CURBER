// Results explorer: bars, heatmap and table for the ten Leak-CURBER subtasks.
(() => {
  const root = document.getElementById('results-explorer');
  if (!root) return;

  const TASKS = [
    ['Kinetic parameters', [['kcat', 'Turnover number (kcat)'], ['km', 'Michaelis constant (KM)'], ['ki', 'Inhibition constant (Ki)']]],
    ['Binding affinity', [['kd', 'Dissociation constant (Kd)'], ['ic50', 'IC50'], ['ec50', 'EC50']]],
    ['Enzyme function', [['ec_classification', 'EC classification'], ['retrieval', 'Enzyme retrieval']]],
    ['Reaction outcome', [['forward_synthesis', 'Forward synthesis'], ['retrosynthesis', 'Retrosynthesis']]],
  ];

  // Split order follows figures.ipynb; names follow the paper.
  const KINETIC_SPLITS = ['Original', 'GR-SMILES', 'GR-seq', 'Mol-2D', 'Mol-3D', 'Time', 'P-struct', 'P-seq'];
  const SPLITS = {
    ec_classification: ['Original', 'GR-seq', 'Time', 'P-seq', 'P-struct'],
    retrieval: ['GR-rxnsmi', 'Time', 'DRFP', 'P-seq', 'P-struct'],
    forward_synthesis: ['Original', 'GR-rxnsmi', 'Time', 'DRFP'],
    retrosynthesis: ['Original', 'GR-rxnsmi', 'Time', 'DRFP'],
  };

  // Allowlist: the 17 evaluated baselines of the paper (MPEK excluded), keyed by task.
  // Other CSV methods (DLKcat, UniKP, DeepEnzyme, CatPred, CLAIRE, Theia, EnzyFormer, GLN, MPEK) stay hidden.
  const MODELS = {
    kcat: ['KcatNet', 'EITLEM', 'CataPro'],
    km: ['EITLEM', 'CataPro', 'DEKP'],
    ki: ['eMOSAIC-T', 'BIND', 'BACPI'],
    kd: ['BIND', 'BACPI', 'Structure-Free DTA'],
    ic50: ['BIND', 'BACPI'],
    ec50: ['BIND', 'BACPI'],
    ec_classification: ['CLEAN', 'GraphEC', 'HitEC'],
    retrieval: ['CREEP', 'CLIPZyme', 'Horizyn'],
    forward_synthesis: ['Root-aligned SMILES', 'MEGAN', 'Chemformer'],
    retrosynthesis: ['Root-aligned SMILES', 'MEGAN', 'Chemformer'],
  };
  const OUTCOME_TASKS = new Set(['forward_synthesis', 'retrosynthesis']);
  const REFERENCES = ['XGBoost', 'MLP', 'BLASTP', 'Foldseek'];
  const MODEL_LABEL = { EITLEM: 'EITLEM-Kinetics', HitEC: 'HIT-EC' };

  const METRIC_LABEL = {
    RMSE: 'RMSE', MAE: 'MAE', R2: 'R²', PCC: 'PCC', SCC: 'SCC',
    L1: 'EC level accuracy, L1 (%)', L2: 'EC level accuracy, L2 (%)',
    L3: 'EC level accuracy, L3 (%)', L4: 'EC level accuracy, L4 (%)',
    Top1: 'Top-1 accuracy (%)', Top10: 'Top-10 accuracy (%)',
  };
  const LOWER_IS_BETTER = new Set(['RMSE', 'MAE']);
  const PERCENT = /^(L\d|Top\d+)$/;
  const PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442'];
  const REF_COLORS = { XGBoost: '#999999', MLP: '#4d4d4d', BLASTP: '#a6761d', Foldseek: '#8c8c8c' };

  const state = { task: 'kcat', metric: 'R2', view: 'bars', paper: true, refs: false };
  const cache = {};
  let chart = null;

  const el = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    node.append(...kids);
    return node;
  };
  const token = (name, fallback) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const label = (m) => MODEL_LABEL[m] || m;
  const splitLabel = (s) => (s === 'Original' ? 'Original (paper)' : s);

  // ---- data ----
  // Small CSV parser: quoted fields, doubled quotes and commas inside quotes.
  function parseCsv(text) {
    const rows = [];
    let row = [], field = '', quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (c === '"') quoted = false;
        else field += c;
      } else if (c === '"') quoted = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(field); rows.push(row); row = []; field = '';
      } else field += c;
    }
    if (field || row.length) { row.push(field); rows.push(row); }
    return rows.filter((r) => r.length > 1 || r[0] !== '');
  }
  async function load(task) {
    if (!cache[task]) {
      const response = await fetch(`data/results/${task}.csv`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const [head, ...lines] = parseCsv(await response.text());
      cache[task] = {
        columns: head,
        rows: lines.map((cells) => Object.fromEntries(head.map((h, i) => [h, cells[i]]))),
      };
    }
    return cache[task];
  }
  const metricsOf = (data) => data.columns.filter((c) => c.endsWith('_mean')).map((c) => c.slice(0, -5));
  const num = (v) => (v === '' || v === undefined ? null : Number(v));

  // Rows shown for the current toggles: [{model, ref, values: {split: {mean, std}}}]
  function selection(data) {
    const { task, metric, paper, refs } = state;
    const splits = (SPLITS[task] || KINETIC_SPLITS).filter((s) => paper || s !== 'Original');
    const names = [...MODELS[task], ...(refs ? REFERENCES : [])];
    const models = names.map((model, i) => {
      const values = {};
      for (const row of data.rows) {
        if (row.Method.replace('StructureFree-DTA', 'Structure-Free DTA') !== model) continue;
        values[row.Split] = { mean: num(row[`${metric}_mean`]), std: num(row[`${metric}_std`]) };
      }
      const ref = REFERENCES.includes(model);
      return { model, ref, color: ref ? REF_COLORS[model] : PALETTE[i % PALETTE.length], values };
    });
    return { splits, models: models.filter((m) => Object.keys(m.values).length) };
  }
  const fmt = (v) => (v === null || v === undefined ? '—' : v.toFixed(PERCENT.test(state.metric) ? 1 : 2));

  const scrollBox = (child, name) =>
    el('div', { class: 'lc-ex-scroll', tabindex: '0', role: 'region', 'aria-label': name }, child);

  // ---- bars ----
  const errorBars = {
    id: 'lcErrorBars',
    afterDatasetsDraw(c) {
      const { ctx, scales: { y } } = c;
      ctx.save();
      ctx.strokeStyle = token('--lc-text', '#10233f');
      ctx.lineWidth = 1;
      c.data.datasets.forEach((ds, i) => {
        c.getDatasetMeta(i).data.forEach((bar, j) => {
          const mean = ds.data[j];
          const std = ds.std[j];
          if (mean === null || std === null) return;
          const [top, bottom] = [y.getPixelForValue(mean + std), y.getPixelForValue(mean - std)];
          const w = Math.min(bar.width / 3, 4);
          ctx.beginPath();
          ctx.moveTo(bar.x, top); ctx.lineTo(bar.x, bottom);
          ctx.moveTo(bar.x - w, top); ctx.lineTo(bar.x + w, top);
          ctx.moveTo(bar.x - w, bottom); ctx.lineTo(bar.x + w, bottom);
          ctx.stroke();
        });
      });
      ctx.restore();
    },
  };

  function hatch(color) {
    const tile = document.createElement('canvas');
    tile.width = tile.height = 8;
    const g = tile.getContext('2d');
    g.fillStyle = token('--lc-card', '#ffffff');
    g.fillRect(0, 0, 8, 8);
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(-2, 10); g.lineTo(10, -2);
    g.moveTo(-2, 2); g.lineTo(2, -2);
    g.moveTo(6, 10); g.lineTo(10, 6);
    g.stroke();
    return g.createPattern(tile, 'repeat');
  }

  function drawBars(host, { splits, models }) {
    const text = token('--lc-text', '#10233f');
    const grid = token('--lc-border', '#d0d7e2');
    Chart.defaults.color = text;
    Chart.defaults.font.family = 'inherit';
    const width = Math.max(splits.length * models.length * 15 + 120, 0);
    const canvas = el('canvas', {
      role: 'img',
      'aria-label': `Grouped bar chart of ${METRIC_LABEL[state.metric]} by split. The table view lists the same values.`,
    });
    const inner = el('div', { class: 'lc-ex-chart', style: `min-width:${width}px` }, canvas);
    host.append(scrollBox(inner, 'Bar chart, scrolls sideways'));
    const percent = PERCENT.test(state.metric);
    chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: splits.map(splitLabel),
        datasets: models.map((m) => ({
          label: label(m.model),
          data: splits.map((s) => m.values[s]?.mean ?? null),
          std: splits.map((s) => m.values[s]?.std ?? null),
          backgroundColor: (c) => (splits[c.dataIndex] === 'Original' ? hatch(m.color) : m.color),
          borderColor: m.color,
          borderWidth: 1,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: text } },
          y: {
            beginAtZero: true,
            max: percent ? 100 : undefined,
            grid: { color: grid },
            ticks: { color: text },
            title: { display: true, text: `${METRIC_LABEL[state.metric]} ${LOWER_IS_BETTER.has(state.metric) ? '(lower is better)' : '(higher is better)'}`, color: text },
          },
        },
      },
      plugins: [errorBars],
    });
  }

  // ---- heatmap and table ----
  const luminance = (hex) => {
    const n = parseInt(hex.replace('#', '').slice(0, 6), 16);
    const [r, g, b] = [n >> 16, (n >> 8) & 255, n & 255].map((v) => v / 255);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  // Marks the best, second best and worst value in one split column.
  function ranks(values) {
    const lower = LOWER_IS_BETTER.has(state.metric);
    const sorted = [...new Set(values.filter((v) => v !== null))].sort((a, b) => (lower ? a - b : b - a));
    return { best: sorted[0], second: sorted[1], worst: sorted.length > 2 ? sorted.at(-1) : undefined };
  }

  function drawGrid(host, { splits, models }, colored) {
    const tone = { best: token('--lc-best', '#99D8C9'), second: token('--lc-second', '#9EBCDA'), worst: token('--lc-worst', '#FDBB84') };
    // Ranks use the published baselines only; reference rows and the Original column stay uncoloured.
    const marks = Object.fromEntries(splits.map((s) => [s, ranks(models.filter((m) => !m.ref).map((m) => m.values[s]?.mean ?? null))]));
    const head = el('tr', {}, el('th', { scope: 'col' }, 'Model'), ...splits.map((s) => el('th', { scope: 'col' }, splitLabel(s))));
    const body = models.map((m) => {
      const cells = splits.map((s) => {
        const v = m.values[s];
        const td = el('td', { class: s === 'Original' ? 'lc-ex-paper' : '' }, v ? fmt(v.mean) : '—');
        if (v && v.std !== null) td.append(el('small', {}, ` ±${v.std.toFixed(PERCENT.test(state.metric) ? 1 : 2)}`));
        if (colored && v && !m.ref && s !== 'Original') {
          const kind = ['best', 'second', 'worst'].find((k) => marks[s][k] === v.mean);
          if (kind) td.style.cssText = `background:${tone[kind]};color:${luminance(tone[kind]) > 0.45 ? '#10233f' : '#ffffff'}`;
        }
        return td;
      });
      return el('tr', { class: m.ref ? 'lc-ex-ref' : '' }, el('th', { scope: 'row' }, label(m.model), m.ref ? el('span', { class: 'lc-ex-tag' }, 'reference') : ''), ...cells);
    });
    const table = el('table', { class: 'lc-ex-table' }, el('thead', {}, head), el('tbody', {}, ...body));
    host.append(scrollBox(table, `Results table for ${METRIC_LABEL[state.metric]}, scrolls sideways`));
    if (colored) host.append(legend());
  }

  const legend = () =>
    el('p', { class: 'lc-ex-key' },
      ...[['best', 'Best'], ['second', 'Second best'], ['worst', 'Worst']].map(([k, t]) =>
        el('span', { class: `lc-ex-swatch lc-ex-${k}` }, t)));

  function download({ splits, models }, data) {
    const metrics = metricsOf(data);
    const header = ['Split', 'Method', ...metrics.flatMap((m) => [`${m}_mean`, `${m}_std`])];
    const lines = [header.join(',')];
    for (const s of splits) {
      for (const m of models) {
        const row = data.rows.find((r) => r.Split === s && r.Method.replace('StructureFree-DTA', 'Structure-Free DTA') === m.model);
        if (row) lines.push([s, label(m.model), ...header.slice(2).map((h) => row[h])].join(','));
      }
    }
    const url = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/csv' }));
    el('a', { href: url, download: `leak-curber-${state.task}.csv` }).click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ---- controls ----
  const taskSelect = el('select', { id: 'lc-ex-task', onchange: (e) => { state.task = e.target.value; refresh(true); } },
    ...TASKS.map(([group, items]) => el('optgroup', { label: group }, ...items.map(([k, t]) => el('option', { value: k }, t)))));
  const metricSelect = el('select', { id: 'lc-ex-metric', onchange: (e) => { state.metric = e.target.value; refresh(); } });
  const views = [['bars', 'Bars'], ['heatmap', 'Heatmap'], ['table', 'Table']];
  const viewButtons = views.map(([k, t]) => el('button', { type: 'button', 'data-view': k, onclick: () => { state.view = k; refresh(); } }, t));
  const check = (id, text, key) =>
    el('label', { class: 'lc-ex-check' }, el('input', { type: 'checkbox', id, ...(state[key] ? { checked: '' } : {}), onchange: (e) => { state[key] = e.target.checked; refresh(); } }), ` ${text}`);
  const banner = el('p', { class: 'lc-ex-banner', role: 'status' }, 'These results will be updated. The outcome baselines are being fine-tuned.');
  const note = el('p', { class: 'lc-ex-note', 'aria-live': 'polite' });
  const view = el('div', { class: 'lc-ex-view' });
  const footnote = el('p', { class: 'lc-ex-note lc-ex-foot' });
  const csvButton = el('button', { type: 'button', class: 'lc-ex-csv' }, 'Download CSV');

  root.append(
    el('div', { class: 'lc-ex-controls' },
      el('label', { for: 'lc-ex-task' }, 'Task ', taskSelect),
      el('label', { for: 'lc-ex-metric' }, 'Metric ', metricSelect),
      el('div', { class: 'lc-ex-seg', role: 'group', 'aria-label': 'View' }, ...viewButtons),
      check('lc-ex-paper', 'Show paper-reported', 'paper'),
      check('lc-ex-refs', 'Show reference baselines', 'refs')),
    banner, note, view, footnote, csvButton);

  async function refresh(taskChanged = false) {
    let data;
    try {
      data = await load(state.task);
    } catch (err) {
      chart?.destroy();
      chart = null;
      view.replaceChildren(el('p', { class: 'lc-ex-error', role: 'alert' }, `Could not load the results for this task (${err.message}). Please reload the page.`));
      return;
    }
    const metrics = metricsOf(data);
    if (taskChanged || !metrics.includes(state.metric)) {
      state.metric = ['R2', 'L4', 'Top1'].find((m) => metrics.includes(m));
    }
    metricSelect.replaceChildren(...metrics.map((m) => el('option', { value: m }, METRIC_LABEL[m] || m)));
    metricSelect.value = state.metric;
    banner.hidden = !OUTCOME_TASKS.has(state.task);
    viewButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === state.view)));
    note.textContent = `${LOWER_IS_BETTER.has(state.metric) ? 'Lower' : 'Higher'} is better. Values are means over three seeds; error bars show the standard deviation. `
      + (state.paper ? 'Hatched bars and italic values are numbers reported in the original papers.' : '');
    chart?.destroy();
    chart = null;
    view.replaceChildren();
    const picked = selection(data);
    const spread = state.task === 'retrieval' && data.rows.find((r) => r.Method === 'Horizyn' && r.Split === 'GR-rxnsmi')?.L4_std;
    footnote.textContent = spread ? `Horizyn on GR-rxnsmi has a large seed spread (SD ${spread} at L4), as reported in the paper.` : '';
    footnote.hidden = !spread;
    if (state.view === 'bars') drawBars(view, picked);
    else drawGrid(view, picked, state.view === 'heatmap');
    csvButton.hidden = state.view !== 'table';
    csvButton.onclick = () => download(picked, data);
  }

  taskSelect.value = state.task;
  refresh();
  new MutationObserver(() => refresh()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => refresh());
})();
