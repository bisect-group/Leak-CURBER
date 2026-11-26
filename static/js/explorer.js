// Results explorer: bars and a color-coded table for the ten Leak-CURBER subtasks.
(() => {
  const root = document.getElementById('results-explorer');
  if (!root) return;

  const TASKS = [
    ['Kinetic parameters', [['kcat', 'kcat'], ['km', 'KM'], ['ki', 'Ki']]],
    ['Binding affinity', [['kd', 'Kd'], ['ic50', 'IC50'], ['ec50', 'EC50']]],
    ['Enzyme function', [['ec_classification', 'Enzyme classification'], ['retrieval', 'Enzyme retrieval']]],
    ['Reaction outcome', [['forward_synthesis', 'Forward prediction'], ['retrosynthesis', 'Retrosynthesis']]],
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
  // Other CSV methods (DLKcat, UniKP, CatPred, CLAIRE, Theia, EnzyFormer, GLN, MPEK) stay hidden.
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
  // Model colors from EMULaToR/figures/make_figures.py.
  const MODEL_COLORS = {
    "KcatNet": "#e41a1c",
    "EITLEM": "#377eb8",
    "CataPro": "#4daf4a",
    "DEKP": "#ff7f00",
    "eMOSAIC-T": "#984ea3",
    "BIND": "#a65628",
    "BACPI": "#f781bf",
    "Structure-Free DTA": "#1b9e77",
    "CLEAN": "#e41a1c",
    "GraphEC": "#377eb8",
    "HitEC": "#4daf4a",
    "CREEP": "#ff7f00",
    "CLIPZyme": "#984ea3",
    "Horizyn": "#a65628",
    "Root-aligned SMILES": "#d95f02",
    "MEGAN": "#7570b3",
    "Chemformer": "#e7298a"
};
  const REF_COLORS = { XGBoost: '#999999', MLP: '#4d4d4d', BLASTP: '#a6761d', Foldseek: '#8c8c8c' };

  const state = { task: 'kcat', metric: 'R2', view: 'bars', paper: false, refs: true };
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
  const ORIGINAL = 'Original paper (other data)';
  const splitLabel = (s) => (s === 'Original' ? ORIGINAL : s);

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
    const models = names.map((model) => {
      const values = {};
      for (const row of data.rows) {
        if (row.Method.replace('StructureFree-DTA', 'Structure-Free DTA') !== model) continue;
        values[row.Split] = { mean: num(row[`${metric}_mean`]), std: num(row[`${metric}_std`]) };
      }
      const ref = REFERENCES.includes(model);
      return { model, ref, color: ref ? REF_COLORS[model] : MODEL_COLORS[model], values };
    });
    return { splits, models: models.filter((m) => Object.keys(m.values).length) };
  }
  const fmt = (v) => (v === null || v === undefined ? '—' : v.toFixed(PERCENT.test(state.metric) ? 1 : 2));

  // The hint shows only while the content is wider than the box.
  function scrollBox(child, name) {
    const box = el('div', { class: 'lc-ex-scroll', tabindex: '0', role: 'region', 'aria-label': name }, child);
    const hint = el('p', { class: 'lc-hint', 'aria-hidden': 'true' }, 'Scroll sideways to see all columns →');
    const update = () => hint.classList.toggle('is-on', box.scrollWidth > box.clientWidth + 1);
    const watch = new ResizeObserver(update);
    watch.observe(box);
    watch.observe(child);
    return el('div', {}, hint, box);
  }

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
    Chart.defaults.font.size = matchMedia('(max-width: 600px)').matches ? 14 : 12;
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
        plugins: {
          tooltip: { callbacks: { afterLabel: item => window.LC_MODEL_CITATIONS[item.dataset.label] || '' } },
          legend: {
            position: 'top', labels: { boxWidth: 12 },
            onHover: (_, item) => { canvas.title = window.LC_MODEL_CITATIONS[item.text] || ''; },
            onLeave: () => { canvas.title = ''; },
          },
        },
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

  function ranks(values) {
    const lower = LOWER_IS_BETTER.has(state.metric);
    const sorted = [...new Set(values.filter((v) => v !== null))].sort((a, b) => (lower ? a - b : b - a));
    return { best: sorted[0], second: sorted[1] };
  }

  function drawGrid(host, { splits, models }) {
    // Ranks use the published baselines only; reference rows and the Original column stay unmarked.
    const marks = Object.fromEntries(splits.map((s) => [s, ranks(models.filter((m) => !m.ref).map((m) => m.values[s]?.mean ?? null))]));
    const lower = LOWER_IS_BETTER.has(state.metric);
    const head = el('tr', {}, el('th', { scope: 'col' }, 'Model'), ...splits.map((s) => el('th', { scope: 'col', class: s === 'Original' ? 'lc-ex-orig' : '' }, splitLabel(s))));
    const body = [];
    for (const m of models) {
      const cells = splits.map((s) => {
        const v = m.values[s];
        const mean = el('span', {}, v ? fmt(v.mean) : '—');
        const td = el('td', { class: s === 'Original' ? 'lc-ex-paper' : '' }, mean);
        if (v && v.std !== null) td.append(el('small', {}, ` ±${v.std.toFixed(PERCENT.test(state.metric) ? 1 : 2)}`));
        if (v && !m.ref && s !== 'Original') {
          const kind = ['best', 'second'].find((k) => marks[s][k] === v.mean);
          if (kind) (kind === 'best' ? td : mean).classList.add(`lc-ex-${kind}`);
        }
        const best = marks[s].best;
        if (m.ref && s !== 'Original' && v?.mean != null && best != null && (lower ? v.mean < best : v.mean > best)) {
          mean.append(el('sup', { title: 'Beats the best published baseline' }, ' ↑'));
        }
        return td;
      });
      if (m.ref && m === models.find(model => model.ref)) {
        body.push(el('tr', { class: 'lc-reference-heading' }, el('th', { colspan: splits.length + 1 }, 'Simple references')));
      }
      body.push(el('tr', {}, el('th', { scope: 'row' }, label(m.model)), ...cells));
    }
    const table = el('table', { class: 'lc-ex-table lc-score-table' }, el('thead', {}, head), el('tbody', {}, ...body));
    host.append(legend(models.some(m => m.ref)));
    host.append(scrollBox(table, `Results table for ${METRIC_LABEL[state.metric]}, scrolls sideways`));
  }

  const legend = (refs) =>
    el('p', { class: 'lc-ex-key' },
      ...(refs ? [el('span', {}, '↑ Beats the best published baseline.')] : []),
      ...[['best', 'Bold: best baseline on this split.'], ['second', 'Underlined: second best.']].map(([k, t]) =>
        el('span', { class: `lc-ex-${k}` }, t)));

  // ---- controls ----
  const taskSelect = el('select', { id: 'lc-ex-task', onchange: (e) => { state.task = e.target.value; refresh(true); } },
    ...TASKS.map(([group, items]) => el('optgroup', { label: group }, ...items.map(([k, t]) => el('option', { value: k }, t)))));
  const metricSelect = el('select', { id: 'lc-ex-metric', onchange: (e) => { state.metric = e.target.value; refresh(); } });
  const views = [['bars', 'Bars'], ['table', 'Table']];
  const viewButtons = views.map(([k, t]) => el('button', { type: 'button', 'data-view': k, onclick: () => { state.view = k; refresh(); } }, t));
  const check = (id, text, key) =>
    el('label', { class: 'lc-ex-check' }, el('input', { type: 'checkbox', id, ...(state[key] ? { checked: '' } : {}), onchange: (e) => { state[key] = e.target.checked; refresh(); } }), ` ${text}`);
  const note = el('p', { class: 'lc-ex-note', 'aria-live': 'polite' });
  const view = el('div', { class: 'lc-ex-view' });

  const caption = el('p', { class: 'lc-caption', 'aria-live': 'polite' });
  const HIGHLIGHT_SPLITS = {
    kcat: 'P-seq', km: 'P-seq', ki: 'P-seq', kd: 'P-seq', ic50: 'P-seq', ec50: 'P-seq',
    ec_classification: 'P-seq', retrieval: 'Time', forward_synthesis: 'DRFP', retrosynthesis: 'DRFP',
  };
  const metricHelp = {
    RMSE: 'RMSE gives larger prediction errors more weight.',
    MAE: 'MAE measures the average size of a prediction error on the log scale.',
    PCC: 'Pearson correlation measures how closely predictions follow a linear relationship with the measurements.',
    SCC: 'Spearman correlation compares the ranking of predicted and measured values.',
    L1: 'Level 1 requires a match on the first part of the EC label.',
    L2: 'Level 2 requires a match on the first two parts of the EC label.',
    L3: 'Level 3 requires a match on the first three parts of the EC label.',
    Top10: 'A prediction counts as correct if any of the first ten candidates matches the target.',
  };
  function resultHighlight({ models }) {
    const primary = {
      kcat: state.metric === 'R2' && 'KcatNet R² falls from 0.49 on GR-seq to 0.13 on P-seq.',
      km: state.metric === 'R2' && (state.refs
        ? 'On P-seq, XGBoost reaches R² 0.32. CataPro is the strongest published baseline at 0.23.'
        : 'DEKP leads on GR-seq (0.47) and GR-SMILES (0.50), while CataPro leads on P-seq (0.23) and Time (0.31).'),
      ki: state.metric === 'R2' && 'eMOSAIC-T reaches R² 0.79 on GR-SMILES, then falls to −0.42 on GR-seq.',
      kd: state.metric === 'R2' && 'Structure-Free DTA reaches R² 0.66 on GR-SMILES and 0.03 on P-seq.',
      ic50: state.metric === 'R2' && 'On P-seq, BACPI reaches R² 0.22, while BIND reaches −0.10.',
      ec50: state.metric === 'R2' && 'BACPI reaches R² 0.70 on GR-SMILES and 0.02 on P-seq.',
      ec_classification: state.metric === 'L4' && 'GraphEC exact EC accuracy falls from 88.8% on GR-seq to 50.7% on P-seq.',
      retrieval: state.metric === 'L4' && 'Horizyn reaches 43.5% exact EC accuracy on GR-rxnsmi and 18.1% on Time.',
      forward_synthesis: state.metric === 'Top1' && 'Chemformer reaches 49.1% top-1 accuracy on GR-rxnsmi and 23.9% on DRFP.',
      retrosynthesis: state.metric === 'Top1' && 'Chemformer reaches 29.9% top-1 accuracy on GR-rxnsmi and 11.3% on DRFP.',
    }[state.task];
    if (primary) return primary;
    if (state.task === 'forward_synthesis' && state.metric === 'Top10') {
      const rootAligned = models.find((model) => model.model === 'Root-aligned SMILES')?.values.DRFP?.mean;
      const chemformer = models.find((model) => model.model === 'Chemformer')?.values.DRFP?.mean;
      if (rootAligned !== undefined && chemformer !== undefined) {
        return `On DRFP, Root-aligned SMILES scores ${fmt(rootAligned)}%, above Chemformer (${fmt(chemformer)}%) at top-10.`;
      }
    }
    const split = HIGHLIGHT_SPLITS[state.task];
    const candidates = models.filter((model) => !model.ref && model.values[split]?.mean !== null && model.values[split]?.mean !== undefined);
    if (!candidates.length) return metricHelp[state.metric] || '';
    const best = candidates.reduce((winner, model) => {
      const candidate = model.values[split].mean;
      const current = winner.values[split].mean;
      return (LOWER_IS_BETTER.has(state.metric) ? candidate < current : candidate > current) ? model : winner;
    });
    const value = best.values[split].mean;
    const direction = LOWER_IS_BETTER.has(state.metric) ? 'lowest' : 'highest';
    const unit = PERCENT.test(state.metric) ? '%' : '';
    const tied = candidates.filter((model) => model.values[split].mean === value).map((model) => label(model.model));
    const names = tied.length > 1 ? `${tied.slice(0, -1).join(', ')} and ${tied.at(-1)} tie` : tied[0];
    return `${names} ${tied.length > 1 ? 'have' : 'has'} the ${direction} ${METRIC_LABEL[state.metric] || state.metric} among the published models on ${split} (${fmt(value)}${unit}).`;
  }
  let revision = 0;
  root.append(
    el('div', { class: 'lc-ex-controls' },
      el('label', { for: 'lc-ex-task' }, 'Subtask ', taskSelect),
      el('label', { for: 'lc-ex-metric' }, 'Metric ', metricSelect),
      el('div', { class: 'lc-ex-seg', role: 'group', 'aria-label': 'View' }, ...viewButtons),
      el('div', { class: 'lc-ex-options' },
        check('lc-ex-paper', `Show ${ORIGINAL}`, 'paper'),
        check('lc-ex-refs', 'Show simple references', 'refs'))),
    note, view, caption);

  async function refresh(taskChanged = false) {
    const request = ++revision;
    let data;
    try {
      data = await load(state.task);
    } catch (err) {
      if (request !== revision) return;
      chart?.destroy();
      chart = null;
      view.replaceChildren(el('p', { class: 'lc-ex-error', role: 'alert' }, `Could not load the results for this subtask (${err.message}). Please reload the page.`));
      return;
    }
    if (request !== revision) return;
    const metrics = metricsOf(data);
    if (taskChanged || !metrics.includes(state.metric)) {
      state.metric = ['R2', 'L4', 'Top1'].find((m) => metrics.includes(m));
    }
    metricSelect.replaceChildren(...metrics.map((m) => el('option', { value: m }, METRIC_LABEL[m] || m)));
    metricSelect.value = state.metric;
    
    viewButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === state.view)));
    const bars = state.view === 'bars';
    note.textContent = `${LOWER_IS_BETTER.has(state.metric) ? 'Lower' : 'Higher'} is better. Values are means over three seeds; `
      + (bars ? 'error bars show the standard deviation. ' : '± gives the standard deviation. ')
      + (state.paper ? `${bars ? 'Hatched bars' : 'Italic values'} show ${ORIGINAL} numbers. ` : '');
    chart?.destroy();
    chart = null;
    view.replaceChildren();
    const picked = selection(data);
    caption.textContent = resultHighlight(picked);
    for (const control of document.querySelectorAll('[data-result-tasks]')) {
      control.hidden = !control.dataset.resultTasks.split(' ').includes(state.task);
      if (taskChanged || control.hidden) control.open = false;
    }
    if (state.view === 'bars') drawBars(view, picked);
    else drawGrid(view, picked);
  }

  for (const link of document.querySelectorAll('[data-result]')) {
    link.addEventListener('click', () => {
      state.task = taskSelect.value = link.dataset.result;
      state.refs = document.getElementById('lc-ex-refs').checked = state.task === 'km';
      state.paper = document.getElementById('lc-ex-paper').checked = false;
      state.view = 'bars';
      refresh(true);
    });
  }
  taskSelect.value = state.task;
  refresh();
  new MutationObserver(() => refresh()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => refresh());
})();
