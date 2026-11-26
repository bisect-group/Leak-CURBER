// Live HTML/SVG versions of the five poster figures. Icons come from static/img/art/.
// Numbers use data-fact attributes, which facts.js fills.
(() => {
  window.LC = { YEARS: [2018, 2019, 2020, 2021, 2022, 2023, 2024] };
  const slot = (name) => document.querySelector(`.lc-figure[data-figure="${name}"]`);

  const h = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    node.append(...kids.flat().filter((k) => k !== null && k !== undefined));
    return node;
  };
  const html = (tag, cls, markup) => {
    const node = h(tag, { class: cls });
    node.innerHTML = markup;
    return node;
  };
  const pill = (...kids) => h('span', { class: 'lc-pill' }, ...kids);
  const art = (name, alt, w, hgt, cls = '') =>
    h('img', { class: `lc-art ${cls}`.trim(), src: `static/img/art/${name}.webp`, alt, width: w, height: hgt, loading: 'lazy' });
  const list = (items, cls = 'lc-list') => h('ul', { class: cls }, items.map((t) => html('li', '', t)));

  // ---- 1. data and statistics ----
  const STATS = [
    ['reactions', 'Reactions', 'db-reactions', 'A database cylinder with an enzyme binding a substrate.', 420, 378],
    ['pairs', 'Enzyme–reaction pairs', 'db-center', 'A database cylinder with a check mark.', 200, 302],
    ['enzymes', 'Enzymes', 'db-enzymes', 'A database cylinder with four protein structures.', 360, 376],
    ['metabolites', 'Metabolites', 'db-metabolites', 'A database cylinder with small-molecule structures.', 260, 391],
    ['properties', 'Measured values', 'db-properties', 'A database cylinder with a protein holding a ligand.', 260, 386],
  ];

  function buildDataStats(root) {
    const tiles = STATS.map(([key, name, img, alt, w, hgt]) =>
      h('figure', { class: `lc-tile lc-tile-${key}` },
        h('figcaption', {}, pill(html('strong', 'lc-num', `<span data-fact="stats.scale.${key}">—</span>`), h('span', {}, name))),
        art(img, alt, w, hgt)));
    root.append(h('div', { class: 'lc-stats-grid' },
      h('div', { class: 'lc-loop', 'aria-hidden': 'true' }), ...tiles, h('p', { class: 'lc-stats-name' }, pill('Leak-CURBER'))));
  }

  // ---- 2. pipeline ----
  const PIPELINE = [
    { title: 'Reactions', img: 'stack-reactions', processed: ['Definition', 'Balanced equation', 'EC number and protein ID', 'Reaction participant IDs'], sources: ['Rhea', 'KEGG', 'SABIO-RK', 'BRENDA'] },
    { title: 'Metabolites', img: 'stack-metabolites', processed: ['Name', 'Canonical SMILES', '2D and 3D structure'], sources: ['ChEBI', 'KEGG', 'PubChem'] },
    { title: 'Proteins', img: 'stack-proteins', processed: ['Protein sequence', 'Structure, first available of: experimental PDB, AlphaFold, ESMFold2, ESM3'], sources: ['UniProt', 'PDBe', 'AlphaFoldDB', 'ESMFold2', 'ESM3'] },
    { title: 'Measured values', img: 'stack-properties', processed: ['Temperature and pH', 'k<sub>cat</sub>, K<sub>M</sub>, K<sub>i</sub>', 'K<sub>d</sub>, EC<sub>50</sub>, IC<sub>50</sub>'], sources: ['BRENDA', 'SABIO-RK', 'BindingDB'] },
    { title: 'Literature', img: 'stack-literature', processed: ['PubMed ID', 'Title', 'Abstract', 'DOI'], sources: ['PubMed'] },
  ];
  const arrow = () => html('li', 'lc-arrow', '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l14 9-14 9z"/></svg>');

  function buildPipeline(root) {
    const items = PIPELINE.flatMap((s, i) => [
      h('li', { class: 'lc-step' },
        h('h3', {}, s.title),
        h('h4', {}, 'Processed data'),
        list(s.processed),
        art(s.img, `Stacked database cylinders for ${s.title.toLowerCase()}.`, 260, 224),
        h('h4', {}, 'Sources'),
        list(s.sources)),
      i < PIPELINE.length - 1 ? arrow() : null,
    ]);
    root.append(h('ol', { class: 'lc-pipeline' }, ...items));
  }

  // ---- 3. modalities ----
  const MODALITIES = [
    { name: 'Molecule sequence', model: 'SMI-TED', input: html('code', 'lc-input-text', 'CC(=O)Nc1ccc(O)cc1') },
    { name: 'Molecule graph', model: 'MolR', input: art('mol-graph', 'Skeletal drawing of paracetamol.', 340, 180) },
    { name: 'Molecule conformer', model: 'Uni-Mol2', input: art('mol-conformer', 'Ball-and-stick 3D model of paracetamol.', 340, 176) },
    { name: 'Reaction', model: 'DRFP', input: art('split-rxn-atoms', 'Four reactant atoms joining into one product.', 700, 60) },
    { name: 'Protein sequence', model: 'ESM-C', input: h('span', { class: 'lc-chips' }, ...['MET', 'LYS', 'PHE', 'GLY'].map((a) => h('span', {}, a))) },
    { name: 'Protein structure', model: 'ESM3', input: art('prot-structure', 'Ribbon drawing of a protein structure.', 220, 186) },
  ];

  function buildModalities(root) {
    const cols = MODALITIES.map((m) =>
      h('li', { class: 'lc-mod' },
        pill(m.name),
        h('div', { class: 'lc-mod-input' }, m.input),
        h('div', { class: 'lc-mod-model' }, m.model),
        h('div', { class: 'lc-vector', 'aria-hidden': 'true' }, ...Array.from({ length: 5 }, () => h('i'))),
        h('span', { class: 'lc-wire', 'aria-hidden': 'true' })));
    root.append(
      h('ol', { class: 'lc-modalities' }, ...cols),
      h('div', { class: 'lc-store' },
        h('span', { class: 'lc-shards' },
          ...Array.from({ length: 3 }, () => h('i', { 'aria-hidden': 'true' })),
          h('strong', {}, 'Embeddings'))));
  }

  // ---- 4. tasks ----
  // ---- 5. splits ----
  const SPLIT_KEY = [['train', 'Train'], ['val', 'Validation'], ['test', 'Test']];

  function buildSplits(root) {
    const gr = h('section', { class: 'lc-split lc-split-gr', 'aria-label': 'GR split' },
      h('h3', {}, pill('GR')),
      art('split-gr-chains', 'Rows of small molecules; most are dark blue, the last two are lighter.', 420, 95),
      art('split-gr-blobs', 'A row of proteins; the last two are lighter and form the held-out groups.', 600, 94),
      h('p', {}, 'Rows that share the same protein, molecule or reaction stay in one set.'),
      html('ul', 'lc-tags', '<li>GR-seq</li><li>GR-SMILES</li><li>GR-rxnsmi</li>'),
      h('ul', { class: 'lc-split-key' }, SPLIT_KEY.map(([k, t]) => html('li', '', `<i class="lc-sq lc-sq-${k}"></i> ${t}`))));
    const sim = h('section', { class: 'lc-split lc-split-sim', 'aria-label': 'Similarity splits' },
      h('h3', {}, pill('Similarity')),
      h('div', { class: 'lc-sim-art' },
        art('split-similarity', 'Heat map of pairwise similarity with a bright diagonal.', 300, 298),
        art('split-protein', 'Ribbon drawing of a protein.', 260, 305),
        art('split-mol-2d', 'Flat drawing of a molecule.', 260, 198),
        art('split-mol-3d', 'Ball-and-stick 3D model of a molecule.', 260, 183)),
      art('split-seq-chain', 'A chain of amino acids.', 640, 73, 'lc-wide'),
      art('split-rxn-atoms', 'Four reactant atoms joining into one product.', 700, 60, 'lc-wide'),
      h('p', {}, 'A test item must stay below a similarity threshold to every training item.'),
      html('ul', 'lc-tags', '<li>P-seq</li><li>P-struct</li><li>Mol-2D</li><li>Mol-3D</li><li>DRFP</li>'));
    const time = h('section', { class: 'lc-split lc-split-time', 'aria-label': 'Time split' },
      h('h3', {}, pill('Time')),
      h('ol', { class: 'lc-timeline' }, window.LC.YEARS.map((y) => html('li', y === 2024 ? 'is-test' : '', `${y}${y === 2024 ? ' <small>test</small>' : ''}`))),
      h('p', {}, 'Training uses older annotations. Test uses the newest.'));
    root.append(h('div', { class: 'lc-splits' }, gr, sim, time));
  }

  const dialog = h('dialog', { class: 'lc-task-dialog', 'aria-labelledby': 'lc-dialog-title' });
  const title = h('h2', { id: 'lc-dialog-title' });
  const close = h('button', { type: 'button', autofocus: '', 'aria-label': 'Close dialog', title: 'Close dialog' });
  close.innerHTML = '<svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="m6 6 12 12M18 6 6 18"/></svg>';
  const content = h('div', {});
  dialog.append(h('header', {}, title, close), content);
  document.body.append(dialog);
  close.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', e => {
    if (e.target === dialog) {
      const r = dialog.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close();
    }
  });
  for (const event of ['wheel', 'touchstart', 'touchend', 'keydown']) {
    dialog.addEventListener(event, e => { if (e.key !== 'Escape') e.stopPropagation(); });
  }
  for (const task of document.querySelectorAll('.lc-subtask')) {
    const trigger = task.querySelector('summary');
    trigger.setAttribute('aria-haspopup', 'dialog');
    trigger.insertAdjacentHTML('beforeend', '<svg class="lc-open-dialog-icon" aria-hidden="true" width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path stroke-linecap="round" d="M14 4h6v6m0-6-7 7M10 20H4v-6m0 6 7-7"/></svg>');
    trigger.addEventListener('click', e => {
      e.preventDefault();
      title.innerHTML = trigger.querySelector('.lc-fold-title').innerHTML;
      content.replaceChildren(task.querySelector('.lc-fold-body').cloneNode(true));
      dialog.showModal();
    });
  }
  dialog.addEventListener('close', () => {
    const tip = dialog.querySelector('#lc-tip');
    if (tip) { tip.hidden = true; document.body.append(tip); }
    content.replaceChildren();
  });

  const builders = { 'data-stats': buildDataStats, pipeline: buildPipeline, modalities: buildModalities, splits: buildSplits };
  for (const [name, build] of Object.entries(builders)) {
    const root = slot(name);
    if (root) build(root);
  }
})();
