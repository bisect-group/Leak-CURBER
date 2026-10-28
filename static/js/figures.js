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
    ['pairs', 'Enzyme–reaction pairs', 'db-center', 'A database cylinder with a check mark.', 200, 294],
    ['enzymes', 'Enzymes', 'db-enzymes', 'A database cylinder with four protein structures.', 360, 379],
    ['metabolites', 'Metabolites', 'db-metabolites', 'A database cylinder with small-molecule structures.', 260, 402],
    ['properties', 'Properties', 'db-properties', 'A database cylinder with a protein holding a ligand.', 260, 398],
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
    { title: 'Properties', img: 'stack-properties', processed: ['Temperature and pH', 'k<sub>cat</sub>, K<sub>M</sub>, K<sub>i</sub>', 'K<sub>d</sub>, EC<sub>50</sub>, IC<sub>50</sub>'], sources: ['BRENDA', 'SABIO-RK', 'BindingDB'] },
    { title: 'Literature', img: 'stack-literature', processed: ['PubMed ID', 'Title', 'Abstract', 'DOI'], sources: ['PubMed'] },
  ];
  const arrow = () => html('li', 'lc-arrow', '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l14 9-14 9z"/></svg>');

  function buildPipeline(root) {
    const items = PIPELINE.flatMap((s, i) => [
      h('li', { class: 'lc-step' },
        h('h3', {}, s.title),
        h('h4', {}, 'Processed data'),
        list(s.processed),
        art(s.img, `Stacked database cylinders for ${s.title.toLowerCase()}.`, 260, 221),
        h('h4', {}, 'Sources'),
        list(s.sources)),
      i < PIPELINE.length - 1 ? arrow() : null,
    ]);
    root.append(h('ol', { class: 'lc-pipeline' }, ...items),
      h('p', { class: 'lc-fig-note' }, 'MetaCyc: private leaderboard test set. It is not a training source.'));
  }

  // ---- 3. modalities ----
  const MODALITIES = [
    { name: 'Molecule sequence', model: 'SMI-TED', input: html('code', 'lc-input-text', 'CC(=O)Nc1ccc(O)cc1') },
    { name: 'Molecule graph', model: 'MolR', input: art('mol-graph', 'Skeletal drawing of paracetamol.', 340, 177) },
    { name: 'Molecule conformer', model: 'Uni-Mol2', input: art('mol-conformer', 'Ball-and-stick 3D model of paracetamol.', 340, 163) },
    { name: 'Reaction', model: 'DRFP', input: art('split-rxn-atoms', 'Four reactant atoms joining into one product.', 700, 54) },
    { name: 'Protein sequence', model: 'ESM-C', input: h('span', { class: 'lc-chips' }, ...['MET', 'LYS', 'PHE', 'GLY'].map((a) => h('span', {}, a))) },
    { name: 'Protein structure', model: 'ESM3', input: art('prot-structure', 'Ribbon drawing of a protein structure.', 220, 193) },
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
        pill('Sharded embedding store'),
        h('span', { class: 'lc-shards', 'aria-hidden': 'true' }, ...Array.from({ length: 4 }, () => h('i'))),
        pill('Leak-CURBER')),
      h('p', { class: 'lc-fig-note' }, 'The reaction column uses DRFP fingerprints, not a learned encoder.'));
  }

  // ---- 4. tasks ----
  const TASKS = [
    { key: 'kinetics', name: 'Kinetic parameters', ask: 'How efficiently does the enzyme act?', img: ['task-kinetics', 'An enzyme binds a substrate and releases a product, with the labels K_m, k_cat and K_i.', 560, 239], subs: ['k<sub>cat</sub>', 'K<sub>M</sub>', 'K<sub>i</sub>'] },
    { key: 'binding', name: 'Binding affinity', ask: 'How strongly does a molecule interact with the protein?', img: ['task-binding', 'A protein ribbon holding a small molecule.', 420, 353], subs: ['K<sub>d</sub>', 'IC<sub>50</sub>', 'EC<sub>50</sub>'] },
    { key: 'function', name: 'Enzyme function', ask: 'Which enzymatic function is associated with the reaction?', img: ['task-function', 'An enzyme binds a substrate; question marks ask for the function.', 560, 235], subs: ['EC classification', 'Enzyme retrieval'], ec: true },
    { key: 'outcome', name: 'Reaction outcome', ask: 'What transformation occurs?', img: ['task-outcome', 'Four coloured building blocks join into one product.', 900, 66], subs: ['Forward synthesis', 'Retrosynthesis'] },
  ];

  function buildTasks(root) {
    const cards = TASKS.map((t) =>
      h('section', { class: `lc-task lc-task-${t.key}`, 'aria-label': t.name },
        h('h3', {}, pill(t.name)),
        h('p', { class: 'lc-task-ask' }, t.ask),
        art(...t.img),
        t.ec ? h('div', { class: 'lc-ec', 'aria-label': 'EC levels 1 to 4' }, ...[1, 2, 3, 4].map((n) => h('span', { class: `lc-ec-${n}` }, `EC${n}`))) : null,
        h('ul', { class: 'lc-chiplist' }, t.subs.map((s) => html('li', '', s)))));
    root.append(h('div', { class: 'lc-tasks' }, ...cards));
  }

  // ---- 5. splits ----
  const SPLIT_KEY = [['train', 'Train'], ['val', 'Validation'], ['test', 'Test']];

  function buildSplits(root) {
    const gr = h('section', { class: 'lc-split lc-split-gr', 'aria-label': 'Grouped random split' },
      h('h3', {}, pill('Grouped random')),
      art('split-gr-chains', 'Rows of small molecules; most are dark blue, the last two are lighter.', 420, 90),
      art('split-gr-blobs', 'A row of proteins; the last two are lighter and form the held-out groups.', 600, 87),
      h('p', {}, 'Rows that share the same protein, molecule or reaction stay in one partition.'),
      html('ul', 'lc-tags', '<li>GR-seq</li><li>GR-SMILES</li><li>GR-rxnsmi</li>'),
      h('ul', { class: 'lc-split-key' }, SPLIT_KEY.map(([k, t]) => html('li', '', `<i class="lc-sq lc-sq-${k}"></i> ${t}`))));
    const sim = h('section', { class: 'lc-split lc-split-sim', 'aria-label': 'Similarity splits' },
      h('h3', {}, pill('Similarity')),
      h('div', { class: 'lc-sim-art' },
        art('split-similarity', 'Heat map of pairwise similarity with a bright diagonal.', 300, 299),
        art('split-protein', 'Ribbon drawing of a protein.', 260, 306),
        art('split-mol-2d', 'Flat drawing of a molecule.', 260, 198),
        art('split-mol-3d', 'Ball-and-stick 3D model of a molecule.', 260, 181)),
      art('split-seq-chain', 'A chain of amino acids.', 640, 66, 'lc-wide'),
      art('split-rxn-atoms', 'Four reactant atoms joining into one product.', 700, 54, 'lc-wide'),
      h('p', {}, 'A test item must stay below a similarity threshold to every training item.'),
      html('ul', 'lc-tags', '<li>P-seq</li><li>P-struct</li><li>Mol-2D</li><li>Mol-3D</li><li>DRFP</li>'));
    const time = h('section', { class: 'lc-split lc-split-time', 'aria-label': 'Time split' },
      h('h3', {}, pill('Time')),
      h('p', { class: 'lc-eg' }, 'Example years'),
      h('ol', { class: 'lc-timeline' }, window.LC.YEARS.map((y) => html('li', y === 2024 ? 'is-test' : '', `${y}${y === 2024 ? ' <small>test</small>' : ''}`))),
      h('p', {}, 'Training uses older annotations. Test uses the newest.'),
      list(['Time'], 'lc-tags'));
    root.append(h('div', { class: 'lc-splits' }, gr, sim, time));
  }

  const builders = { 'data-stats': buildDataStats, pipeline: buildPipeline, modalities: buildModalities, tasks: buildTasks, splits: buildSplits };
  for (const [name, build] of Object.entries(builders)) {
    const root = slot(name);
    if (root) build(root);
  }
})();
