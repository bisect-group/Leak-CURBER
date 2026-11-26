// Plain-language hints for technical terms. Mark a term in HTML as
// <button type="button" class="lc-term" data-term="kcat">...</button>.
(() => {
  const GLOSSARY = {
    data_licence: 'Dataset licence: CC BY-NC-SA 4.0 (non-commercial use).',
    code_licence: 'Code licence: MIT.',
    amylase: 'An enzyme in saliva that breaks starch into smaller sugars.',
    enzyme: 'A protein that speeds up a chemical reaction in a living cell.',
    substrate: 'The molecule an enzyme acts on.',
    ec: 'An EC number describes the reaction an enzyme catalyzes. In 1.1.1.1, the first number identifies the broad class. A Level 4 prediction must match all four numbers.',
    kcat: 'How fast an enzyme works: how many reactions one enzyme does each second.',
    km: 'How much substrate an enzyme needs to work at half its top speed.',
    ki: 'How strongly an inhibitor blocks an enzyme. A smaller value means stronger blocking.',
    kd: 'How tightly two molecules bind. A smaller value means tighter binding.',
    ic50: 'The amount of a molecule that cuts an enzyme’s activity by half.',
    ec50: 'The amount of a molecule that gives half of its largest effect.',
    embedding: 'Turns a protein or molecule into numbers a model can use.',
    split: 'A way to divide data into training parts and test parts.',
    leakage: 'When test data looks like training data, so the score looks too high.',
    topk: 'The share of cases where the right answer is among the model’s top k guesses.',
    r2: 'A regression score. 1 is perfect, 0 equals always guessing the mean, and below 0 is worse.',
    retrieval: 'Finding the right enzyme for a given reaction from a large list.',
    baseline: 'A published model that we retrain and test on the benchmark.',
    validation: 'The rows used to tune the model. They are not used for the final score.',
    seed: 'One random start of training. We report the mean over three seeds.',
    top10: 'The right answer is in the model’s first 10 guesses.',
    drfp: 'A fingerprint of the change in a reaction.',
    smiles: 'A text string that writes a molecule.',
    analog: 'A molecule with a chemical structure similar to another molecule. For example, the two may differ by a single chemical group.',
    homologous: 'Two proteins are homologous if they share an evolutionary ancestor. Their sequences and functions can change over time.',
    homolog: 'A protein that shares an evolutionary ancestor with another protein.',
    ligand: 'A molecule that binds a protein.',
    tanimoto: 'A similarity score from 0 to 1 for molecule fingerprints.',
  };

  const citations = {
    "KcatNet": "Pan et al. (2026). A geometric deep learning framework for genome-wide prediction of enzyme turnover number.",
    "CataPro": "Wang et al. (2025). Robust enzyme discovery and engineering with deep learning using CataPro.",
    "EITLEM-Kinetics": "Shen et al. (2024). EITLEM-Kinetics: A deep-learning framework for kinetic parameter prediction of mutant enzymes.",
    "DEKP": "Wang et al. (2025). DEKP: a deep learning model for enzyme kinetic parameter prediction based on pretrained models and graph neural networks.",
    "DLTKcat": "Qiu et al. (2024). DLTKcat: deep learning-based prediction of temperature-dependent enzyme turnover rates.",
    "MPEK": "Wang et al. (2024). MPEK: a multitask deep learning framework based on pretrained language models for enzymatic reaction kinetic parameters prediction.",
    "ProSmith": "Kroll et al. (2024). A multimodal Transformer Network for protein-small molecule interactions enhances predictions of kinase inhibition and enzyme-substrate relationships.",
    "eMOSAIC-T": "Badkul et al. (2025). Multimodal out-of-distribution individual uncertainty quantification enhances binding affinity prediction for polypharmacology.",
    "BIND": "Lam et al. (2024). Protein language models are performant in structure-free virtual screening.",
    "BACPI": "Li et al. (2022). BACPI: a bi-directional attention neural network for compound–protein interaction and binding affinity prediction.",
    "MambaTransDTA": "Lou et al. (2026). MambaTransDTA: A Hybrid Mamba-Transformer Architecture for Accurate Drug-Target Binding Affinity Prediction.",
    "ColdDTA": "Fang et al. (2023). ColdDTA: Utilizing data augmentation and attention-based feature fusion for drug-target binding affinity prediction.",
    "Structure-Free DTA": "Bidgoli et al. (2026). Structure-free drug–target affinity prediction using protein and molecule language models.",
    "DeepDTAGen": "Shah et al. (2025). DeepDTAGen: a multitask deep learning framework for drug-target affinity prediction and target-aware drugs generation.",
    "CREEP": "Yang et al. (2025). CARE: a Benchmark Suite for the Classification and Retrieval of Enzymes.",
    "CLIPZyme": "Mikhael et al. (2024). CLIPZyme: reaction-conditioned virtual screening of enzymes.",
    "Horizyn": "Rocks et al. (2026). Dual-encoder contrastive learning accelerates enzyme discovery.",
    "CLEAN": "Yu et al. (2023). Enzyme function prediction using contrastive learning.",
    "GraphEC": "Song et al. (2024). Accurately predicting enzyme functions through geometric graph learning on ESMFold-predicted structures.",
    "HIT-EC": "Dumontet et al. (2026). Trustworthy prediction of enzyme commission numbers using a hierarchical interpretable transformer.",
    "Root-aligned SMILES": "Zhong et al. (2022). Root-aligned SMILES: a tight representation for chemical reaction prediction.",
    "MEGAN": "Sacha et al. (2021). Molecule Edit Graph Attention Network: Modeling Chemical Reactions as Sequences of Graph Edits.",
    "Chemformer": "Irwin et al. (2022). Chemformer: a pre-trained transformer for computational chemistry.",
    "TurNuP": "Kroll et al. (2023). Turnover number predictions for kinetically uncharacterized enzymes using machine and deep learning.",
    "UniKP": "Yu et al. (2023). UniKP: a unified framework for the prediction of enzyme kinetic parameters.",
    "CatPred": "Boorla and Maranas (2025). CatPred: a comprehensive framework for deep learning in vitro enzyme kinetic parameters.",
    "DLKcat": "Li et al. (2022). Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction.",
    "XGBoost": "Chen and Guestrin (2016). XGBoost: A Scalable Tree Boosting System.",
    "Foldseek": "van Kempen et al. (2024). Fast and accurate protein structure search with Foldseek.",
    "BLASTP": "Camacho et al. (2009). BLAST+: architecture and applications.",
    "EITLEM": "Shen et al. (2024). EITLEM-Kinetics: A deep-learning framework for kinetic parameter prediction of mutant enzymes.",
    "HitEC": "Dumontet et al. (2026). Trustworthy prediction of enzyme commission numbers using a hierarchical interpretable transformer.",
    "MLP": "Leak-CURBER camera-ready paper, experimental setup. The authors trained this multilayer perceptron on the benchmark embeddings."
};
  Object.assign(citations, {
    "ESP": "Kroll et al. (2023). A general model to predict small molecule substrates of enzymes based on machine and deep learning.",
    "EnzymeMap": "Heid et al. (2023). EnzymeMap: curation, validation and data-driven prediction of enzymatic reactions.",
    "ECREACT": "Probst et al. (2022). Biocatalysed synthesis planning using data-driven learning.",
    "CARE": "Yang et al. (2024). CARE: a Benchmark Suite for the Classification and Retrieval of Enzymes.",
    "ReactZyme": "Hua et al. (2024). ReactZyme: A Benchmark for Enzyme-Reaction Prediction.",
    "ProteinShake": "Kucera et al. (2023). ProteinShake: Building datasets and benchmarks for deep learning on protein structures.",
    "Rhea": "Bansal et al. (2022). Rhea, the reaction knowledgebase in 2022.",
    "KEGG": "Kanehisa et al. (2025). KEGG: biological systems database as a model of the real world.",
    "SABIO-RK": "Wittig et al. (2018). SABIO-RK: an updated resource for manually curated biochemical reaction kinetics.",
    "BRENDA": "Chang et al. (2021). BRENDA, the ELIXIR core data resource in 2021: new developments and updates.",
    "MetaCyc": "Caspi et al. (2020). The MetaCyc database of metabolic pathways and enzymes - a 2019 update.",
    "UniProt": "The UniProt Consortium (2025). UniProt: the Universal Protein Knowledgebase in 2025.",
    "PDBe": "Armstrong et al. (2020). PDBe: improved findability of macromolecular structure data in the PDB.",
    "PDB": "wwPDB consortium (2019). Protein Data Bank: the single global archive for 3D macromolecular structure data.",
    "BindingDB": "Liu et al. (2025). BindingDB in 2024: a FAIR knowledgebase of protein-small molecule binding data.",
    "ChEBI": "Hastings et al. (2016). ChEBI in 2016: Improved services and an expanding collection of metabolites.",
    "PubChem": "Kim et al. (2025). PubChem 2025 update.",
    "PubMed": "White (2020). PubMed 2.0.",
    "AlphaFoldDB": "Varadi et al. (2024). AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences.",
    "ESMFold2": "Candido et al. (2026). Language Modeling Materializes a World Model of Protein Biology.",
    "ESM3": "Hayes et al. (2025). Simulating 500 million years of evolution with a language model.",
    "ESMC": "EvolutionaryScale. Evolutionary Scale · ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning.",
    "SMI-TED": "Soares et al. (2025). An open-source family of large encoder-decoder foundation models for chemistry.",
    "MolR": "Wang et al. (2021). Chemical-Reaction-Aware Molecule Representation Learning.",
    "Uni-Mol2": "Ji et al. (2024). Exploring Molecular Pretraining Model at Scale.",
    "SMILES": "Weininger (1988). SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules.",
    "DRFP": "Probst et al. (2022). Reaction classification and yield prediction using the differential reaction fingerprint DRFP.",
    "MMseqs2": "Steinegger and Söding (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.",
    "BitBIRCH": "Pérez et al. (2025). BitBIRCH: efficient clustering of large molecular libraries.",
    "DataSAIL": "Joeres et al. (2025). Data splitting to avoid information leakage with DataSAIL.",
    "RDKit": "Landrum et al. (2026). rdkit/rdkit: 2026_09_2 (Q3 2026) Release.",
    "Swiss-Prot": "Bairoch and Boeckmann (1994). The SWISS-PROT protein sequence data bank: current status.",
    "CD-HIT": "Fu et al. (2012). CD-HIT: accelerated for clustering the next-generation sequencing data.",
    "ESM-2": "Lin et al. (2022). Language models of protein sequences at the scale of evolution enable accurate structure prediction.",
    "Davis": "Davis et al. (2011). Comprehensive analysis of kinase inhibitor selectivity.",
    "KIBA": "Tang et al. (2014). Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets: A Comparative and Integrative Analysis.",
    "AlphaFold": "Jumper et al. (2021). Highly accurate protein structure prediction with AlphaFold.",
    "ECFP": "Rogers and Hahn (2010). Extended-connectivity fingerprints.",
    "Morgan": "Morgan (1965). The generation of a unique machine description for chemical structures.",
    "Butina": "Butina (1999). Unsupervised data base clustering based on daylight’s fingerprint and Tanimoto similarity.",
    "UMAP": "McInnes, Healy and Melville (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.",
    "Adam": "Kingma and Ba (2015). Adam: A Method for Stochastic Optimization.",
    "AdamW": "Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization.",
    "USPTO": "Lowe (2017). Chemical reactions from US patents (1976–Sep2016). Parent patent reaction collection.",
    "ZINC-15": "Sterling and Irwin (2015). ZINC 15: Ligand Discovery for Everyone.",
    "BioRender": "BioRender. Scientific illustration software. biorender.com",
    "Tanimoto": "Tanimoto (1958). An Elementary Mathematical Theory of Classification and Prediction.",
    "Hugging Face": "BiSECt Group. Leak-CURBER dataset collection on Hugging Face.",
    "Harvard Dataverse": "BiSECt Group. Leak-CURBER data release. Harvard Dataverse. DOI 10.7910/DVN/CZW47K.",
    "ESM-C": "EvolutionaryScale. Evolutionary Scale · ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning."
});
  window.LC_MODEL_CITATIONS = citations;
  const tip = document.createElement('div');
  tip.id = 'lc-tip';
  tip.className = 'lc-tip';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  document.body.append(tip);
  let current = null;

  function show(button) {
    current = button;
    const dialog = button.closest('dialog');
    const host = dialog || document.body;
    if (tip.parentElement !== host) host.append(tip);
    tip.style.position = dialog ? 'fixed' : 'absolute';
    const definition = GLOSSARY[button.dataset.term];
    const citation = citations[button.dataset.term === "drfp" ? "DRFP" : button.dataset.term] || citations[button.textContent.trim()];
    tip.textContent = [definition, citation].filter(Boolean).join(' ');
    tip.hidden = false;
    const r = button.getBoundingClientRect();
    const width = tip.offsetWidth;
    const left = Math.min(Math.max(8, r.left + r.width / 2 - width / 2), document.documentElement.clientWidth - width - 8);
    tip.style.left = `${left + (dialog ? 0 : scrollX)}px`;
    const top = r.bottom + tip.offsetHeight + 6 <= innerHeight ? r.bottom + 6 : Math.max(4, r.top - tip.offsetHeight - 6);
    tip.style.top = `${top + (dialog ? 0 : scrollY)}px`;
  }
  function hide() {
    current = null;
    tip.hidden = true;
  }

  const pattern = new RegExp('\\b(?:' + Object.keys(citations).sort((a,b) => b.length-a.length).join('|') + ')\\b', 'g');
  function annotate(node) {
    const parent = node.parentElement;
    if (!parent || parent.closest('script,style,textarea,option,pre,code,.lc-buttons,#use .lc-links,#leaderboard table,.lc-term,#lc-tip,svg,canvas')) return;
    const matches = [...node.data.matchAll(pattern)].filter(match => {
      if (match[0] === 'DRFP') return Boolean(parent.closest('.lc-mod-model'));
      return !(match[0] === 'SMILES' && node.data.slice(0, match.index).endsWith('GR-'));
    });
    if (!matches.length) return;
    const fragment = document.createDocumentFragment();
    let start = 0;
    for (const match of matches) {
      fragment.append(node.data.slice(start, match.index));
      const button = document.createElement(parent.closest('a,button,label,summary') ? 'span' : 'button');
      if (button.tagName === 'BUTTON') button.type = 'button';
      else button.tabIndex = 0;
      button.className = 'lc-term lc-citation';
      button.dataset.term = match[0];
      button.setAttribute('aria-describedby', 'lc-tip');
      button.textContent = match[0];
      fragment.append(button);
      start = match.index + match[0].length;
    }
    fragment.append(node.data.slice(start));
    node.replaceWith(fragment);
  }
  function scan(root) {
    if (root.nodeType === Node.TEXT_NODE) return annotate(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(annotate);
  }
  scan(document.body);
  new MutationObserver(records => {
    for (const record of records) for (const node of record.addedNodes) {
      if (node.isConnected) scan(node);
    }
  }).observe(document.body, { childList: true, subtree: true });
  document.querySelectorAll('.lc-term').forEach(b => b.setAttribute('aria-describedby', 'lc-tip'));
  for (const event of ['pointerover','focusin','click']) {
    document.addEventListener(event, e => {
      const term = e.target.closest('.lc-term');
      if (term) show(term);
    });
  }
  for (const event of ['pointerout','focusout']) {
    document.addEventListener(event, e => { if (e.target === current) hide(); });
  }
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
  document.addEventListener('pointerdown', (e) => { if (current && e.target !== current) hide(); });
})();
