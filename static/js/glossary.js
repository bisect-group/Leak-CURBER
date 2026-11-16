// Plain-language hints for technical terms. Mark a term in HTML as
// <button type="button" class="lc-term" data-term="kcat">...</button>.
(() => {
  const GLOSSARY = {
    enzyme: 'A protein that speeds up a chemical reaction in a living cell.',
    substrate: 'The molecule an enzyme acts on.',
    ec: 'A four-level code, from a broad class to the exact function.',
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
  };

  const tip = document.createElement('div');
  tip.id = 'lc-tip';
  tip.className = 'lc-tip';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  document.body.append(tip);
  let current = null;

  function show(button) {
    current = button;
    tip.textContent = GLOSSARY[button.dataset.term];
    tip.hidden = false;
    const r = button.getBoundingClientRect();
    const width = tip.offsetWidth;
    const left = Math.min(Math.max(8, r.left + r.width / 2 - width / 2), document.documentElement.clientWidth - width - 8);
    tip.style.left = `${left + scrollX}px`;
    tip.style.top = `${r.bottom + scrollY + 6}px`;
  }
  function hide() {
    current = null;
    tip.hidden = true;
  }

  document.querySelectorAll('.lc-term').forEach((button) => {
    button.setAttribute('aria-describedby', 'lc-tip');
    for (const type of ['mouseenter', 'focus', 'click']) button.addEventListener(type, () => show(button));
    for (const type of ['mouseleave', 'blur']) button.addEventListener(type, hide);
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
  document.addEventListener('pointerdown', (e) => { if (current && e.target !== current) hide(); });
})();
