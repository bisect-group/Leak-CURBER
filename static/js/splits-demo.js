// The similarity rule is deliberately simple enough to check by eye.
(() => {
  const demo = document.getElementById('splits-demo');
  const rows = [...demo.querySelectorAll('tbody tr')];
  const sequences = rows.map(row => row.querySelector('code').textContent);
  const years = rows.map(row => Number(row.cells[2].textContent));
  const closest = sequences.map((sequence, i) => Math.max(...sequences.map((other, j) =>
    i === j ? 0 : [...sequence].filter((letter, k) => letter === other[k]).length)));
  const modes = {
    random: {
      test: [1, 2], // One fixed example draw so readers can revisit the same comparison.
      caption: 'B and C go to test. Each has a close match in training.',
    },
    similarity: {
      test: rows.map((_, i) => i).filter(i => closest[i] < 4),
      caption: 'E and F go to test. Neither matches four letters elsewhere.',
    },
    time: {
      test: rows.map((_, i) => i).filter(i => years[i] === Math.max(...years)),
      caption: 'D and F go to test. Both records were added in 2024.',
    },
  };
  const buttons = [...demo.querySelectorAll('[data-mode]')];
  function show(mode) {
    const { test, caption } = modes[mode];
    for (const button of buttons) button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
    rows.forEach((row, i) => {
      const heldOut = test.includes(i);
      row.classList.toggle('lc-held-out', heldOut);
      row.cells[3].textContent = heldOut ? 'Test' : 'Train';
    });
    demo.querySelector('.lc-demo-caption').textContent = caption;
  }
  buttons.forEach(button => button.addEventListener('click', () => show(button.dataset.mode)));
  show('random');
})();
