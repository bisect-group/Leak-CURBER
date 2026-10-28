// Toy illustration of random, similarity and time splits. No real data.
(() => {
  const host = document.getElementById('splits-demo');
  if (!host) return;

  const NS = 'http://www.w3.org/2000/svg';
  const W = 600, H = 340, PER = 10;
  const COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00'];
  const { YEARS } = window.LC;
  const CENTERS = {
    random: [[95, 90], [255, 75], [415, 100], [110, 230], [280, 245], [450, 235]],
    // Clusters 2 and 5 move to the test strip on the right.
    similarity: [[85, 90], [250, 80], [510, 95], [105, 235], [275, 245], [510, 240]],
  };
  const CAPTIONS = {
    random: 'Random split: test points (rings) sit next to near-identical training points. Lines show that leak.',
    similarity: 'Similarity split: whole clusters go to test. No training point is close to a test point.',
    time: 'Time split: the newest entries (2024) go to test. Older entries train the model.',
  };

  // Seeded generator, so the picture is the same on every load.
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

  const dots = Array.from({ length: COLORS.length * PER }, (_, i) => {
    const angle = rand() * 2 * Math.PI, r = 12 + rand() * 36;
    return { cluster: Math.floor(i / PER), dx: Math.cos(angle) * r, dy: Math.sin(angle) * r * 0.8 };
  });

  const years = YEARS.flatMap((y, k) => Array(k === 6 ? 7 : k === 5 ? 8 : 9).fill(y));
  years.sort(() => rand() - 0.5);
  dots.forEach((d, i) => { d.year = years[i]; d.jitter = rand() * 16 - 8; });

  const isTest = {
    random: dots.map((d, i) => i % PER < 2),
    similarity: dots.map((d) => d.cluster === 2 || d.cluster === 5),
    time: dots.map((d) => d.year === YEARS.at(-1)),
  };

  const position = {
    random: dots.map((d) => [CENTERS.random[d.cluster][0] + d.dx, CENTERS.random[d.cluster][1] + d.dy]),
    similarity: dots.map((d) => [CENTERS.similarity[d.cluster][0] + d.dx * 0.8, CENTERS.similarity[d.cluster][1] + d.dy * 0.8]),
    time: dots.map((d) => [50 + ((d.year - 2018) / 6) * 500 + d.jitter, 50 + d.cluster * 45 + d.jitter]),
  };

  // Each test dot links to its two nearest training dots in the random layout.
  const links = position.random.flatMap((p, i) => {
    if (!isTest.random[i]) return [];
    return position.random
      .map((q, j) => [j, Math.hypot(p[0] - q[0], p[1] - q[1])])
      .filter(([j]) => !isTest.random[j])
      .sort((a, b) => a[1] - b[1])
      .slice(0, 2)
      .map(([j]) => [i, j]);
  });

  // Strip behind the test region and the year axis, as numbers that animate: x, width, opacity.
  const shade = { random: [0, 0, 0], similarity: [428, 164, 1], time: [50 + (5.5 / 6) * 500, 90, 1] };
  const axisOpacity = { random: 0, similarity: 0, time: 1 };
  const linkOpacity = { random: 1, similarity: 0, time: 0 };

  const svgEl = (tag, attrs = {}) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };

  const svg = svgEl('svg', {
    viewBox: `0 0 ${W} ${H}`, role: 'img', class: 'lc-demo-svg',
    'aria-label': 'Illustration of random, similarity and time splits on coloured clusters of dots.',
  });
  const strip = svgEl('rect', { y: 28, height: H - 56, rx: 8, class: 'lc-demo-strip' });
  const stripLabel = svgEl('text', { y: 44, class: 'lc-demo-label', 'text-anchor': 'middle' });
  stripLabel.textContent = 'Test';
  const axis = svgEl('g', { class: 'lc-demo-axis' });
  YEARS.forEach((y, k) => {
    const t = svgEl('text', { x: 50 + (k / 6) * 500, y: H - 10, 'text-anchor': 'middle' });
    t.textContent = y;
    axis.append(t);
  });
  const lineEls = links.map(() => svgEl('line', { class: 'lc-demo-link' }));
  const dotEls = dots.map((d) => svgEl('circle', { r: 6, class: 'lc-demo-dot', style: `--dot:${COLORS[d.cluster]}` }));
  const badge = svgEl('text', { x: 12, y: 20, class: 'lc-demo-badge' });
  badge.textContent = 'Illustration';
  svg.append(strip, stripLabel, axis, ...lineEls, ...dotEls, badge);

  const caption = document.createElement('p');
  caption.className = 'lc-demo-caption';
  caption.setAttribute('aria-live', 'polite');
  const buttons = Object.keys(CAPTIONS).map((mode) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = { random: 'Random', similarity: 'Similarity', time: 'Time' }[mode];
    b.addEventListener('click', () => show(mode));
    b.dataset.mode = mode;
    return b;
  });
  const bar = document.createElement('div');
  bar.className = 'lc-demo-buttons';
  bar.setAttribute('role', 'group');
  bar.setAttribute('aria-label', 'Split type');
  bar.append(...buttons);
  const legend = document.createElement('p');
  legend.className = 'lc-demo-legend';
  legend.innerHTML = '<span class="lc-demo-key train"></span> Train <span class="lc-demo-key test"></span> Test';
  host.append(bar, svg, legend, caption);

  // Animated state: [x, y] per dot, then strip x, strip width, strip opacity, axis opacity, link opacity.
  const vectorFor = (mode) => [...position[mode].flat(), ...shade[mode], axisOpacity[mode], linkOpacity[mode]];
  let current = vectorFor('random');
  let frame = 0;

  function paint(v, mode) {
    dotEls.forEach((c, i) => {
      c.setAttribute('cx', v[2 * i]);
      c.setAttribute('cy', v[2 * i + 1]);
      c.classList.toggle('is-test', isTest[mode][i]);
    });
    const n = 2 * dots.length;
    strip.setAttribute('x', v[n]);
    strip.setAttribute('width', Math.max(v[n + 1], 0));
    strip.style.opacity = v[n + 2];
    stripLabel.setAttribute('x', v[n] + v[n + 1] / 2);
    stripLabel.style.opacity = v[n + 2];
    axis.style.opacity = v[n + 3];
    links.forEach(([a, b], k) => {
      lineEls[k].setAttribute('x1', v[2 * a]);
      lineEls[k].setAttribute('y1', v[2 * a + 1]);
      lineEls[k].setAttribute('x2', v[2 * b]);
      lineEls[k].setAttribute('y2', v[2 * b + 1]);
      lineEls[k].style.opacity = v[n + 4];
    });
  }

  function show(mode) {
    cancelAnimationFrame(frame);
    buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    caption.textContent = CAPTIONS[mode];
    const from = current, to = vectorFor(mode);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      current = to;
      paint(to, mode);
      return;
    }
    const start = performance.now(), duration = 700;
    const step = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const e = t < 0.5 ? 2 * t * t : 1 - ((-2 * t + 2) ** 2) / 2;
      current = from.map((a, i) => a + (to[i] - a) * e);
      paint(current, mode);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
  }

  show('random');
})();
