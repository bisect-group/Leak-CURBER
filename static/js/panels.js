// Sections as horizontal panels. Without JavaScript the sections stack and the nav links are plain anchors.
(() => {
  const nav = document.querySelector(".lc-nav");
  const header = document.querySelector(".lc-header");
  new IntersectionObserver(() => {
    nav.classList.toggle("is-stuck", header.getBoundingClientRect().bottom <= 0);
  }, { threshold: [0, 0.001] }).observe(header);
  const viewport = document.querySelector(".lc-viewport");
  const track = viewport.firstElementChild;
  const panels = [...track.querySelectorAll(":scope > .lc-panel")];
  const tabs = [...nav.querySelectorAll("a")];
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const behavior = () => (reduced.matches ? "auto" : "smooth");
  let active = -1;
  let lastTop = null;

  nav.querySelector("ul").setAttribute("role", "tablist");
  nav.querySelectorAll("li").forEach((li) => li.setAttribute("role", "presentation"));
  tabs.forEach((a, i) => {
    a.id = "tab-" + panels[i].id;
    a.setAttribute("role", "tab");
    a.setAttribute("aria-controls", panels[i].id);
    panels[i].setAttribute("role", "tabpanel");
    panels[i].setAttribute("aria-labelledby", a.id);
  });

  let settle = null; // re-scrolls to the hash target while the page is still loading
  const size = new ResizeObserver(() => {
    viewport.style.height = panels[active].offsetHeight + "px";
    settle?.();
  });

  function step(label, i) {
    const b = Object.assign(document.createElement("button"), { type: "button", className: "lc-step", textContent: `${label}: ${tabs[i].textContent}` });
    b.dataset.go = panels[i].id;
    return b;
  }
  panels.forEach((p, i) => {
    const bar = Object.assign(document.createElement("div"), { className: "lc-steps" });
    if (i > 0) bar.append(step("Previous", i - 1));
    if (i < panels.length - 1) bar.append(Object.assign(step("Next", i + 1), { className: "lc-step lc-next" }));
    p.append(bar);
  });

  function activate(i, rise = "") {
    if (i === active) return;
    track.dataset.mode = rise ? "vertical" : "horizontal";
    if (active >= 0) size.unobserve(panels[active]);
    active = i;
    panels.forEach((p, k) => {
      p.inert = k !== i;
      if (k === i) p.removeAttribute("aria-hidden");
      else p.setAttribute("aria-hidden", "true");
    });
    tabs.forEach((a, k) => {
      if (k === i) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
      a.setAttribute("aria-selected", k === i);
      a.tabIndex = k === i ? 0 : -1;
    });
    track.style.transform = `translateX(${-100 * i}%)`;
    if (rise) {
      panels[i].dataset.rise = rise;
      panels[i].addEventListener("animationend", () => delete panels[i].dataset.rise, { once: true });
    }
    viewport.style.height = panels[i].offsetHeight + "px";
    size.observe(panels[i]);
    nav.scrollTo({ left: tabs[i].offsetLeft - (nav.clientWidth - tabs[i].offsetWidth) / 2, behavior: behavior() });
    clearTimeout(activate.timer);
    activate.timer = setTimeout(() => window.dispatchEvent(new Event("resize")), reduced.matches ? 0 : 400);
  }

  // Open the panel that holds the element, then scroll to it just below the sticky bar.
  function go(id, { push = false, replace = false, smooth = true, rise = "", settling = false } = {}) {
    if (!settling) settle = null;
    const target = document.getElementById(id);
    const panel = target?.closest(".lc-panel");
    if (!panel) return false;
    activate(panels.indexOf(panel), rise);
    for (let d = target.closest("details"); d; d = d.parentElement.closest("details")) d.open = true;
    if ((push || replace) && location.hash !== "#" + id) history[push ? "pushState" : "replaceState"](null, "", "#" + id);
    const at = target === panel ? viewport : target;
    lastTop = Math.max(0, Math.round(at.getBoundingClientRect().top + scrollY - nav.offsetHeight - (at === viewport ? 0 : 8)));
    scrollTo({ top: lastTop, behavior: smooth ? behavior() : "auto" });
    if (!smooth) lastTop = scrollY;
    return true;
  }

  document.addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="#"], button[data-go]');
    const id = a?.dataset.go || a?.getAttribute("href").slice(1);
    if (!id || e.ctrlKey || e.metaKey || e.shiftKey || e.button) return;
    const step = a.dataset.go ? { replace: true, smooth: false, rise: panels.indexOf(document.getElementById(id)) > active ? "down" : "up" } : { push: true };
    if (go(id, step)) e.preventDefault();
  });

  nav.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(document.activeElement);
    const to = { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (i < 0 || to === undefined) return;
    e.preventDefault();
    const n = (to + tabs.length) % tabs.length;
    tabs[n].focus();
    go(panels[n].id, { push: true });
  });

  // Scrolling past the end of a panel slides to the next one. Past the top of the page it slides back.
  const PUSH = 100, DWELL = 250, IDLE = 800, LOCK = 600;
  let acc = 0, dir = 0, lastIntent = 0, settledAt = 0, lockedUntil = 0, touchY = null;
  const atBottom = () => active < panels.length - 1 && panels[active].getBoundingClientRect().bottom <= innerHeight + 2;
  const atTop = () => active > 0 && scrollY <= 1;
  const ignored = (el) => {
    const t = document.getElementById("lc-tip");
    return !el.closest || el.closest("#results-explorer, input, textarea, select, [contenteditable], pre") || (t && !t.hidden) ||
      (el.closest(".lc-scroll") && el.closest(".lc-scroll").scrollWidth > el.closest(".lc-scroll").clientWidth);
  };
  function slide(d) {
    lockedUntil = performance.now() + LOCK;
    acc = 0;
    const to = panels[active + d];
    if (d > 0) return go(to.id, { replace: true, smooth: false, rise: "down" });
    go(to.id, { replace: true, smooth: false, rise: "up" });
    scrollTo({ top: viewport.getBoundingClientRect().bottom + scrollY - innerHeight, behavior: "auto" });
  }
  function push(d, amount) {
    const now = performance.now();
    if (now < lockedUntil || now - settledAt < DWELL) return false;
    if (d !== dir || now - lastIntent > IDLE) { acc = 0; }
    dir = d; lastIntent = now; acc += amount;
    if (acc >= PUSH) slide(d);
    return true;
  }
  addEventListener("scroll", () => { settledAt = performance.now(); });
  addEventListener("wheel", (e) => {
    const d = Math.sign(e.deltaY);
    if (!d || e.ctrlKey || ignored(e.target) || !(d > 0 ? atBottom() : atTop())) return;
    if (push(d, Math.abs(e.deltaY) * (e.deltaMode ? 16 : 1)) || d > 0) e.preventDefault();
  }, { passive: false });
  addEventListener("keydown", (e) => {
    const d = { PageDown: 1, " ": 1, ArrowDown: 1, End: 1, PageUp: -1, ArrowUp: -1, Home: -1 }[e.key];
    if (!d || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.target.closest(".lc-nav, button, a, summary, [role=tab]") || ignored(e.target) || !(d > 0 ? atBottom() : atTop())) return;
    if (push(d, e.key === "ArrowDown" || e.key === "ArrowUp" ? 40 : PUSH) || d > 0) e.preventDefault();
  });
  addEventListener("touchstart", (e) => { touchY = e.touches.length === 1 && !ignored(e.target) ? e.touches[0].clientY : null; }, { passive: true });
  addEventListener("touchend", (e) => {
    if (touchY === null) return;
    const dy = touchY - e.changedTouches[0].clientY, d = Math.sign(dy);
    touchY = null;
    if (Math.abs(dy) >= 80 && performance.now() >= lockedUntil && (d > 0 ? atBottom() : atTop())) slide(d);
  }, { passive: true });

  addEventListener("popstate", () => { if (!go(location.hash.slice(1), { smooth: true })) go(panels[0].id, { smooth: true }); });
  history.scrollRestoration = "manual";

  const start = location.hash.slice(1);
  activate(Math.max(0, panels.indexOf(document.getElementById(start)?.closest(".lc-panel"))));
  if (start) {
    go(start, { smooth: false });
    settle = () => { if (Math.abs(scrollY - lastTop) < 3) go(start, { smooth: false, settling: true }); else settle = null; };
    setTimeout(() => (settle = null), 4000);
  }
  requestAnimationFrame(() => requestAnimationFrame(() => track.classList.add("is-ready")));
})();
