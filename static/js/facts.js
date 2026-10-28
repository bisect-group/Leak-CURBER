(() => {
  const root = document.documentElement;
  const fmt = (v) => (v === null || v === undefined ? "—" : typeof v === "number" ? v.toLocaleString("en-US") : String(v));
  const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);

  // Labels may contain <sub> only. Everything else stays text.
  function labelWithSubscripts(el, label) {
    el.textContent = "";
    label.split(/<sub>(.*?)<\/sub>/).forEach((part, i) => {
      el.append(i % 2 ? Object.assign(document.createElement("sub"), { textContent: part }) : part);
    });
  }

  function renderCards(el, items) {
    el.innerHTML = "";
    for (const it of items) {
      const card = document.createElement("div");
      card.className = "lc-card";
      const add = (cls, text) => {
        if (text == null) return;
        const d = document.createElement("div");
        d.className = cls;
        d.textContent = fmt(text);
        card.appendChild(d);
      };
      add("lc-label", it.label);
      labelWithSubscripts(card.querySelector(".lc-label"), it.label);
      add("lc-value", it.from);
      (it.to || []).forEach((t) => add("lc-to", "→ " + t));
      add("lc-sub", it.note);
      el.appendChild(card);
    }
  }

  fetch("data/facts.json")
    .then((r) => r.json())
    .then((facts) => {
      document.querySelectorAll("[data-fact]").forEach((el) => (el.textContent = fmt(get(facts, el.dataset.fact))));
      document.querySelectorAll("[data-stat]").forEach((el) => (el.textContent = fmt(get(facts, "stats.scale." + el.dataset.stat))));
      document.querySelectorAll("[data-cards]").forEach((el) => renderCards(el, get(facts, el.dataset.cards) || []));
      window.LC_FACTS = facts;
      document.dispatchEvent(new CustomEvent("lc:facts", { detail: facts }));
    })
    .catch((e) => console.error("facts.json failed to load", e));

  const toggle = document.getElementById("theme-toggle");
  const dark = () => root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const sync = () => {
    toggle.innerHTML = `<i class="fa-solid ${dark() ? "fa-sun" : "fa-moon"}"></i>`;
    toggle.setAttribute("aria-pressed", dark());
    toggle.title = dark() ? "Dark mode is on" : "Dark mode is off";
  };
  sync();
  toggle.addEventListener("click", () => {
    root.dataset.theme = dark() ? "light" : "dark";
    localStorage.setItem("lc-theme", root.dataset.theme);
    sync();
  });

  const copy = document.getElementById("copy-bib");
  copy.addEventListener("click", () => {
    navigator.clipboard.writeText(document.getElementById("bibtex").textContent).then(() => {
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1500);
    });
  });
})();
