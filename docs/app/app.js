/* CatFlow demo — frontend logic */
"use strict";

/* Backend base URL. The UI is served from GitHub Pages while the model runs
 * on a GPU server behind a cloudflared HTTPS tunnel. The tunnel URL is read at
 * startup from ./api_url.txt with a cache-buster, so a changed URL reaches
 * returning visitors even though Pages caches app.js for ~10 min. */
let API = "";
async function resolveAPI() {
  try {
    const r = await fetch("./api_url.txt?t=" + Date.now(), { cache: "no-store" });
    if (r.ok) {
      const u = (await r.text()).trim().replace(/\/+$/, "");
      if (u) API = u;
    }
  } catch (e) { /* keep API empty — loadMenus will surface the error */ }
}
const state = {
  mode: "denovo",
  adsorbates: [],
  compositions: [],
  adsorbate: null,
  composition: null,
  compSource: "dataset",       // "dataset" | "custom"
  customComposition: "",
  result: null,
  sampleIdx: 0,
  viewers: [],          // active NGL stages, disposed on sample switch
};

const VIEWER_BG = "#0e1118";   // dark molecular-viewer background

/* ---------------------------------------------------------------- helpers */
const $ = (id) => document.getElementById(id);
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
function disposeViewers() {
  state.viewers.forEach((v) => { try { v.dispose(); } catch (e) {} });
  state.viewers = [];
}

/* ------------------------------------------------------------ data loading */
async function loadMenus() {
  const [a, c] = await Promise.all([
    fetch(API + "/api/adsorbates").then((r) => r.json()),
    fetch(API + "/api/compositions").then((r) => r.json()),
  ]);
  state.adsorbates = a;
  state.compositions = c;
  renderAdsorbates("");
  renderCompositions("");
}

/* --------------------------------------------------------------- selectors */
function renderAdsorbates(query) {
  const box = $("ads-options");
  box.innerHTML = "";
  const q = query.trim().toLowerCase();
  const list = state.adsorbates.filter(
    (a) => !q || a.name.toLowerCase().includes(q) ||
           a.formula.toLowerCase().includes(q));
  list.slice(0, 200).forEach((a) => {
    const row = el("div", "opt");
    if (state.adsorbate && state.adsorbate.id === a.id) row.classList.add("sel");
    row.innerHTML =
      `<span><span class="o-main">${a.name}</span>` +
      ` <span class="o-formula o-sub">${a.display}</span></span>` +
      `<span class="o-sub">${a.n_atoms} atom${a.n_atoms > 1 ? "s" : ""}` +
      `${a.evaluable ? "" : ' <span class="tagx">no E</span>'}</span>`;
    row.onclick = () => {
      state.adsorbate = a;
      renderAdsorbates(query);
      updateGenBtn();
    };
    box.appendChild(row);
  });
  if (!list.length) box.appendChild(el("div", "opt o-sub", "No match"));
}

function renderCompositions(query) {
  const box = $("comp-options");
  box.innerHTML = "";
  const q = query.trim().toLowerCase();
  const list = state.compositions.filter(
    (c) => !q || c.formula.toLowerCase().includes(q) ||
           c.elements.join("").toLowerCase().includes(q));
  list.slice(0, 300).forEach((c) => {
    const row = el("div", "opt");
    if (state.composition && state.composition.id === c.id)
      row.classList.add("sel");
    row.innerHTML =
      `<span class="o-main o-formula">${c.display}</span>` +
      `<span class="o-sub">${c.n_atoms}-atom cell</span>`;
    row.onclick = () => {
      state.composition = c;
      renderCompositions(query);
      updateGenBtn();
    };
    box.appendChild(row);
  });
  if (!list.length) box.appendChild(el("div", "opt o-sub", "No match"));
}

/* -------------------------------------------------- custom composition */
const ELEMENTS = new Set(("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca " +
  "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh " +
  "Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb " +
  "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu").split(" "));

// "Pt3Ni" / "Pd Ag" / "Cu4"  ->  {ok, formula, n_atoms, error}
function parseCustomComposition(text) {
  const t = (text || "").trim();
  if (!t) return { ok: false };
  if (!/^[A-Za-z0-9\s]+$/.test(t))
    return { ok: false, error: "letters and numbers only" };
  const re = /([A-Z][a-z]?)\s*(\d*)/g;
  const counts = {};
  let n = 0, m;
  while ((m = re.exec(t)) !== null) {
    if (!m[1]) continue;
    if (!ELEMENTS.has(m[1]))
      return { ok: false, error: `unknown element “${m[1]}”` };
    const c = m[2] ? parseInt(m[2], 10) : 1;
    if (c < 1) return { ok: false, error: "count must be ≥ 1" };
    counts[m[1]] = (counts[m[1]] || 0) + c;
    n += c;
  }
  if (n === 0) return { ok: false, error: "could not read any element" };
  if (n > 48) return { ok: false, error: "max 48 atoms per cell" };
  const els = Object.keys(counts).sort();
  const formula = els.map((e) => e + (counts[e] > 1 ? counts[e] : "")).join("");
  return { ok: true, formula, n_atoms: n };
}

function renderCustomPreview() {
  const box = $("comp-custom-preview");
  const r = parseCustomComposition(state.customComposition);
  if (!state.customComposition.trim()) {
    box.className = "custom-preview";
    box.textContent = "";
  } else if (r.ok) {
    box.className = "custom-preview ok";
    box.innerHTML = `<span class="cp-formula">${r.formula}</span>` +
      `<span class="cp-meta">${r.n_atoms}-atom primitive cell</span>`;
  } else {
    box.className = "custom-preview bad";
    box.textContent = "⚠ " + (r.error || "enter a valid formula");
  }
}

function setCompSource(src) {
  state.compSource = src;
  document.querySelectorAll("#comp-source .subtab").forEach((t) =>
    t.classList.toggle("active", t.dataset.src === src));
  $("comp-dataset-panel").classList.toggle("hidden", src !== "dataset");
  $("comp-custom-panel").classList.toggle("hidden", src !== "custom");
  updateGenBtn();
}

function updateGenBtn() {
  let ok = !!state.adsorbate;
  if (state.mode === "structure") {
    ok = ok && (state.compSource === "dataset"
      ? !!state.composition
      : parseCustomComposition(state.customComposition).ok);
  }
  $("gen-btn").disabled = !ok;
}

/* -------------------------------------------------------------------- mode */
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.mode === mode));
  $("comp-card").classList.toggle("hidden", mode === "denovo");
  updateGenBtn();
}

/* ---------------------------------------------------------------- generate */
async function generate() {
  const btn = $("gen-btn");
  btn.disabled = true;
  $("gen-error").classList.add("hidden");
  $("gen-status").classList.remove("hidden");
  $("gen-status-text").textContent =
    "Running flow matching (50 steps, 10 samples)… this can take ~10 s.";
  $("result-card").classList.add("hidden");
  disposeViewers();

  try {
    const body = {
      mode: state.mode,
      adsorbate_id: state.adsorbate.id,
      composition_id: null,
      custom_composition: null,
    };
    if (state.mode === "structure") {
      if (state.compSource === "custom")
        body.custom_composition = state.customComposition;
      else
        body.composition_id = state.composition ? state.composition.id : null;
    }
    const r = await fetch(API + "/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    state.result = await r.json();
    state.sampleIdx = 0;
    renderResult();
  } catch (e) {
    $("gen-error").textContent = "⚠ " + e.message;
    $("gen-error").classList.remove("hidden");
  } finally {
    $("gen-status").classList.add("hidden");
    btn.disabled = false;
  }
}

/* ----------------------------------------------------------------- results */
function renderResult() {
  const res = state.result;
  $("result-card").classList.remove("hidden");

  const sum = $("result-summary");
  sum.innerHTML = "";
  sum.appendChild(el("span", "pill ok",
    `${res.n_valid} / ${res.n_generated} valid samples`));
  sum.appendChild(el("span", "pill",
    `Adsorbate ${res.adsorbate.name}`));
  if (res.composition)
    sum.appendChild(el("span", "pill", `Slab ${res.composition.display}`));

  const strip = $("sample-strip");
  strip.innerHTML = "";
  if (!res.samples.length) {
    $("sample-view").innerHTML =
      '<div class="hint" style="margin-top:12px;">' +
      "No structurally valid sample this run — try generating again.</div>";
    return;
  }
  res.samples.forEach((s, i) => {
    const chip = el("div", "chip", `Sample ${i + 1}`);
    if (i === state.sampleIdx) chip.classList.add("active");
    chip.onclick = () => { state.sampleIdx = i; renderResult(); };
    strip.appendChild(chip);
  });
  renderSample(state.sampleIdx);
}

function renderSample(idx) {
  disposeViewers();
  const sample = state.result.samples[idx];
  const host = $("sample-view");
  host.innerHTML = "";

  /* ---- window 1 : final structure ---- */
  const b1 = el("div", "viewer-block");
  b1.appendChild(el("div", "vlabel",
    `<span>Final structure · ${sample.final.formula} · ` +
    `${sample.final.n_atoms} atoms</span>`));
  const vp1 = el("div", "viewport");
  vp1.id = "vp-final-" + idx;
  b1.appendChild(vp1);
  const ctrl1 = el("div", "vctrl");
  const bCell = el("button", "ghost", "Unit cell");
  const bSuper = el("button", "ghost", "2×2 supercell");
  const bSpin = el("button", "ghost", "Spin");
  ctrl1.append(bCell, bSuper, bSpin);
  b1.appendChild(ctrl1);
  b1.appendChild(el("div", "legend",
    "Drag to rotate · pinch / scroll to zoom · " +
    "atoms coloured by element."));
  host.appendChild(b1);

  /* ---- window 2 : trajectory ---- */
  const b2 = el("div", "viewer-block");
  const nFrames = sample.trajectory.length;
  b2.appendChild(el("div", "vlabel",
    `<span>Generation trajectory · ${nFrames} valid frames</span>`));
  const vp2 = el("div", "viewport traj");
  vp2.id = "vp-traj-" + idx;
  b2.appendChild(vp2);
  const ctrl2 = el("div", "vctrl");
  const bPlay = el("button", "ghost", "▶ Play");
  const slider = el("input", "slider");
  slider.type = "range";
  slider.min = 0;
  slider.max = nFrames - 1;
  slider.value = nFrames - 1;
  const fl = el("span", "frame-label");
  ctrl2.append(bPlay, slider, fl);
  b2.appendChild(ctrl2);
  host.appendChild(b2);

  /* ---- eval ---- */
  const evalBtn = el("button", "eval", "⚡ E eval — relax &amp; compute ΔE_ads");
  if (!state.result.adsorbate.evaluable) {
    evalBtn.disabled = true;
    evalBtn.textContent = "E eval unavailable for this adsorbate";
  }
  host.appendChild(evalBtn);
  const evalStatus = el("div", "status hidden");
  evalStatus.style.marginTop = "10px";
  evalStatus.innerHTML = '<div class="spinner"></div><span>Relaxing…</span>';
  host.appendChild(evalStatus);
  const evalErr = el("div", "err hidden");
  evalErr.style.marginTop = "8px";
  host.appendChild(evalErr);
  const energyBox = el("div", "energy hidden");
  host.appendChild(energyBox);

  /* ---- build viewers ---- */
  const finalViewer = makeViewer(vp1.id, sample.final);
  state.viewers.push(finalViewer.stage);
  bCell.onclick = () => {
    const on = finalViewer.toggleCell();
    bCell.classList.toggle("on", on);
  };
  bSuper.onclick = () => {
    const on = finalViewer.toggleSupercell();
    bSuper.classList.toggle("on", on);
    bSuper.textContent = on ? "Hide supercell" : "2×2 supercell";
  };
  bSpin.onclick = () => {
    const on = finalViewer.toggleSpin();
    bSpin.classList.toggle("on", on);
    bSpin.textContent = on ? "Stop" : "Spin";
  };

  const traj = makeTrajViewer(vp2.id, sample.trajectory);
  state.viewers.push(traj.stage);
  function showFrame(i) {
    const fr = sample.trajectory[i];
    fl.textContent = `step ${fr.step} / ${sample.n_traj_total - 1}`;
    traj.show(i);
  }
  slider.oninput = () => { stopPlay(); showFrame(+slider.value); };
  showFrame(nFrames - 1);

  let playTimer = null;
  function stopPlay() {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
    bPlay.textContent = "▶ Play";
    bPlay.classList.remove("on");
  }
  bPlay.onclick = () => {
    if (playTimer) { stopPlay(); return; }
    bPlay.textContent = "❚❚ Pause";
    bPlay.classList.add("on");
    let i = 0;
    slider.value = 0; showFrame(0);
    playTimer = setInterval(() => {
      i++;
      if (i >= nFrames) { stopPlay(); return; }
      slider.value = i; showFrame(i);
    }, 130);
  };

  /* ---- eval action ---- */
  evalBtn.onclick = async () => {
    evalBtn.disabled = true;
    evalErr.classList.add("hidden");
    energyBox.classList.add("hidden");
    evalStatus.classList.remove("hidden");
    try {
      const r = await fetch(API + "/api/eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: state.result.job_id,
          sample_idx: idx,
          adsorbate_id: state.result.adsorbate.id,
        }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      renderEnergy(energyBox, await r.json());
      energyBox.classList.remove("hidden");
    } catch (e) {
      evalErr.textContent = "⚠ " + e.message;
      evalErr.classList.remove("hidden");
    } finally {
      evalStatus.classList.add("hidden");
      evalBtn.disabled = !state.result.adsorbate.evaluable;
    }
  };
}

/* -------------------------------------------------------------- energy view */
function renderEnergy(box, ev) {
  box.innerHTML = "";
  box.appendChild(el("div", "ev-formula",
    `<b>${ev.chemical_formula}</b>`));
  const grid = el("div", "ev-grid");
  const cell = (k, v, hi) => {
    const c = el("div", "ev-cell" + (hi ? " hi" : ""));
    c.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    return c;
  };
  grid.appendChild(cell("ΔE_ads (relaxed)",
    ev.e_ads_relaxed.toFixed(2) + " eV", true));
  grid.appendChild(cell("ΔE_ads (initial)",
    ev.e_ads_initial.toFixed(2) + " eV"));
  grid.appendChild(cell("E_system (relaxed)",
    ev.e_sys_relaxed.toFixed(2) + " eV"));
  grid.appendChild(cell("E_slab (relaxed)",
    ev.e_slab_relaxed.toFixed(2) + " eV"));
  box.appendChild(grid);
  box.appendChild(el("div", "legend",
    `Relaxed in ${ev.relax_steps} LBFGS steps · ` +
    `ΔE_ads = E_sys − E_slab − E_adsorbate.`));

  if (ev.volcano) drawVolcano(box, ev.volcano, ev.adsorbate_name);
}

function drawVolcano(box, v, adsName) {
  const wrap = el("div", "volcano");
  wrap.appendChild(el("div", "vt",
    `Activity volcano · ${adsName}`));
  wrap.appendChild(el("div", "vd", v.descriptor));

  const W = 320, H = 170, padL = 34, padR = 12, padT = 12, padB = 30;
  const xs = v.curve_x, ys = v.curve_y;
  const xmin = Math.min(...xs, v.point_x), xmax = Math.max(...xs, v.point_x);
  const X = (x) => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
  const Y = (y) => padT + (1 - y) * (H - padT - padB);

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("class", "vplot");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const mk = (tag, attrs) => {
    const e = document.createElementNS(svgNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  };
  const C = {
    axis: "#39414f", grid: "#262d3a", flow: "#46e0c8",
    opt: "#73d99a", point: "#ff6a4d", ink: "#8b93a7", bg: "#0e1118",
  };
  // fill under curve
  let area = "M" + X(xs[0]) + " " + Y(0) + " ";
  xs.forEach((x, i) => { area += "L" + X(x) + " " + Y(ys[i]) + " "; });
  area += "L" + X(xs[xs.length - 1]) + " " + Y(0) + " Z";
  svg.appendChild(mk("path", { d: area, fill: "rgba(70,224,200,0.10)" }));
  // axes
  svg.appendChild(mk("line", { x1: padL, y1: H - padB, x2: W - padR,
    y2: H - padB, stroke: C.axis }));
  svg.appendChild(mk("line", { x1: padL, y1: padT, x2: padL,
    y2: H - padB, stroke: C.axis }));
  // curve
  let d = "";
  xs.forEach((x, i) => { d += (i ? "L" : "M") + X(x) + " " + Y(ys[i]) + " "; });
  svg.appendChild(mk("path", { d, fill: "none", stroke: C.flow,
    "stroke-width": 2.4, "stroke-linejoin": "round" }));
  // optimum marker
  svg.appendChild(mk("line", { x1: X(v.optimum), y1: padT,
    x2: X(v.optimum), y2: H - padB, stroke: C.opt,
    "stroke-dasharray": "4 3", "stroke-width": 1 }));
  // computed point
  svg.appendChild(mk("circle", { cx: X(v.point_x), cy: Y(v.point_y),
    r: 6.5, fill: C.point, stroke: C.bg, "stroke-width": 2.5 }));
  // label for point
  const lbl = mk("text", { x: X(v.point_x), y: Y(v.point_y) - 11,
    "text-anchor": "middle", "font-size": 11, fill: C.point,
    "font-weight": "bold", "font-family": "'IBM Plex Mono', monospace" });
  lbl.textContent = v.point_x.toFixed(2) + " eV";
  svg.appendChild(lbl);
  // axis labels
  const tx = mk("text", { x: (W + padL) / 2, y: H - 6,
    "text-anchor": "middle", "font-size": 10, fill: C.ink });
  tx.textContent = "binding energy (eV)  →";
  svg.appendChild(tx);
  const ty = mk("text", { x: 11, y: H / 2, "font-size": 10,
    fill: C.ink, transform: `rotate(-90 11 ${H / 2})`,
    "text-anchor": "middle" });
  ty.textContent = "activity";
  svg.appendChild(ty);
  // optimum tag
  const ot = mk("text", { x: X(v.optimum), y: padT + 9,
    "text-anchor": "middle", "font-size": 9, fill: C.opt });
  ot.textContent = "optimum";
  svg.appendChild(ot);

  wrap.appendChild(svg);
  wrap.appendChild(el("div", "legend",
    "Illustrative scaling-relation volcano — the red point marks this " +
    "catalyst's computed binding energy."));
  box.appendChild(wrap);
}

/* --------------------------------------------------------------- NGL view */
const ELEM_COLOR = "element";

function structureRepresentations(comp, tags, supercellGhost) {
  // element-coloured spacefill spheres, matching the project-page viewer.
  // every atom keeps its true element colour — the adsorbate is not recoloured.
  comp.addRepresentation("spacefill", {
    colorScheme: ELEM_COLOR,
    radiusScale: supercellGhost ? 0.42 : 0.5,
    opacity: supercellGhost ? 0.4 : 1.0,
  });
}

function makeViewer(elemId, payload) {
  const stage = new NGL.Stage(elemId, { backgroundColor: VIEWER_BG });
  const blob = new Blob([payload.pdb], { type: "text/plain" });
  let mainComp = null, cellRep = null;
  let superComps = [], superOn = false, spinning = false;

  stage.loadFile(blob, { ext: "pdb", defaultRepresentation: false })
    .then((comp) => {
      mainComp = comp;
      structureRepresentations(comp, payload.tags, false);
      cellRep = comp.addRepresentation("unitcell",
        { colorValue: "#888", visible: false });
      comp.autoView();
    });

  window.addEventListener("resize", () => stage.handleResize());

  return {
    stage,
    toggleCell() {
      if (!cellRep) return false;
      const vis = !cellRep.visible;
      cellRep.setVisibility(vis);
      return vis;
    },
    toggleSpin() {
      spinning = !spinning;
      stage.setSpin(spinning ? [0, 1, 0] : null, spinning ? 0.012 : null);
      return spinning;
    },
    toggleSupercell() {
      if (!mainComp) return false;
      if (!superOn) {
        if (!superComps.length && payload.pdb_super) {
          // 2x2x1 supercell tiled server-side in ASE (consistent cell+coords).
          // Rendered as a translucent ghost over the solid single cell; the
          // ghost copies of the original atoms are smaller (radiusScale 0.42
          // < 0.5) so they hide inside the solid spheres, leaving only the
          // replica atoms visibly ghosted.
          stage.loadFile(new Blob([payload.pdb_super], { type: "text/plain" }),
            { ext: "pdb", defaultRepresentation: false }).then((cl) => {
              structureRepresentations(cl, payload.tags, true);
              superComps.push(cl);
            });
        } else superComps.forEach((c) => c.setVisibility(true));
        superOn = true;
        setTimeout(() => stage.autoView(400), 250);
      } else {
        superComps.forEach((c) => c.setVisibility(false));
        superOn = false;
        stage.autoView(400);
      }
      return superOn;
    },
  };
}

function makeTrajViewer(elemId, frames) {
  const stage = new NGL.Stage(elemId, { backgroundColor: VIEWER_BG });
  let firstView = true;
  window.addEventListener("resize", () => stage.handleResize());

  function show(i) {
    const fr = frames[i];
    stage.removeAllComponents();
    const blob = new Blob([fr.pdb], { type: "text/plain" });
    stage.loadFile(blob, { ext: "pdb", defaultRepresentation: false })
      .then((comp) => {
        structureRepresentations(comp, fr.tags, false);
        if (firstView) { comp.autoView(); firstView = false; }
      });
  }
  return { stage, show };
}

/* --------------------------------------------------------------------- init */
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => setMode(t.dataset.mode);
});
document.querySelectorAll("#comp-source .subtab").forEach((t) => {
  t.onclick = () => setCompSource(t.dataset.src);
});
$("ads-search").oninput = (e) => renderAdsorbates(e.target.value);
$("comp-search").oninput = (e) => renderCompositions(e.target.value);
$("comp-custom").oninput = (e) => {
  state.customComposition = e.target.value;
  renderCustomPreview();
  updateGenBtn();
};
$("gen-btn").onclick = generate;

/* theme toggle — initial data-theme is set by an inline <head> script */
(function initTheme() {
  const btn = $("theme-toggle");
  const cur = () =>
    document.documentElement.getAttribute("data-theme") || "light";
  const paint = () => { if (btn) btn.textContent = cur() === "light" ? "☾" : "☀"; };
  paint();
  if (btn) btn.onclick = () => {
    const next = cur() === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("catflow-theme", next); } catch (e) {}
    paint();
  };
})();

resolveAPI().then(loadMenus).catch((e) => {
  $("gen-error").textContent = "Failed to load menus: " + e.message;
  $("gen-error").classList.remove("hidden");
});
