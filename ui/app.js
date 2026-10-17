// WaferSight UI. Talks only to the API (same server; use ?api=http://host:8000 for a different one).
const API = new URLSearchParams(location.search).get("api") || "";
const LANG = "en";
const CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"];
const CLASS_COLORS = {
  "Center": "#7c3aed", "Donut": "#c026d3", "Edge-Loc": "#ea580c", "Edge-Ring": "#dc2626", "Loc": "#ca8a04",
  "Near-full": "#9f1239", "Random": "#0891b2", "Scratch": "#4f46e5", "none": "#10b981", "necunoscut": "#d97706",
};
const $ = (id) => document.getElementById(id);
const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const label = (c) => (c === "none" ? "No defect" : c === "necunoscut" ? "Unknown" : c);
const color = (c) => CLASS_COLORS[c] || "#64748b";
const icon = (name) => `<svg><use href="#i-${name}"/></svg>`;
const cls = (c) => `<span class="cls"><i style="background:${color(c)}"></i>${esc(label(c))}</span>`;
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;

const state = {
  lots: [], tools: {}, src: "lot", current: null, cost: null, uploads: [], upMap: {},
  params: { pret_wafer: 1500, ore_oprire: 2, cost_ora: 5000 }, overviewLoaded: false,
};

// ---------- API ----------
async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not JSON */ }
    throw new Error(`${r.status} ${msg}`);
  }
  return r.json();
}

function loading(on, title = "Analyzing…", sub = "") {
  $("loading-text").textContent = title;
  $("loading-sub").textContent = sub;
  $("loading").classList.toggle("hidden", !on);
}

let toastTimer;
function toast(msg, kind = "error") {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (kind === "info" ? " info" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 5000);
}

// ---------- wafer rendering ----------
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function drawWafer(canvas, map) {
  if (!canvas || !map || !map.length) return;
  const n = map.length, m = map[0].length;
  const px = Math.max(4, Math.floor(512 / Math.max(n, m)));
  const pad = px;
  canvas.width = m * px + 2 * pad;
  canvas.height = n * px + 2 * pad;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const ok = cssVar("--die-ok"), fail = cssVar("--die-fail");
  const gap = px >= 6 ? 1 : 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const v = map[i][j];
      if (!v) continue;
      ctx.fillStyle = v === 2 ? fail : ok;
      ctx.fillRect(pad + j * px, pad + i * px, px - gap, px - gap);
    }
  }
  // wafer edge + notch
  ctx.strokeStyle = cssVar("--wafer-edge");
  ctx.lineWidth = Math.max(1.5, px / 3);
  const cx = canvas.width / 2, cy = canvas.height / 2, r = Math.min(cx, cy) - pad / 2;
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI / 2 + 0.06, Math.PI / 2 - 0.06 + 2 * Math.PI);
  ctx.stroke();
}

// ---------- navigation ----------
function showView(name) {
  document.querySelectorAll(".nav-item").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  window.scrollTo({ top: 0 });
  if (name === "feedback") loadFeedback();
  if (name === "overview" && !state.overviewLoaded) loadOverview();
}
document.querySelectorAll(".nav-item").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));

document.querySelectorAll("#src-seg button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.src)));
function setSource(src) {
  state.src = src;
  if (src === "uploaded") refreshUploadSelect();
  document.querySelectorAll("#src-seg button").forEach((x) => x.classList.toggle("active", x.dataset.src === src));
  document.querySelectorAll(".src-fields").forEach((f) => f.classList.toggle("hidden", f.dataset.src !== src));
}

// ---------- startup ----------
async function init() {
  $("fb-class").innerHTML = CLASSES.map((c) => `<option value="${c}">${label(c)}</option>`).join("");
  syncSliders();
  try {
    const h = await api("/health");
    setSys("st-model", h.mode === "model" ? "on" : "warn", h.mode === "model" ? `${(h.history / 1000).toFixed(1)}k wafers` : "Demo data");
    setSys("st-rag", h.rag ? "on" : "off", h.rag ? "Online" : "Offline");
    setSys("st-llm", h.llm_model ? "on" : "off", h.llm_model || "Offline");
    $("a-hist").max = Math.max(0, h.history - 1);
    $("a-hist-info").textContent = h.history ? `0 – ${(h.history - 1).toLocaleString("en-US")}` : "";
    if (h.error) toast("Vision model unavailable; the API is serving demo data. " + h.error);
  } catch (e) {
    ["st-model", "st-rag", "st-llm"].forEach((id) => setSys(id, "off", "Offline"));
    toast("Cannot reach the API: " + e.message);
    $("o-lots").innerHTML = `<tr><td colspan="7" class="empty-row">API unavailable.</td></tr>`;
    return;
  }
  try { state.lots = await api("/lots"); } catch { state.lots = []; }
  state.lots.forEach((l) => { state.tools[l.lot_id] = l.echipament; });
  const opts = state.lots.map((l) => `<option value="${esc(l.lot_id)}">Lot ${esc(l.lot_id)}${l.echipament ? " · " + esc(l.echipament) : ""}</option>`).join("");
  $("a-lot").innerHTML = opts;
  $("l-lot").innerHTML = opts;
  fillWafers();
  lotInfo();
  if (state.lots.some((l) => l.lot_id === "B")) { $("a-lot").value = "B"; fillWafers(); $("a-wafer").value = "B_w10"; }
  loadOverview();
}

function setSys(id, cls, text) {
  const row = $(id);
  row.className = "sys-row " + cls;
  row.querySelector("b").textContent = text;
  row.title = text;
}

function fillWafers() {
  const lot = state.lots.find((l) => l.lot_id === $("a-lot").value);
  $("a-wafer").innerHTML = (lot?.wafer_ids || []).map((w, i) => `<option value="${esc(w)}">#${String(i + 1).padStart(2, "0")} · ${esc(w)}</option>`).join("");
}
$("a-lot").addEventListener("change", fillWafers);

function lotInfo() {
  const lot = state.lots.find((l) => l.lot_id === $("l-lot").value);
  $("l-equip").textContent = lot ? `${lot.n_wafers} wafers · tool ${lot.echipament || "n/a"}` : "";
}
$("l-lot").addEventListener("change", lotInfo);


// ---------- overview ----------
async function loadOverview() {
  if (!state.lots.length) return;
  state.overviewLoaded = true;
  $("o-refresh").disabled = true;
  try {
    const results = await Promise.all(state.lots.map((l) =>
      api("/lot", { lot_id: l.lot_id, use_llm: false, include_maps: false, cost: state.params, lang: LANG })));
    renderOverview(results);
    $("o-updated").textContent = "Updated " + new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  } catch (e) {
    state.overviewLoaded = false;
    toast("Could not load the fab overview: " + e.message);
  } finally {
    $("o-refresh").disabled = false;
  }
}
$("o-refresh").addEventListener("click", loadOverview);

function dominant(wafers) {
  const c = {};
  wafers.forEach((w) => { if (w.class_name !== "none") c[w.class_name] = (c[w.class_name] || 0) + 1; });
  const top = Object.entries(c).sort((a, b) => b[1] - a[1])[0];
  return top ? { cls: top[0], n: top[1] } : null;
}

function renderOverview(results) {
  const all = results.flatMap((r) => r.wafers);
  const defects = all.filter((w) => w.class_name !== "none").length;
  const alarms = results.filter((r) => r.alarm);
  const holds = results.filter((r) => r.cost?.recomandare === "opreste").length;
  $("o-kpis").innerHTML = `
    <div class="kpi"><div class="lbl">Lots monitored</div><div class="val">${results.length}</div><div class="sub">${all.length} wafers inspected</div></div>
    <div class="kpi ${defects / all.length > 0.3 ? "warn" : ""}"><div class="lbl">Defect rate</div><div class="val">${pct(defects / all.length, 1)}</div><div class="sub">${defects} wafers with a defect pattern</div></div>
    <div class="kpi ${alarms.length ? "bad" : "ok"}"><div class="lbl">Active drift alarms</div><div class="val">${alarms.length}</div><div class="sub">${alarms.length ? "Lots " + alarms.map((r) => r.lot_id).join(", ") : "All lots stable"}</div></div>
    <div class="kpi ${holds ? "bad" : "ok"}"><div class="lbl">Recommended line holds</div><div class="val">${holds}</div><div class="sub">at current cost assumptions</div></div>`;

  $("o-lots").innerHTML = results.map((r) => {
    const d = r.wafers.filter((w) => w.class_name !== "none").length;
    const dom = dominant(r.wafers);
    const status = r.alarm
      ? `<span class="chip bad">${icon("alert")}Drift · ${esc(label(r.repeated_class))} from #${r.first_alarm_index}</span>`
      : `<span class="chip ok">${icon("ok")}Stable</span>`;
    const seq = r.wafers.map((w) => `<span class="${w.alarm ? "alarm" : ""}" style="background:${color(w.class_name)}" title="#${w.index} ${esc(label(w.class_name))} (${pct(w.confidence)})"></span>`).join("");
    return `<tr class="click" data-lot="${esc(r.lot_id)}">
      <td class="lot-id">Lot ${esc(r.lot_id)}</td>
      <td class="mono">${esc(state.tools[r.lot_id] || "n/a")}</td>
      <td>${status}</td>
      <td class="num">${pct(d / r.n_wafers)}<span class="bar-mini"><b style="width:${(d / r.n_wafers) * 100}%"></b></span></td>
      <td>${dom ? cls(dom.cls) + ` <span class="muted small">×${dom.n}</span>` : `<span class="muted">n/a</span>`}</td>
      <td><div class="seq">${seq}</div></td>
      <td><button class="btn sm">Open ${icon("arrow")}</button></td>
    </tr>`;
  }).join("");
  document.querySelectorAll("#o-lots tr.click").forEach((tr) => tr.addEventListener("click", () => openLot(tr.dataset.lot)));

  $("o-alerts").innerHTML = alarms.length ? alarms.map((r) => {
    const stop = r.cost?.recomandare === "opreste";
    return `<div class="alert-item" data-lot="${esc(r.lot_id)}">
      <div class="drift-icon bad">${icon("alert")}</div>
      <div><div class="a-t">Lot ${esc(r.lot_id)} · ${esc(label(r.repeated_class))} drift</div>
        <div class="a-s">${esc(state.tools[r.lot_id] || "n/a")} · first alarm at wafer #${r.first_alarm_index}${r.alarm_analysis ? " (" + esc(r.alarm_analysis.wafer_id) + ")" : ""}</div></div>
      <span class="chip ${stop ? "bad" : "neutral"}">${stop ? "Hold line" : "Monitor"}</span>
    </div>`;
  }).join("") : `<div class="alerts-empty">${icon("ok")}No active drift alarms. All lots are stable.</div>`;
  document.querySelectorAll("#o-alerts .alert-item").forEach((el) => el.addEventListener("click", () => openLot(el.dataset.lot)));

  const counts = {};
  all.forEach((w) => { counts[w.class_name] = (counts[w.class_name] || 0) + 1; });
  const rows = Object.entries(counts).sort((a, b) => (a[0] === "none") - (b[0] === "none") || b[1] - a[1]);
  const max = Math.max(...rows.map((r) => r[1]));
  $("o-dist").innerHTML = rows.map(([c, n]) => `
    <div class="dist-row">${cls(c)}<div class="dist-bar"><b style="width:${(n / max) * 100}%;background:${color(c)}"></b></div><span class="n">${n}</span></div>`).join("");
}

function openLot(lotId) {
  $("l-lot").value = lotId;
  lotInfo();
  showView("lot");
  runLot();
}

// ---------- wafer inspection ----------
async function runAnalyze() {
  const req = { use_llm: $("a-llm").checked, include_gradcam: $("a-cam").checked, cost: state.params, lang: LANG };
  let source;
  if (state.src === "lot") {
    req.wafer_id = $("a-wafer").value;
    if (!req.wafer_id) return toast("Select a wafer first.");
    source = "Lot replay";
  } else if (state.src === "history") {
    req.wafer_id = String($("a-hist").value || "0");
    source = "Historical record";
  } else {
    const up = state.upMap[$("a-up").value];
    if (!up) return toast("Upload a wafer first.");
    req.wafer_id = up.wafer_id;
    req.wafer = up.wafer;
    if ($("a-up-lot").value.trim()) req.lot_id = $("a-up-lot").value.trim();
    source = `Uploaded · ${up.source_file}`;
  }
  loading(true, `Inspecting ${req.wafer_id}`, req.use_llm ? "Classifying, retrieving similar cases and generating the root-cause analysis…" : "Classifying and retrieving similar cases…");
  try {
    const r = await api("/analyze", req);
    state.current = { ...r, fromFile: state.src === "uploaded", source, at: new Date(), aiRequested: req.use_llm };
    renderAnalyze(state.current);
  } catch (e) {
    toast("Inspection failed: " + e.message);
  } finally {
    loading(false);
  }
}
$("a-run").addEventListener("click", runAnalyze);
$("a-print").addEventListener("click", () => window.print());

function gauge(value, c) {
  const r = 40, circ = 2 * Math.PI * r;
  return `<div class="gauge">
    <svg viewBox="0 0 92 92"><circle cx="46" cy="46" r="${r}" stroke="var(--surface-3)" stroke-width="6" fill="none"/>
    <circle cx="46" cy="46" r="${r}" stroke="${c}" stroke-width="6" fill="none" stroke-linecap="round" stroke-dasharray="${circ * value} ${circ}"/></svg>
    <div class="g-val"><div><b>${(value * 100).toFixed(1)}%</b><span>confidence</span></div></div></div>`;
}

function renderAnalyze(r) {
  $("a-empty").classList.add("hidden");
  $("a-results").classList.remove("hidden");
  $("a-print").classList.remove("hidden");

  $("a-meta").innerHTML = [
    ["Wafer", esc(r.wafer_id)], ["Lot", r.lot_id ? esc(r.lot_id) : "n/a"], ["Tool", esc(state.tools[r.lot_id] || "n/a")],
    ["Source", r.source], ["Inspected", r.at.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })],
  ].map(([k, v]) => `<span>${k}<b>${v}</b></span>`).join("");

  // verdict
  const kind = r.is_unknown ? "unknown" : r.is_defect ? "defect" : "clean";
  const sims = r.similar_images || [];
  const agree = sims.filter((s) => s.class_name === r.class_name).length;
  const badges = [
    r.is_unknown ? `<span class="chip warn">${icon("alert")}Out-of-distribution pattern</span>` : "",
    r.uncertain ? `<span class="chip warn">${icon("alert")}Low confidence: manual review</span>` : "",
    !r.is_unknown ? (r.is_defect ? `<span class="chip bad">Defect</span>` : `<span class="chip ok">${icon("ok")}Pass</span>`) : "",
    r.drift?.alarm ? `<span class="chip bad">${icon("alert")}Drift alarm</span>` : "",
  ].join("");
  const v = $("a-verdict");
  v.className = "card verdict " + kind;
  v.innerHTML = `
    <div class="v-top">
      <div>
        <div class="v-eyebrow">Predicted pattern</div>
        <div class="v-class"><i style="background:${color(r.class_name)}"></i>${esc(label(r.class_name))}</div>
        <div class="v-badges" style="margin-top:10px">${badges}</div>
      </div>
      ${gauge(r.confidence || 0, color(r.class_name))}
    </div>
    <div class="v-facts">
      <div>Known pattern<b>${r.is_unknown ? "No (OOD)" : "Yes"}</b></div>
      <div>Historical agreement<b>${sims.length ? `${agree} / ${sims.length}` : "n/a"}</b></div>
      <div>Lot drift<b>${r.drift?.alarm ? "Alarm" : r.drift?.window?.length ? "Stable" : "n/a"}</b></div>
    </div>`;

  // maps
  drawWafer($("a-map"), r.wafer);
  if (r.gradcam_png) {
    $("a-cam-img").src = "data:image/png;base64," + r.gradcam_png;
    $("a-cam-fig").classList.remove("hidden");
  } else $("a-cam-fig").classList.add("hidden");

  // similar
  $("a-similar").innerHTML = sims.map((s, i) => `
    <div class="sim">
      <canvas data-sim="${i}"></canvas>
      <div class="sim-row">${cls(s.class_name)}<span class="sim-p">${s.probability.toFixed(1)}%</span></div>
      <div class="sim-id">${esc(s.wafer_id)}</div>
    </div>`).join("") || `<p class="muted small">No similar wafers (pattern is outside the known classes).</p>`;
  sims.forEach((s, i) => drawWafer(document.querySelector(`[data-sim="${i}"]`), s.wafer));

  renderDrift($("a-drift"), r.drift);

  // root cause
  $("a-expl").textContent = r.explanation || "";
  $("a-action").textContent = r.action || "";
  $("a-action-box").classList.toggle("hidden", !r.action);
  $("a-sources").innerHTML = (r.sources || []).map((s) => `<span class="ref">${icon("doc")}${esc(s)}</span>`).join("");
  $("a-src-label").textContent = r.is_unknown || r.uncertain ? "Rule-based · escalation"
    : r.ai_generated ? "AI agent + knowledge base" : r.aiRequested ? "Rule-based · AI unavailable" : "Rule-based";

  // decision
  state.cost = { p_defect: r.cost.p_defect, wafere_ramase: r.cost.wafere_ramase };
  $("s-ram").value = r.cost.wafere_ramase;
  syncSliders();
  renderDecision(r.cost);

  // review
  $("fb-wrong").classList.add("hidden");
  $("fb-done").classList.add("hidden");
  $("fb-yes").disabled = $("fb-no").disabled = false;
  $("fb-comment").value = "";
  $("fb-class").value = CLASSES.includes(r.class_name) ? r.class_name : "none";
}

function renderDrift(el, d) {
  if (!d) { el.innerHTML = ""; return; }
  const win = d.window || [];
  const cells = win.map((c, i) => {
    const last = i === win.length - 1;
    return `<div class="w ${d.alarm && c === d.repeated_class ? "hit" : ""} ${last ? "cur" : ""}" title="${esc(label(c))}"><small>${last ? "Current" : "t−" + (win.length - 1 - i)}</small>${esc(label(c))}</div>`;
  }).join("");
  el.innerHTML = `
    <div class="drift-head">
      <div class="drift-icon ${d.alarm ? "bad" : "ok"}">${icon(d.alarm ? "alert" : "ok")}</div>
      <div>
        <h3>${d.alarm ? `Drift alarm: ${esc(label(d.repeated_class))}` : "No lot drift"}</h3>
        <p>${d.alarm
          ? `The same pattern appears on ${d.affected} of the last ${win.length} wafers, which indicates a process issue rather than an isolated wafer.`
          : win.length ? `No pattern repeats on 3 or more of the last ${win.length} wafers.` : "Wafer is not part of a tracked lot; no drift window available."}</p>
      </div>
    </div>
    ${win.length ? `<div class="window">${cells}</div>` : ""}`;
}

function costHtml(c) {
  const stop = c.recomandare === "opreste";
  return `
    <div class="cost-compare">
      <div class="cost-box ${stop ? "" : "lower"}">
        <div class="lbl">Expected scrap if line continues${stop ? "" : '<span class="lower-tag">lower</span>'}</div>
        <div class="val">${usd.format(c.cost_rebut)}</div>
        <div class="formula">${c.wafere_ramase} wafers × value × ${pct(c.p_defect)} risk</div>
      </div>
      <div class="cost-box ${stop ? "lower" : ""}">
        <div class="lbl">Cost of stopping to inspect${stop ? '<span class="lower-tag">lower</span>' : ""}</div>
        <div class="val">${usd.format(c.cost_oprire)}</div>
        <div class="formula">downtime × line cost per hour</div>
      </div>
    </div>`;
}

function recoHtml(c) {
  const stop = c.recomandare === "opreste";
  return `<div class="reco ${stop ? "stop" : "go"}">${icon(stop ? "stop" : "ok")}
    <div><div class="r-t">${stop ? "Hold the line and inspect the tool" : "Continue production"}</div>
    <div class="r-s">${stop ? "Expected scrap exceeds the cost of stopping." : "Stopping would cost more than the expected scrap."}</div></div></div>`;
}

function renderDecision(c) {
  $("a-decision").innerHTML = `
    <div class="card-h" style="margin-bottom:0"><h3>Line decision</h3><span class="muted small">cost-based recommendation</span></div>
    ${recoHtml(c)}
    <div class="pbar">Defect risk for remaining wafers<div class="track"><b style="width:${c.p_defect * 100}%"></b></div><strong>${pct(c.p_defect)}</strong></div>
    ${costHtml(c)}`;
}

// ---------- cost sliders ----------
function syncSliders() {
  state.params = { pret_wafer: +$("s-pret").value, ore_oprire: +$("s-ore").value, cost_ora: +$("s-ora").value };
  $("o-pret").textContent = usd.format(state.params.pret_wafer);
  $("o-ore").textContent = state.params.ore_oprire + " h";
  $("o-ora").textContent = usd.format(state.params.cost_ora);
  $("o-ram").textContent = $("s-ram").value;
}

let costTimer;
["s-pret", "s-ore", "s-ora", "s-ram"].forEach((id) => $(id).addEventListener("input", () => {
  syncSliders();
  if (!state.cost) return;
  clearTimeout(costTimer);
  costTimer = setTimeout(async () => {
    try {
      renderDecision(await api("/cost", { p_defect: state.cost.p_defect, wafere_ramase: +$("s-ram").value, params: state.params }));
    } catch (e) { toast("Cost recalculation failed: " + e.message); }
  }, 120);
}));

// ---------- engineer review ----------
async function sendFeedback(correct) {
  const r = state.current;
  if (!r) return;
  const body = { wafer_id: r.wafer_id, predicted_class: r.class_name, correct };
  if (!correct) {
    body.true_class = $("fb-class").value;
    if ($("fb-comment").value.trim()) body.comment = $("fb-comment").value.trim();
  }
  if (r.fromFile) body.wafer = r.wafer;
  try {
    const res = await api("/feedback", body);
    $("fb-wrong").classList.add("hidden");
    $("fb-yes").disabled = $("fb-no").disabled = true;
    $("fb-done").textContent = `Review #${res.id} saved.` + (res.used_for_similar ? " The confirmed label is now part of the similarity search." : "");
    $("fb-done").classList.remove("hidden");
  } catch (e) { toast("Review could not be saved: " + e.message); }
}
$("fb-yes").addEventListener("click", () => sendFeedback(true));
$("fb-no").addEventListener("click", () => $("fb-wrong").classList.toggle("hidden"));
$("fb-send").addEventListener("click", () => sendFeedback(false));

async function loadFeedback() {
  try {
    const rows = await api("/feedback?limit=500");
    const ok = rows.filter((f) => f.correct).length;
    $("f-kpis").innerHTML = `
      <div class="kpi"><div class="lbl">Total reviews</div><div class="val">${rows.length}</div></div>
      <div class="kpi ok"><div class="lbl">Confirmed predictions</div><div class="val">${ok}</div><div class="sub">${rows.length ? pct(ok / rows.length) + " agreement" : "no reviews yet"}</div></div>
      <div class="kpi ${rows.length - ok ? "warn" : ""}"><div class="lbl">Corrections</div><div class="val">${rows.length - ok}</div></div>`;
    $("f-body").innerHTML = rows.length ? rows.map((f) => `
      <tr>
        <td class="mono">#${f.id}</td>
        <td>${new Date(f.ts * 1000).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })}</td>
        <td class="mono">${esc(f.wafer_id)}</td>
        <td>${cls(f.predicted_class)}</td>
        <td>${f.correct ? `<span class="chip ok">${icon("ok")}Confirmed</span>` : `<span class="chip warn">Corrected</span>`}</td>
        <td>${f.true_class ? cls(f.true_class) : `<span class="muted">n/a</span>`}</td>
        <td>${esc(f.comment || "")}</td>
      </tr>`).join("") : `<tr><td colspan="7" class="empty-row">No reviews yet. Confirm or correct predictions from Wafer Inspection.</td></tr>`;
  } catch (e) { toast("Could not load the review log: " + e.message); }
}
$("f-refresh").addEventListener("click", loadFeedback);

// ---------- lot monitor ----------
async function runLot() {
  const lot_id = $("l-lot").value;
  if (!lot_id) return toast("No lots available.");
  const llm = $("l-llm").checked;
  loading(true, `Replaying lot ${lot_id}`, llm ? "Classifying 25 wafers and analysing the first alarm…" : "Classifying 25 wafers…");
  try {
    renderLot(await api("/lot", { lot_id, use_llm: llm, include_maps: true, cost: state.params, lang: LANG }));
  } catch (e) {
    toast("Lot replay failed: " + e.message);
  } finally {
    loading(false);
  }
}
$("l-run").addEventListener("click", runLot);

function lotChart(r) {
  const W = 1000, H = 220, L = 38, R = 8, T = 22, B = 26;
  const n = r.wafers.length, bw = (W - L - R) / n;
  const y = (v) => T + (1 - v) * (H - T - B);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Confidence per wafer">`;
  [0, 0.5, 1].forEach((v) => { s += `<line class="grid-l" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="ax" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`; });
  r.wafers.forEach((w, i) => {
    const x = L + i * bw;
    if (w.alarm) s += `<rect class="alarm-zone" x="${x}" y="${T}" width="${bw}" height="${H - T - B}"/>`;
  });
  r.wafers.forEach((w, i) => {
    const x = L + i * bw + bw * 0.18, h = y(0) - y(w.confidence);
    s += `<rect class="bar" data-wafer="${esc(w.wafer_id)}" x="${x}" y="${y(w.confidence)}" width="${bw * 0.64}" height="${Math.max(h, 1)}" rx="2" fill="${color(w.class_name)}"><title>#${w.index} ${esc(w.wafer_id)} · ${esc(label(w.class_name))} · ${pct(w.confidence, 1)}</title></rect>`;
    s += `<text class="ax" x="${L + i * bw + bw / 2}" y="${H - 8}" text-anchor="middle">${w.index}</text>`;
  });
  if (r.first_alarm_index) {
    const x = L + (r.first_alarm_index - 1) * bw;
    s += `<line class="alarm-line" x1="${x}" x2="${x}" y1="${T - 14}" y2="${H - B}"/><text class="alarm-lbl" x="${x + 6}" y="${T - 6}">Drift alarm · ${esc(label(r.repeated_class))}</text>`;
  }
  return s + "</svg>";
}

function renderLot(r) {
  $("l-empty").classList.add("hidden");
  $("l-results").classList.remove("hidden");
  const defects = r.wafers.filter((w) => w.class_name !== "none").length;
  const stop = r.cost?.recomandare === "opreste";
  $("l-kpis").innerHTML = `
    <div class="kpi"><div class="lbl">Wafers tested</div><div class="val">${r.n_wafers}</div><div class="sub">Lot ${esc(r.lot_id)} · ${esc(state.tools[r.lot_id] || "n/a")}</div></div>
    <div class="kpi ${defects ? "warn" : "ok"}"><div class="lbl">Defective wafers</div><div class="val">${defects}</div><div class="sub">${pct(defects / r.n_wafers)} of the lot</div></div>
    <div class="kpi ${r.alarm ? "bad" : "ok"}"><div class="lbl">Drift alarm</div><div class="val">${r.alarm ? `Wafer #${r.first_alarm_index}` : "None"}</div><div class="sub">${r.alarm ? esc(label(r.repeated_class)) + " repeating" : "No repeating pattern"}</div></div>
    <div class="kpi ${r.alarm && stop ? "bad" : "ok"}"><div class="lbl">Line decision</div><div class="val">${r.alarm && stop ? "Hold" : "Continue"}</div><div class="sub">${r.cost ? `scrap ${usd.format(r.cost.cost_rebut)} vs stop ${usd.format(r.cost.cost_oprire)}` : "no alarm raised"}</div></div>`;

  const present = [...new Set(r.wafers.map((w) => w.class_name))];
  $("l-legend").innerHTML = present.map((c) => `<span>${cls(c)}</span>`).join("") + (r.alarm ? `<span><i class="lg-alarm"></i>Alarm window</span>` : "");
  $("l-chart").innerHTML = lotChart(r);
  $("l-chart").querySelectorAll(".bar").forEach((b) => b.addEventListener("click", () => openWafer(r.lot_id, b.dataset.wafer)));

  $("l-grid").innerHTML = r.wafers.map((w, i) => `
    <div class="tile ${w.alarm ? "alarm" : ""} ${w.index === r.first_alarm_index ? "first" : ""}" data-wafer="${esc(w.wafer_id)}" title="Inspect ${esc(w.wafer_id)}">
      <span class="t-idx">#${w.index}</span>
      <canvas data-tile="${i}"></canvas>
      <div class="t-cls">${cls(w.class_name)}</div>
      <div class="t-conf"><span class="mono">${esc(w.wafer_id)}</span><span>${pct(w.confidence)}</span></div>
    </div>`).join("");
  r.wafers.forEach((w, i) => drawWafer(document.querySelector(`[data-tile="${i}"]`), w.wafer));
  document.querySelectorAll(".tile").forEach((t) => t.addEventListener("click", () => openWafer(r.lot_id, t.dataset.wafer)));

  const box = $("l-alarm");
  const a = r.alarm_analysis;
  if (r.alarm && a) {
    box.classList.remove("hidden");
    box.innerHTML = `
      <div class="card-h"><h3>Root-cause analysis · first alarm on ${esc(a.wafer_id)} (#${r.first_alarm_index})
        <span class="tag" style="margin-left:8px">${a.ai_generated ? "AI agent" : "Rule-based"}</span></h3>
        <button class="btn" id="l-open">Open full inspection ${icon("arrow")}</button></div>
      <div class="alarm-grid">
        <div>${a.gradcam_png ? `<img src="data:image/png;base64,${a.gradcam_png}" alt="Grad-CAM heatmap">` : `<canvas id="l-alarm-map"></canvas>`}</div>
        <div>
          <p class="explanation">${esc(a.explanation)}</p>
          <div class="action"><div class="action-h">Recommended action</div>${esc(a.action)}</div>
          <div class="refs">${(a.sources || []).map((s) => `<span class="ref">${icon("doc")}${esc(s)}</span>`).join("")}</div>
        </div>
        ${r.cost ? `<div class="alarm-side">${recoHtml(r.cost)}${costHtml(r.cost)}</div>` : ""}
      </div>`;
    if (!a.gradcam_png) drawWafer($("l-alarm-map"), a.wafer);
    $("l-open").addEventListener("click", () => openWafer(r.lot_id, a.wafer_id));
  } else {
    box.classList.add("hidden");
  }
}

function openWafer(lotId, waferId) {
  if (state.upMap[waferId]) {          // wafer încărcat de utilizator: se trimite harta, cu lotul în care a rulat
    setSource("uploaded");
    $("a-up").value = waferId;
    $("a-up-lot").value = lotId || "";
  } else {
    setSource("lot");
    $("a-lot").value = lotId;
    fillWafers();
    $("a-wafer").value = waferId;
  }
  showView("analyze");
  runAnalyze();
}

// ---------- upload ----------
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

function defaultLotId() {
  const d = new Date(), p = (n) => String(n).padStart(2, "0");
  return `UPLOAD-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

async function uploadFiles(files) {
  files = [...files];
  if (!files.length) return;
  const tooBig = files.filter((f) => f.size > MAX_UPLOAD_BYTES);
  files = files.filter((f) => f.size <= MAX_UPLOAD_BYTES);
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  fd.append("existing", state.uploads.map((w) => w.wafer_id).join(","));
  let res = { wafers: [], errors: [] };
  if (files.length) {
    loading(true, `Reading ${files.length} file${files.length > 1 ? "s" : ""}`, "Parsing and validating wafer maps…");
    try {
      const r = await fetch(API + "/upload", { method: "POST", body: fd });
      if (!r.ok) {
        let msg = r.statusText;
        try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not JSON */ }
        throw new Error(`${r.status} ${msg}`);
      }
      res = await r.json();
    } catch (e) {
      toast("Upload failed: " + e.message);
      return;
    } finally {
      loading(false);
    }
  }
  res.errors.push(...tooBig.map((f) => ({ file: f.name, error: "file is larger than 20 MB" })));
  res.wafers.forEach((w) => { state.uploads.push(w); state.upMap[w.wafer_id] = w; });
  renderUploadErrors(res.errors);
  renderUploads();
  if (res.wafers.length) toast(`${res.wafers.length} wafer${res.wafers.length > 1 ? "s" : ""} added.`, "info");
}

function renderUploadErrors(errors) {
  const box = $("u-errors");
  box.classList.toggle("hidden", !errors.length);
  box.innerHTML = errors.length ? `<h3>${icon("alert")}${errors.length} file${errors.length > 1 ? "s" : ""} could not be read</h3>
    <ul>${errors.map((e) => `<li><b>${esc(e.file)}</b>: ${esc(e.error)}</li>`).join("")}</ul>` : "";
}

function renderUploads() {
  const ups = state.uploads;
  $("u-work").classList.toggle("hidden", !ups.length);
  $("nav-up-count").textContent = ups.length;
  $("nav-up-count").classList.toggle("hidden", !ups.length);
  $("u-count").textContent = ups.length;
  const files = new Set(ups.map((w) => w.source_file)).size;
  const warned = ups.filter((w) => w.warnings.length).length;
  $("u-summary").textContent = `from ${files} file${files !== 1 ? "s" : ""}` + (warned ? ` · ${warned} with warnings` : "");
  $("u-run").disabled = !ups.length;
  if (!$("u-lot").value) $("u-lot").value = defaultLotId();

  $("u-grid").innerHTML = ups.map((w, i) => `
    <div class="up-card ${w.warnings.length ? "warn" : ""}">
      <span class="u-idx">#${i + 1}</span>
      <button class="u-x" data-del="${esc(w.wafer_id)}" title="Remove">${icon("x")}</button>
      <canvas data-up="${i}"></canvas>
      <div class="u-id" title="${esc(w.wafer_id)}">${esc(w.wafer_id)}</div>
      <div class="u-meta"><span>${esc(w.format)} · ${w.shape[0]}×${w.shape[1]}</span><span>${w.stats.fail_pct}% fail</span></div>
      ${w.warnings.map((m) => `<div class="u-warn">${esc(m)}</div>`).join("")}
      <div class="u-actions"><button class="btn sm" data-inspect="${esc(w.wafer_id)}">Inspect ${icon("arrow")}</button></div>
    </div>`).join("");
  ups.forEach((w, i) => drawWafer(document.querySelector(`[data-up="${i}"]`), w.wafer));
  document.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", () => removeUpload(b.dataset.del)));
  document.querySelectorAll("[data-inspect]").forEach((b) => b.addEventListener("click", () => openWafer(null, b.dataset.inspect)));
  refreshUploadSelect();
}

function removeUpload(id) {
  state.uploads = state.uploads.filter((w) => w.wafer_id !== id);
  delete state.upMap[id];
  renderUploads();
}

function refreshUploadSelect() {
  const sel = $("a-up"), keep = sel.value, has = state.uploads.length > 0;
  sel.innerHTML = state.uploads.map((w) => `<option value="${esc(w.wafer_id)}">${esc(w.wafer_id)}</option>`).join("");
  if (state.upMap[keep]) sel.value = keep;
  $("a-up-field").classList.toggle("hidden", !has);
  $("a-up-empty").classList.toggle("hidden", has);
}

async function runUploadedLot() {
  const lot_id = $("u-lot").value.trim() || defaultLotId();
  $("u-lot").value = lot_id;
  if (state.lots.some((l) => l.lot_id === lot_id)) return toast(`"${lot_id}" is a built-in lot. Choose another lot ID.`);
  const llm = $("u-llm").checked;
  loading(true, `Replaying ${lot_id}`, `Classifying ${state.uploads.length} uploaded wafers in order${llm ? " and analysing the first alarm" : ""}…`);
  try {
    const r = await api("/lot", {
      lot_id, use_llm: llm, include_maps: true, cost: state.params, lang: LANG,
      wafers: state.uploads.map((w) => ({ wafer_id: w.wafer_id, wafer: w.wafer })),
    });
    state.tools[lot_id] = "Uploaded";
    showView("lot");
    renderLot(r);
  } catch (e) {
    toast("Lot replay failed: " + e.message);
  } finally {
    loading(false);
  }
}

$("u-input").addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });
const drop = $("u-drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
// fișiere trase oriunde pe pagina de upload
$("view-upload").addEventListener("dragover", (e) => e.preventDefault());
$("view-upload").addEventListener("drop", (e) => { e.preventDefault(); if (!drop.contains(e.target)) uploadFiles(e.dataTransfer.files); });
$("u-run").addEventListener("click", runUploadedLot);
$("u-clear").addEventListener("click", () => {
  state.uploads = []; state.upMap = {};
  $("u-lot").value = "";
  renderUploadErrors([]);
  renderUploads();
});
$("u-sample").addEventListener("click", async () => {
  try {
    const r = await fetch(API + "/upload/sample?fmt=json");
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    await uploadFiles([new File([await r.blob()], "wafersight_sample.json", { type: "application/json" })]);
  } catch (e) { toast("Could not load the sample: " + e.message); }
});
$("a-up-go").addEventListener("click", (e) => { e.preventDefault(); showView("upload"); });
$("u-dl-json").href = API + "/upload/sample?fmt=json";
$("u-dl-csv").href = API + "/upload/sample?fmt=csv";

// redraw canvases when the OS theme changes
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (state.current) renderAnalyze(state.current);
});

init();
