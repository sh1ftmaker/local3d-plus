import { Stage, nullStage, fetchBuffer, fetchText, WIRE_FACE_LIMIT } from './stage.js';

// ------------------------------------------------------------------ helpers
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtBytes = (n) => n == null ? '–' : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : n < 1073741824 ? `${(n / 1048576).toFixed(n < 10485760 ? 1 : 0)} MB` : `${(n / 1073741824).toFixed(2)} GB`;
const fmtInt = (n) => n == null ? '–' : Math.round(n).toLocaleString('en-US');
const fmtShort = (n) => n == null ? '' : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);
const fmtPct = (x, d = 1) => x == null ? '–' : `${(x * 100).toFixed(d)}%`;
const fmtDur = (s) => { if (s == null) return '–'; s = Math.round(s); const m = Math.floor(s / 60); return m ? `${m}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`; };
const fmtMs = (ms) => ms == null ? '–' : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
const nowS = () => Date.now() / 1000;
const fmtAgo = (t) => {
  const d = Math.max(0, nowS() - t);
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
};
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const phoneMQ = window.matchMedia('(max-width: 768px), (max-height: 520px) and (pointer: coarse)');
const isPhone = () => phoneMQ.matches;
const PANE_COLORS = ['#6cb6ff', '#ffb84d', '#c38bff', '#4cc38a'];
const MAX_PANES = 4;
const RAW_CONFIRM_BYTES = 40 * 1048576;

// download concurrency (phones + proxy: 2 at a time)
const slots = { n: 2, q: [] };
const acquire = () => new Promise((r) => { if (slots.n > 0) { slots.n--; r(); } else slots.q.push(r); });
const release = () => { const r = slots.q.shift(); if (r) r(); else slots.n++; };

// ------------------------------------------------------------------ state
const S = {
  data: null, version: '', runs: new Map(), groups: new Map(), runGroup: new Map(),
  sel: [], focusStem: null, panes: new Map(),
  refIdx: 0, refSrc: store.get('tv.refSrc', 'crop'),
  ov: Object.assign({ mode: 'fade', value: 0.5, outline: false }, store.get('tv.ov', {})),
  cols: store.get('tv.cols', 'auto'),
  look: Object.assign({ mode: 'textured', light: 'studio', normals: true, wire: false, bg: 'dark', autoRotate: false, photoInPanes: false }, store.get('tv.look', {})),
  active: null, abPrev: null,
  seen: store.get('tv.seen', null), seenInputs: store.get('tv.seenInputs', null),
  filter: '', fovOverride: {},
};
const firstVisit = S.seen == null;
S.seen ||= {};
S.seenInputs ||= {};

let stage;
try { stage = new Stage($('#gl')); } catch (e) {
  console.error(e);
  stage = nullStage(e);
  setTimeout(() => toast('<span class="warn">WebGL unavailable</span> – images and stats still work, 3D views are disabled.', null, 15000), 500);
}
stage.setLook(S.look);
stage.photoInPanes = S.look.photoInPanes;

// ------------------------------------------------------------------ toasts / lightbox
function toast(html, action, ms = 7000) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = `<span>${html}</span>`;
  if (action) {
    const b = document.createElement('button');
    b.className = 'ghost-btn'; b.textContent = action.label;
    b.onclick = () => { action.fn(); el.remove(); };
    el.append(b);
  }
  el.addEventListener('click', (e) => { if (e.target === el) el.remove(); });
  $('#toasts').append(el);
  while ($('#toasts').children.length > 4) $('#toasts').firstChild.remove();
  setTimeout(() => el.remove(), ms);
}
function lightbox(url, cap) {
  const lb = $('#lightbox');
  $('img', lb).src = url;
  $('.lb-cap', lb).textContent = cap || '';
  lb.hidden = false;
}
$('#lightbox').onclick = () => { $('#lightbox').hidden = true; $('#lightbox img').src = ''; };

// ------------------------------------------------------------------ data
function isNew(run) { return !S.seen[run.id]; }
function markSeen(ids) {
  for (const id of ids) S.seen[id] = 1;
  store.set('tv.seen', S.seen);
}
function bestFinal(run) {
  const st = run.finals.filter((f) => f.stable);
  return st[st.length - 1] || null;
}
const SAMPLES = 'samples/';
const sp = (p) => (p ? SAMPLES + String(p).replace(/^\.?\//, '') : null); // index paths are relative to samples/
const isoS = (t) => { const v = Date.parse(t); return Number.isFinite(v) ? v / 1000 : Date.now() / 1000; };
function groupThumb(g) {
  if (g.thumbUrl) return g.thumbUrl;
  if (g.image) return g.image.url;
  const r = g.runs.find((x) => x.crop);
  return r ? r.crop.url : '';
}
function groupImageUrl(g) { return g?.image?.url || null; }

// what each preset changes (shown per run)
const PRESET_NOTES = {
  baseline: 'Stock ComfyUI Pixal3D template settings, no Local3D+ mods.',
  plus: 'Local3D+ mods: mesh repair, inverted-shell removal, QEF remesh, isotropic remesh and photo-to-texture projection; more steps and Heun sampling.',
  quality: 'Local3D+ quality preset.',
  lod: 'Low-poly LOD: simplified to a triangle budget, re-unwrapped, with base color and normal map re-baked from the full-resolution model.',
};
function runNote(run) {
  const m = run.meta || {};
  const parts = [];
  const base = PRESET_NOTES[m.preset] || (m.preset ? `Preset “${m.preset}”.` : '');
  if (base) parts.push(base);
  if (m.model) parts.push(`Model: ${m.model === 'pixal3d' ? 'Pixal3D (pixel-aligned)' : m.model}.`);
  if (/quad/.test(run.tag) || run.objs.length) parts.push('Quad retopology (Instant Meshes), OBJ with true quads.');
  const ov = m.overrides && Object.keys(m.overrides).length ? Object.entries(m.overrides).map(([k, v]) => `${k}=${v}`).join(', ') : '';
  if (ov) parts.push(`Overrides: ${ov}.`);
  return parts.join(' ');
}

// samples/index.json (exporter contract) -> the internal groups/runs model the UI uses
function adapt(idx) {
  const groups = [];
  for (const inp of idx.inputs || []) {
    const g = {
      stem: inp.id, title: inp.title || inp.id,
      image: inp.image ? { name: inp.image.split('/').pop(), url: sp(inp.image) } : null,
      thumbUrl: sp(inp.thumb) || sp(inp.image) || '',
      runs: [], mtime: 0,
    };
    for (const r of inp.runs || []) {
      const created = isoS(r.created);
      const st = r.stats || {};
      const glb = r.glb ? { name: r.glb.split('/').pop(), url: sp(r.glb), size: r.size_bytes ?? null, mtime: created, stable: true, stats: { vertices: st.verts, faces: st.faces } } : null;
      const obj = r.obj ? { name: r.obj.split('/').pop(), url: sp(r.obj), size: null, mtime: created, stable: true, stats: null } : null;
      g.runs.push({
        id: r.id, tag: r.label || r.id, finals: glb ? [glb] : [], raws: [], objs: obj ? [obj] : [], mtls: [], textures: [], other: [],
        crop: r.crop ? { name: 'crop', url: sp(r.crop) } : null, mask: r.mask ? { name: 'mask', url: sp(r.mask) } : null,
        meta: r.meta || null, metaFile: null, pre: st, renders: (r.renders || []).map(sp),
        mtime: created, created, status: glb ? 'done' : 'incomplete',
      });
    }
    g.runs.sort((x, y) => x.created - y.created);
    g.latest = Math.max(0, ...g.runs.map((r) => r.mtime));
    groups.push(g);
  }
  groups.sort((x, y) => y.latest - x.latest);
  const reports = (idx.reports || []).map((r) => ({ name: r.title || r.image, url: sp(r.image) }));
  return { groups, reports, generated: idx.generated };
}

let lastText = '';
async function loadIndex() {
  try {
    const res = await fetch(`${SAMPLES}index.json?ts=${Date.now()}`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    setLive('on');
    if (text === lastText && S.data) return;
    lastText = text;
    const data = adapt(JSON.parse(text));
    data.version = String(text.length) + ':' + (data.generated || '') + ':' + [...text].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 0);
    apply(data);
  } catch (e) {
    console.error(e);
    setLive('off');
    if (!S.data) $('#gallery').innerHTML = `<div class="empty">Could not load samples/index.json<br><span class="tiny">${esc(e.message || e)}</span></div>`;
  }
}

function apply(data) {
  const prevRuns = S.runs, prevGroups = S.groups, hadData = !!S.data;
  if (data.version === S.version && hadData) return;
  S.data = data; S.version = data.version;
  S.runs = new Map(); S.groups = new Map(); S.runGroup = new Map();
  for (const g of data.groups) {
    S.groups.set(g.stem, g);
    for (const r of g.runs) { S.runs.set(r.id, r); S.runGroup.set(r.id, g); }
  }
  if (firstVisit && !hadData) {
    // first ever visit: only runs from the last 24 h count as NEW
    for (const r of S.runs.values()) if (r.created < Date.now() / 1000 - 86400) S.seen[r.id] = 1;
    for (const g of S.groups.values()) S.seenInputs[g.stem] = 1;
    store.set('tv.seen', S.seen); store.set('tv.seenInputs', S.seenInputs);
  }
  if (hadData) {
    for (const g of S.groups.values()) {
      if (!prevGroups.has(g.stem)) toast(`New input <b>${esc(g.title || g.stem)}</b>`, { label: 'View', fn: () => openGroup(g.stem) });
    }
    for (const r of S.runs.values()) {
      if (prevRuns.has(r.id)) continue;
      const g = S.runGroup.get(r.id);
      toast(`New run: <b>${esc(g.title || g.stem)}</b> · ${esc(r.tag)}`, { label: 'Open', fn: () => addRun(r.id) });
    }
  }
  renderSidebar();
  renderGallery();
  renderReports();
  for (const [id, p] of S.panes) { const r = S.runs.get(id); if (r) p.update(r); else p.gone(); }
  updateRef();
  renderStats();
  if (!hadData) route();
  $('#updated').textContent = data.generated ? `updated ${new Date(data.generated).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}` : '';
}

// ------------------------------------------------------------------ "live" updates: poll the static index (Pages redeploys on push)
function setLive(state) {
  const el = $('#live');
  el.className = `live ${state}`;
  $('span', el).textContent = state === 'on' ? 'synced' : state === 'off' ? 'offline' : 'loading';
}
setInterval(() => { if (document.visibilityState === 'visible') loadIndex(); }, 60000);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') { loadIndex(); stage.requestRender(); } });

// ------------------------------------------------------------------ sidebar / gallery
function runBadges(r) {
  let b = '';
  if (isNew(r)) b += '<span class="badge new">new</span>';
  if (r.status === 'running') b += '<span class="badge running">running</span>';
  else if (r.status === 'incomplete' || (r.status === 'done' && !r.finals.length)) b += '<span class="badge incomplete">no model</span>';
  if (r.objs.length) b += '<span class="badge quad">quad</span>';
  return b;
}
function matchesFilter(g) {
  const f = S.filter.trim().toLowerCase();
  if (!f) return true;
  return g.stem.toLowerCase().includes(f) || g.runs.some((r) => r.tag.toLowerCase().includes(f));
}
function renderSidebar() {
  const box = $('#groups');
  const groups = [...S.groups.values()].filter(matchesFilter);
  if (!groups.length) { box.innerHTML = `<div class="empty">${S.groups.size ? 'No match' : 'No inputs yet'}</div>`; return; }
  const scroll = box.scrollTop;
  box.innerHTML = groups.map((g) => {
    const nNew = g.runs.filter(isNew).length;
    const newInput = !S.seenInputs[g.stem];
    const runs = [...g.runs].reverse().map((r) => {
      const f = bestFinal(r);
      const faces = f?.stats?.faces;
      const right = faces ? `${fmtShort(faces)} f` : r.status === 'running' ? '' : f ? fmtBytes(f.size) : '';
      const meta = r.meta || {};
      const sub = [meta.seed != null ? `seed ${meta.seed}` : null, meta.seconds ? fmtDur(meta.seconds) : null, fmtAgo(r.mtime)].filter(Boolean).join(' · ');
      return `<div class="run ${S.sel.includes(r.id) ? 'sel' : ''}" data-run="${esc(r.id)}">
        <span class="chk"></span>
        <span class="run-tag" title="${esc(r.id)}">${esc(r.tag)}${runBadges(r)}</span>
        <span class="run-meta">${right}</span>
        <span class="run-sub">${esc(sub)}</span>
      </div>`;
    }).join('');
    const th = groupThumb(g);
    return `<div class="grp" data-stem="${esc(g.stem)}">
      <div class="grp-head" data-group="${esc(g.stem)}" title="Compare all runs of this input">
        ${th ? `<img class="thumb" loading="lazy" src="${th}" alt="">` : '<div class="thumb"></div>'}
        <div class="grp-info">
          <div class="grp-name">${esc(g.title || g.stem)}${nNew ? `<span class="badge new">${nNew} new</span>` : newInput ? '<span class="badge new">new</span>' : ''}</div>
          <div class="grp-sub">${g.runs.length} run${g.runs.length === 1 ? '' : 's'} · ${fmtAgo(g.latest)}</div>
        </div>
      </div>
      ${runs ? `<div class="runs">${runs}</div>` : '<div class="grp-sub" style="margin:6px 4px 0">No runs yet</div>'}
    </div>`;
  }).join('');
  box.scrollTop = scroll;
  const strip = $('#thumbStrip');
  if (strip) {
    const cur = S.focusStem;
    strip.innerHTML = [...S.groups.values()].map((g) => {
      const th = groupThumb(g);
      const n = g.runs.filter(isNew).length || !S.seenInputs[g.stem];
      return `<button data-jump="${esc(g.stem)}" class="${g.stem === cur ? 'cur' : ''}">${th ? `<img loading="lazy" src="${th}" alt="">` : '<img alt="">'}${n ? '<i></i>' : ''}<span>${esc(g.title || g.stem)}</span></button>`;
    }).join('');
  }
  const anyNew = [...S.runs.values()].some(isNew);
  $('#menuBadge').hidden = !anyNew;
}
$('#groups').addEventListener('click', (e) => {
  const run = e.target.closest('[data-run]');
  if (run) { toggleRun(run.dataset.run); return; }
  const g = e.target.closest('[data-group]');
  if (g) { openGroup(g.dataset.group); closeDrawer(); }
});
$('#thumbStrip').addEventListener('click', (e) => {
  const b = e.target.closest('[data-jump]');
  if (!b) return;
  const el = $(`.grp[data-stem="${CSS.escape(b.dataset.jump)}"]`);
  if (el) el.scrollIntoView({ block: 'start', behavior: 'smooth' });
});
$('#drawerClose').onclick = () => closeDrawer();
$('#search').addEventListener('input', (e) => { S.filter = e.target.value; renderSidebar(); renderGallery(); });
$('#markSeen').onclick = () => {
  markSeen([...S.runs.keys()]);
  for (const g of S.groups.keys()) S.seenInputs[g] = 1;
  store.set('tv.seenInputs', S.seenInputs);
  renderSidebar(); renderGallery();
};

function renderGallery() {
  const groups = [...S.groups.values()].filter(matchesFilter);
  const nRuns = S.runs.size;
  $('#homeSummary').textContent = `${S.groups.size} input${S.groups.size === 1 ? '' : 's'} · ${nRuns} run${nRuns === 1 ? '' : 's'} · tap an input to compare its runs`;
  $('#gallery').innerHTML = groups.length ? groups.map((g) => {
    const nNew = g.runs.filter(isNew).length;
    const running = g.runs.some((r) => r.status === 'running');
    const img = groupThumb(g);
    const chips = g.runs.slice(-8).map((r) => `<span class="chip">${esc(r.tag)}</span>`).join('');
    return `<div class="gcard" data-group="${esc(g.stem)}">
      <div class="gimg" style="background-image:url('${img}')">${nNew ? `<span class="badge new">${nNew} new</span>` : !S.seenInputs[g.stem] ? '<span class="badge new">new</span>' : ''}${running ? '<span class="badge running" style="position:absolute;top:8px;right:8px">running</span>' : ''}</div>
      <div class="gbody"><div class="gname">${esc(g.title || g.stem)}</div>
        <div class="muted tiny">${g.runs.length} run${g.runs.length === 1 ? '' : 's'} · ${fmtAgo(g.latest)}</div>
        <div class="chips">${chips}</div></div>
    </div>`;
  }).join('') : `<div class="empty">${S.data ? 'No inputs found in test_inputs/ or outputs/' : 'Loading…'}</div>`;
}
$('#gallery').addEventListener('click', (e) => { const g = e.target.closest('[data-group]'); if (g) openGroup(g.dataset.group); });

function renderReports() {
  const reps = S.data?.reports || [];
  $('#reportsWrap').hidden = !reps.length;
  $('#reports').innerHTML = reps.map((f) => `<img loading="lazy" src="${esc(f.url)}" data-full="${esc(f.url)}" title="${esc(f.name)}" alt="${esc(f.name)}">`).join('');
}
$('#reports').addEventListener('click', (e) => { const i = e.target.closest('img'); if (i) lightbox(i.dataset.full, i.title); });

// drawer (phones)
const openDrawer = () => document.body.classList.add('drawer');
const closeDrawer = () => document.body.classList.remove('drawer');
$('#menuBtn').onclick = () => document.body.classList.toggle('drawer');
$('#scrim').onclick = closeDrawer;
$('#drawerDone').onclick = closeDrawer;

// ------------------------------------------------------------------ routing / selection
function writeHash() {
  const h = S.sel.length ? `#/c/${S.sel.map(encodeURIComponent).join(',')}` : S.focusStem ? `#/i/${encodeURIComponent(S.focusStem)}` : '#/';
  if (location.hash !== h) history.replaceState(null, '', h);
  syncPanes();
}
function route() {
  const h = location.hash || '';
  let m;
  if ((m = h.match(/^#\/c\/(.+)$/))) {
    S.sel = m[1].split(',').map(decodeURIComponent).filter((id) => S.runs.has(id)).slice(0, MAX_PANES);
    S.focusStem = S.sel.length ? S.runGroup.get(S.sel[0]).stem : null;
  } else if ((m = h.match(/^#\/i\/(.+)$/))) {
    S.focusStem = decodeURIComponent(m[1]); S.sel = [];
  } else { S.sel = []; S.focusStem = null; }
  syncPanes();
}
window.addEventListener('hashchange', route);
$('#homeLink').onclick = (e) => { e.preventDefault(); S.sel = []; S.focusStem = null; history.pushState(null, '', '#/'); syncPanes(); };

function openGroup(stem) {
  const g = S.groups.get(stem);
  if (!g) return;
  S.seenInputs[stem] = 1; store.set('tv.seenInputs', S.seenInputs);
  const withModel = g.runs.filter((r) => r.finals.length || r.status === 'running');
  const pick = (withModel.length ? withModel : g.runs).slice(-MAX_PANES);
  S.sel = pick.map((r) => r.id);
  S.focusStem = stem;
  S.active = S.sel[S.sel.length - 1] || null;
  S.abPrev = S.sel.length > 1 ? S.sel[0] : null;
  S.refIdx = Math.max(0, S.sel.length - 1);
  history.pushState(null, '', S.sel.length ? `#/c/${S.sel.map(encodeURIComponent).join(',')}` : `#/i/${encodeURIComponent(stem)}`);
  syncPanes();
  window.scrollTo({ top: 0 });
}
function addRun(id) {
  if (!S.sel.includes(id)) {
    if (S.sel.length >= MAX_PANES) { const drop = S.sel.shift(); toast(`Max ${MAX_PANES} panes: removed ${esc(S.runs.get(drop)?.tag || drop)}`, null, 3500); }
    S.sel.push(id);
  }
  if (S.active !== id) { S.abPrev = S.active; S.active = id; }
  S.focusStem = S.runGroup.get(id)?.stem || S.focusStem;
  writeHash();
}
function toggleRun(id) {
  if (S.sel.includes(id)) {
    S.sel = S.sel.filter((x) => x !== id);
    if (!S.sel.length && S.focusStem) { /* keep focus on input */ }
    writeHash();
  } else addRun(id);
}

// ------------------------------------------------------------------ panes
class Pane {
  constructor(run) {
    this.run = run;
    this.el = document.createElement('div');
    this.el.className = 'pane card';
    this.el.innerHTML = `
      <div class="card-head">
        <span class="pane-dot"></span>
        <div class="pane-title"><span class="stem"></span><span class="tag"></span></div>
        <button class="pane-btn close" title="Remove from compare">✕</button>
      </div>
      <div class="slot">
        <div class="loader" hidden><div class="box"><div class="msg"></div><div class="bar"><i></i></div><div class="actions"></div></div></div>
        <div class="update-banner" hidden><span></span><button class="ghost-btn">Load</button></div>
        <span class="corner-tag" hidden></span>
        <div class="photo-full"><img alt="input photo" draggable="false"></div>
        <button class="pip" aria-label="Show input photo"><img alt="" draggable="false"></button>
        <span class="view-hint">double-tap: frame</span>
      </div>
      <div class="pane-tools">
        <select class="pane-src" aria-label="Model file"></select>
        <span class="spacer"></span>
        <button class="pane-btn topo">Topology</button>
        <button class="pane-btn ref desktop-only">Use in overlay</button>
        <button class="pane-btn turn" hidden>Turntable</button>
        <button class="pane-btn rm phone-only">Remove</button>
        <div class="run-note"></div>
      </div>`;
    this.slot = $('.slot', this.el);
    this.view = stage.addPane(this.slot);
    this.loaded = null; this.loadingKey = null; this.abort = null; this.topo = null; this.error = null;
    $('.close', this.el).onclick = () => toggleRun(this.run.id);
    $('.rm', this.el).onclick = () => toggleRun(this.run.id);
    $('.turn', this.el).onclick = () => { const u = this.run.renders?.[0]; if (u) lightbox(u, `${this.run.tag} · turntable`); };
    // keep taps on in-viewer UI away from OrbitControls (pointer capture would swallow the click)
    for (const el of $$('.pip, .photo-full, .loader, .update-banner', this.el)) el.addEventListener('pointerdown', (e) => e.stopPropagation());
    $('.pip', this.el).onclick = (e) => { e.stopPropagation(); this.el.classList.toggle('show-photo'); stage.requestRender(); };
    $('.photo-full', this.el).onclick = () => { this.el.classList.remove('show-photo'); stage.requestRender(); };
    // no page scroll / rubber-band while a finger is on the viewer (older iOS ignores touch-action)
    this.slot.addEventListener('touchmove', (e) => { if (!e.target.closest('.loader, .update-banner')) e.preventDefault(); }, { passive: false });
    // double-tap (or double-click) to frame
    let lastTap = 0, downAt = null;
    this.slot.addEventListener('pointerdown', (e) => { downAt = { x: e.clientX, y: e.clientY, t: performance.now() }; });
    this.slot.addEventListener('pointerup', (e) => {
      if (!downAt || e.target.closest('button, .loader, .photo-full')) return;
      const moved = Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y), dt = performance.now() - downAt.t;
      if (moved > 14 || dt > 450) { lastTap = 0; return; }
      const now = performance.now();
      if (now - lastTap < 420) { lastTap = 0; stage.frameAll(); syncToolbar(); } else lastTap = now;
    });
    $('.pane-src', this.el).onchange = (e) => this.load(e.target.value);
    $('.topo', this.el).onclick = () => this.analyze();
    $('.ref', this.el).onclick = () => { S.refIdx = S.sel.indexOf(this.run.id); updateRef(); };
    $('.update-banner button', this.el).onclick = () => { const f = bestFinal(this.run); if (f) this.load(`final:${f.name}`); };
    for (const ev of ['pointerdown', 'wheel', 'touchstart']) this.slot.addEventListener(ev, () => markSeen([this.run.id]), { passive: true, once: true });
    this.update(run);
  }

  get group() { return S.runGroup.get(this.run.id); }

  sources() {
    const r = this.run, out = [];
    for (const f of [...r.finals].reverse()) out.push({ key: `final:${f.name}`, kind: 'final', file: f, label: `Web GLB${f.size ? ` · ${fmtBytes(f.size)}` : ''}` });
    for (const f of r.objs) out.push({ key: `obj:${f.name}`, kind: 'obj', file: f, mtl: { url: f.url.replace(/\.obj$/i, '.mtl'), size: 0 }, label: `Quad OBJ${f.size ? ` · ${fmtBytes(f.size)}` : ''}${f.stats?.quads ? ` · ${fmtShort(f.stats.quads)} quads` : ''}` });
    for (const f of [...r.raws].reverse()) out.push({ key: `raw:${f.name}`, kind: 'raw', file: f, label: `Raw mesh · ${fmtBytes(f.size)}${f.stats?.faces ? ` · ${fmtShort(f.stats.faces)} f` : ''} ⚠` });
    return out;
  }

  update(run) {
    this.run = run;
    const g = this.group;
    const idx = S.sel.indexOf(run.id);
    $('.pane-dot', this.el).style.background = PANE_COLORS[idx % PANE_COLORS.length] || PANE_COLORS[0];
    $('.stem', this.el).textContent = g?.stem || '';
    $('.tag', this.el).innerHTML = esc(run.tag) + (run.status === 'running' ? '<span class="badge running">running</span>' : '');
    $('.tag', this.el).title = run.id;
    $('.run-note', this.el).textContent = runNote(run);
    $('.turn', this.el).hidden = !run.renders?.length;
    const srcs = this.sources();
    const sel = $('.pane-src', this.el);
    const cur = this.loadingKey || this.loaded?.key || '';
    sel.innerHTML = (srcs.length ? '' : '<option value="">No model files yet</option>') + srcs.map((s) => `<option value="${esc(s.key)}" ${s.file.stable ? '' : 'disabled'}>${esc(s.label)}</option>`).join('');
    if (cur) sel.value = cur;
    // textures for the photo overlay
    const photoUrl = run.crop?.url || groupImageUrl(g);
    this.view.photo = stage.texture(photoUrl);
    this.view.mask = stage.texture(run.mask?.url || photoUrl);
    this.view.maskAlpha = !run.mask;
    if (photoUrl) {
      const pip = $('.pip img', this.el), full = $('.photo-full img', this.el);
      const thumb = run.crop?.url || groupThumb(g) || photoUrl;
      if (pip.getAttribute('src') !== thumb) pip.src = thumb;
      if (full.getAttribute('src') !== photoUrl) full.src = photoUrl;
    }
    // auto-load / new-version handling (phones: only the visible pane)
    const best = bestFinal(run);
    if (!this.loaded && !this.loadingKey && !this.error) {
      if (best && this.isActive()) this.load(`final:${best.name}`);
      else if (best) this.showMsg('Not loaded (phones load one run at a time).', undefined, [{ label: 'Load', primary: true, fn: () => this.load(`final:${best.name}`) }]);
      else this.showWaiting();
    } else if (this.loaded?.kind === 'final' && best) {
      const lf = this.loaded.file;
      const newer = best.name !== lf.name || best.mtime !== lf.mtime || best.size !== lf.size;
      const ban = $('.update-banner', this.el);
      ban.hidden = !newer;
      if (newer) $('span', ban).textContent = `New model: ${best.name} (${fmtBytes(best.size)})`;
    }
    if (!this.loaded && !this.loadingKey && !best && !this.error) this.showWaiting();
  }

  isActive() { return !isPhone() || S.active === this.run.id; }

  unload() {
    this.abort?.abort();
    this.loadingKey = null;
    if (this.loaded) { stage.clearModel(this.view); this.loaded = null; this.topo = null; }
  }

  gone() {
    this.showMsg('This run folder was removed.', null);
  }

  showMsg(msg, progress, actions = []) {
    const L = $('.loader', this.el);
    L.hidden = false;
    $('.msg', L).innerHTML = msg;
    const bar = $('.bar', L);
    bar.hidden = progress === undefined;
    bar.classList.toggle('indet', progress === null);
    $('i', bar).style.width = progress == null ? '' : `${Math.round(progress * 100)}%`;
    const A = $('.actions', L);
    A.innerHTML = '';
    for (const a of actions) {
      const b = document.createElement('button');
      b.className = a.primary ? 'primary-btn' : 'ghost-btn';
      b.textContent = a.label; b.onclick = a.fn;
      A.append(b);
    }
  }
  hideMsg() { $('.loader', this.el).hidden = true; }

  showWaiting() {
    const r = this.run;
    const acts = [];
    const obj = this.sources().find((s) => s.kind === 'obj' && s.file.stable);
    const raw = this.sources().find((s) => s.kind === 'raw' && s.file.stable);
    if (obj) acts.push({ label: 'Load quad OBJ', fn: () => this.load(obj.key) });
    if (raw) acts.push({ label: `Load raw (${fmtBytes(raw.file.size)})`, fn: () => this.load(raw.key) });
    if (r.status === 'running') this.showMsg(`Run in progress${r.finals.length ? ' — final GLB still being written' : ' — waiting for the final GLB'}.<br>It will load automatically.`, null, acts);
    else this.showMsg(r.finals.length ? 'Final GLB is still being written…' : 'No final GLB in this run.', undefined, acts);
  }

  async load(key, confirmed = false) {
    const src = this.sources().find((s) => s.key === key);
    if (!src) return;
    if (src.kind === 'raw' && !confirmed && (src.file.size > RAW_CONFIRM_BYTES || isPhone())) {
      const faces = src.file.stats?.faces;
      this.showMsg(`<b>Raw decoder mesh</b><br>${fmtBytes(src.file.size)}${faces ? ` · ${fmtInt(faces)} faces` : ''}.<br><span class="warn">${isPhone() ? 'On a phone this will likely be very slow and may crash or reload the tab.' : 'Heavy: may be slow or crash a phone browser.'} Shown as flat clay${faces > WIRE_FACE_LIMIT ? ', wireframe disabled' : ''}.</span>`, undefined, [
        { label: 'Load raw mesh', primary: true, fn: () => this.load(key, true) },
        { label: 'Cancel', fn: () => { const s = $('.pane-src', this.el); s.value = this.loaded?.key || ''; if (this.loaded) this.hideMsg(); else this.update(this.run); } },
      ]);
      return;
    }
    this.abort?.abort();
    const ac = new AbortController();
    this.abort = ac;
    this.loadingKey = key; this.error = null;
    $('.pane-src', this.el).value = key;
    $('.update-banner', this.el).hidden = true;
    const name = esc(src.file.name);
    const cancel = { label: 'Cancel', fn: () => { ac.abort(); this.loadingKey = null; if (this.loaded) { this.hideMsg(); $('.pane-src', this.el).value = this.loaded.key; } else this.showMsg('Cancelled.', undefined, [{ label: 'Retry', primary: true, fn: () => this.load(key, true) }]); } };
    this.showMsg(`Queued ${name}`, 0, [cancel]);
    await acquire();
    let released = false;
    const rel = () => { if (!released) { released = true; release(); } };
    try {
      if (ac.signal.aborted) return;
      const t0 = performance.now();
      const prog = (got, total) => this.showMsg(`Downloading ${name}<br>${fmtBytes(got)} / ${fmtBytes(total || src.file.size)}`, total ? got / total : null, [cancel]);
      let stats;
      if (src.kind === 'obj') {
        const text = await fetchText(src.file.url, src.file.size, prog, ac.signal);
        const mtl = src.mtl ? await fetchText(src.mtl.url, src.mtl.size, null, ac.signal).catch(() => null) : null;
        const dlMs = Math.round(performance.now() - t0);
        rel();
        if (ac.signal.aborted) return;
        this.showMsg(`Parsing ${name}…`, null);
        await nextFrame();
        stats = await stage.loadOBJ(this.view, text, mtl, src.file.url.replace(/[^/]*$/, ''));
        stats.dlMs = dlMs;
      } else {
        const bytes = await fetchBuffer(src.file.url, src.file.size || 0, prog, ac.signal);
        src.got = bytes.length;
        const dlMs = Math.round(performance.now() - t0);
        rel();
        if (ac.signal.aborted) return;
        this.showMsg(`Parsing ${name}${src.kind === 'raw' ? ' (large)' : ''}…`, null);
        await nextFrame();
        stats = await stage.loadGLB(this.view, bytes, { raw: src.kind === 'raw' });
        stats.dlMs = dlMs;
      }
      if (this.abort !== ac) return;
      stats.bytes = src.got || src.file.size;
      this.loaded = { key, kind: src.kind, file: src.file, lite: !!src.lite };
      this.loadingKey = null; this.topo = null;
      this.hideMsg();
      const tag = $('.corner-tag', this.el);
      tag.hidden = src.kind === 'final' && !src.lite;
      tag.textContent = src.lite ? 'lite 1K' : src.kind === 'raw' ? `raw${this.view.faces > WIRE_FACE_LIMIT ? ' · no wire' : ''}` : src.kind === 'obj' ? 'quad obj' : '';
      markSeen([this.run.id]);
      renderSidebar(); renderGallery();
      updateRef();
      renderStats();
      this.update(this.run);
    } catch (e) {
      if (e.name === 'AbortError' || ac.signal.aborted) return;
      console.error(e);
      this.loadingKey = null; this.error = e;
      this.showMsg(`<span class="warn">Failed to load ${name}</span><br>${esc(e.message || e)}`, undefined, [{ label: 'Retry', primary: true, fn: () => { this.error = null; this.load(key, true); } }]);
    } finally {
      rel();
    }
  }

  async analyze() {
    if (!this.view.model) return;
    if (this.view.faces > 12e6) { toast('Mesh too large for in-browser topology analysis', null, 3000); return; }
    this.topo = { busy: true };
    renderStats();
    try { this.topo = await stage.analyzeTopology(this.view); } catch (e) { this.topo = { error: String(e.message || e) }; }
    renderStats();
  }

  dispose() {
    this.abort?.abort();
    stage.removePane(this.view);
    this.el.remove();
  }
}

function syncPanes() {
  const inCompare = S.sel.length > 0 || !!S.focusStem;
  $('#home').hidden = inCompare;
  $('#compare').hidden = !inCompare;
  for (const [id, p] of S.panes) if (!S.sel.includes(id)) { p.dispose(); S.panes.delete(id); }
  const box = $('#panes');
  for (const id of S.sel) {
    let p = S.panes.get(id);
    if (!p) { p = new Pane(S.runs.get(id)); S.panes.set(id, p); }
    box.append(p.el); // keeps DOM order == selection order
  }
  if (!S.sel.includes(S.active)) { S.active = S.sel[0] || null; }
  if (!S.sel.includes(S.abPrev)) S.abPrev = S.sel.find((id) => id !== S.active) || null;
  for (const [id, p] of S.panes) p.el.classList.toggle('active', id === S.active);
  evict();
  for (const p of S.panes.values()) p.update(p.run);
  renderRunTabs();
  document.body.classList.toggle('on-home', !inCompare);
  if (!S.sel.length) {
    box.innerHTML = S.focusStem ? '<div class="card empty">No runs for this input yet — they will appear here live.</div>' : '';
  } else $$('.empty', box).forEach((e) => e.remove());
  if (S.refIdx >= S.sel.length) S.refIdx = 0;
  layoutCols();
  renderSidebar();
  updateRef();
  renderStats();
  stage.requestRender();
}

// phones keep at most two models in memory: the visible run and the A/B partner
function evict() {
  if (!isPhone()) return;
  for (const [id, p] of S.panes) if (id !== S.active && id !== S.abPrev && (p.loaded || p.loadingKey)) p.unload();
}
function activate(id) {
  if (!S.sel.includes(id) || id === S.active) return;
  S.abPrev = S.active; S.active = id;
  for (const [pid, p] of S.panes) p.el.classList.toggle('active', pid === id);
  evict();
  S.panes.get(id)?.update(S.runs.get(id));
  S.refIdx = S.sel.indexOf(id);
  renderRunTabs(); updateRef(); renderStats();
  markSeen([id]);
  stage.requestRender();
}
function renderRunTabs() {
  const box = $('#runTabs');
  if (!box) return;
  box.innerHTML = S.sel.map((id, i) => {
    const r = S.runs.get(id);
    return `<button class="run-tab ${id === S.active ? 'on' : ''}" role="tab" data-id="${esc(id)}" style="--tabc:${PANE_COLORS[i]}"><i style="background:${PANE_COLORS[i]}"></i><span>${esc(r?.tag || id)}</span>${r?.status === 'running' ? '<span class="badge running"></span>' : ''}</button>`;
  }).join('');
  $('#abBtn').disabled = S.sel.length < 2;
  box.querySelector('.on')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}
$('#runTabs').addEventListener('click', (e) => { const b = e.target.closest('[data-id]'); if (b) activate(b.dataset.id); });
$('#abBtn').onclick = () => {
  const other = S.abPrev && S.sel.includes(S.abPrev) ? S.abPrev : S.sel.find((id) => id !== S.active);
  if (other) activate(other);
};
$('#addRunBtn').onclick = () => openDrawer();
// swipe left/right on the run bar to switch runs
(() => {
  let sx = null, sy = 0;
  const bar = $('#runBar');
  bar.addEventListener('touchstart', (e) => { sx = e.touches[0].clientX; sy = e.touches[0].clientY; }, { passive: true });
  bar.addEventListener('touchend', (e) => {
    if (sx == null) return;
    const dx = e.changedTouches[0].clientX - sx, dy = e.changedTouches[0].clientY - sy;
    sx = null;
    if (Math.abs(dx) < 70 || Math.abs(dy) > 40 || e.target.closest('.run-tabs')) return;
    const i = S.sel.indexOf(S.active), j = i + (dx < 0 ? 1 : -1);
    if (j >= 0 && j < S.sel.length) activate(S.sel[j]);
  });
})();

function layoutCols() {
  const n = Math.max(1, S.sel.length);
  const w = $('#panes').clientWidth || window.innerWidth;
  let cols = S.cols === 'auto' ? (n === 1 ? 1 : n === 2 ? 2 : n === 3 ? (w > 900 ? 3 : 2) : (w > 1400 ? 4 : 2)) : Math.min(+S.cols, n);
  $('#panes').style.setProperty('--cols', cols);
  // a single pane should not exceed the viewport height
  $('#panes').style.maxWidth = n === 1 && S.cols === 'auto' ? `max(320px, calc(100dvh - 280px))` : '';
  $('#colsBtn').textContent = `Cols: ${S.cols}`;
  stage.requestRender();
}
$('#colsBtn').onclick = () => {
  const order = ['auto', '1', '2', '3', '4'];
  S.cols = order[(order.indexOf(String(S.cols)) + 1) % order.length];
  store.set('tv.cols', S.cols);
  layoutCols();
};
new ResizeObserver(() => { layoutCols(); stage.requestRender(); }).observe($('#main'));

// ------------------------------------------------------------------ input / overlay panel
function refPane() { return S.panes.get(S.sel[S.refIdx]) || S.panes.get(S.sel[0]) || null; }
function updateRef() {
  const p = refPane();
  const run = p?.run || null;
  const g = run ? S.runGroup.get(run.id) : S.groups.get(S.focusStem);
  const srcAvail = { crop: !!run?.crop, input: !!g?.image, mask: !!run?.mask };
  let src = S.refSrc;
  if (!srcAvail[src]) src = srcAvail.crop ? 'crop' : srcAvail.input ? 'input' : srcAvail.mask ? 'mask' : null;
  $$('.refsrc button').forEach((b) => { b.classList.toggle('on', b.dataset.v === src); b.disabled = !srcAvail[b.dataset.v]; });
  const url = src === 'crop' ? run.crop.url : src === 'mask' ? run.mask.url : src === 'input' ? g.image.url : '';
  const img = $('#refImg');
  if (img.getAttribute('src') !== url) { if (url) img.src = url; else img.removeAttribute('src'); }
  $('#refTitle').textContent = g ? (g.title || g.stem) : '';
  // models
  const sel = $('#refModel');
  sel.innerHTML = S.sel.map((id, i) => `<option value="${i}">${esc(S.runs.get(id)?.tag || id)}</option>`).join('') || '<option>—</option>';
  sel.value = String(S.refIdx);
  sel.disabled = S.sel.length < 2;
  // fov
  const metaFov = run?.meta?.fov;
  const key = run?.id || '';
  const fov = S.fovOverride[key] ?? metaFov ?? 30;
  for (const fi of $$('.fov-input')) if (document.activeElement !== fi) fi.value = (+fov).toFixed(2).replace(/\.?0+$/, '');
  const fovNote = S.fovOverride[key] != null ? 'manual' : metaFov != null ? 'from meta.json' : run ? 'no meta.json: assumed, edit to tune' : '';
  $$('.fov-note').forEach((e) => { e.textContent = fovNote; });
  if (Math.abs(stage.inputFov - fov) > 1e-6) stage.setInputCamera(+fov);
  // hint
  let hint = '';
  if (run && !run.crop && src !== 'mask') hint = 'No crop.png: showing original input, overlay alignment approximate';
  else if (p && !p.view.model) hint = 'Overlay appears once the model loads';
  else if (!run) hint = '';
  $('#refHint').textContent = hint;
  // overlay
  $$('.ovmode button').forEach((b) => b.classList.toggle('on', b.dataset.v === S.ov.mode));
  $$('.ov-slider').forEach((e) => { if (document.activeElement !== e) e.value = S.ov.value; });
  $$('.ov-outline').forEach((e) => e.classList.toggle('on', S.ov.outline));
  $('#phOv').hidden = !S.look.photoInPanes;
  const photoTex = url ? stage.texture(url) : null;
  stage.setRef(p ? {
    slot: $('#refSlot'), pane: p.view, photo: photoTex,
    mask: p.view.mask, maskAlpha: p.view.maskAlpha,
    mode: S.ov.mode, value: S.ov.value, split: S.ov.value, outline: S.ov.outline,
  } : null);
  if (stage.matched && run) stage.matchInputCamera();
  $('#matchCam').classList.toggle('hl', stage.matched);
}
$$('.refsrc').forEach((el) => el.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b || b.disabled) return; e.stopPropagation(); S.refSrc = b.dataset.v; store.set('tv.refSrc', S.refSrc); updateRef(); }));
$$('.ovmode').forEach((el) => el.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; S.ov.mode = b.dataset.v; store.set('tv.ov', S.ov); updateRef(); }));
$$('.ov-slider').forEach((el) => el.addEventListener('input', (e) => { S.ov.value = +e.target.value; store.set('tv.ov', S.ov); updateRef(); }));
$$('.ov-outline').forEach((el) => el.addEventListener('click', (e) => { e.stopPropagation(); S.ov.outline = !S.ov.outline; store.set('tv.ov', S.ov); updateRef(); }));
$('#refPanel').classList.toggle('folded', !!store.get('tv.refFolded', false));
$('#refFold').onclick = () => { const f = $('#refPanel').classList.toggle('folded'); store.set('tv.refFolded', f); stage.requestRender(); };
$('#refModel').onchange = (e) => { S.refIdx = +e.target.value; updateRef(); };
$$('.fov-input').forEach((el) => el.addEventListener('change', (e) => {
  const run = refPane()?.run; if (!run) return;
  const v = parseFloat(e.target.value);
  if (!(v > 1 && v < 170)) return;
  if (run.meta?.fov != null && Math.abs(run.meta.fov - v) < 1e-6) delete S.fovOverride[run.id]; else S.fovOverride[run.id] = v;
  updateRef();
}));
// drag across the reference image to scrub the blend / wipe position
(() => {
  const slot = $('#refSlot');
  let down = false, sx = 0, sy = 0, horiz = null;
  const set = (e) => {
    const r = slot.getBoundingClientRect();
    S.ov.value = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    updateRef();
  };
  slot.addEventListener('pointerdown', (e) => { down = true; sx = e.clientX; sy = e.clientY; horiz = e.pointerType === 'mouse' ? true : null; if (horiz) set(e); });
  slot.addEventListener('pointermove', (e) => {
    if (!down) return;
    if (horiz === null) { const dx = Math.abs(e.clientX - sx), dy = Math.abs(e.clientY - sy); if (dx + dy > 6) horiz = dx > dy; }
    if (horiz) set(e);
  });
  const up = () => { down = false; store.set('tv.ov', S.ov); };
  slot.addEventListener('pointerup', up); slot.addEventListener('pointercancel', up);
  slot.addEventListener('dblclick', () => { const u = $('#refImg').src; if (u) lightbox(u, $('#refTitle').textContent); });
})();

// ------------------------------------------------------------------ toolbar
function syncToolbar() {
  for (const seg of $$('.toolbar .seg[data-key]')) for (const b of $$('button', seg)) b.classList.toggle('on', S.look[seg.dataset.key] === b.dataset.v);
  for (const t of $$('.toolbar .tog[data-key]')) t.classList.toggle('on', !!S.look[t.dataset.key]);
  $('#matchCam').classList.toggle('hl', stage.matched);
  for (const b of $$('#bottomBar [data-mode]')) b.classList.toggle('on', S.look.mode === b.dataset.mode);
  $('#bbMatch').classList.toggle('on', stage.matched);
  $('#bbRotate').classList.toggle('on', !!S.look.autoRotate);
  $('#bbOverlay').classList.toggle('on', !!S.look.photoInPanes);
  const ph = $('#phOv'); if (ph) ph.hidden = !S.look.photoInPanes;
  document.body.classList.toggle('ov-on', !!S.look.photoInPanes);
}
function setLook(patch) {
  if (patch.photoInPanes && !S.look.photoInPanes && isPhone()) { stage.matchInputCamera(); } // overlay only lines up from the input camera
  Object.assign(S.look, patch);

  store.set('tv.look', S.look);
  stage.photoInPanes = S.look.photoInPanes;
  stage.setLook(S.look);
  syncToolbar();
}
$('#toolbar').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  const seg = b.closest('.seg[data-key]');
  if (seg) return setLook({ [seg.dataset.key]: b.dataset.v });
  if (b.classList.contains('tog') && b.dataset.key) return setLook({ [b.dataset.key]: !S.look[b.dataset.key] });
  if (b.dataset.view) { stage.viewFrom(b.dataset.view); syncToolbar(); }
});
// bottom action bar + "More" sheet (phones)
const openSheet = () => document.body.classList.add('sheet');
const closeSheet = () => { document.body.classList.remove('sheet'); stage.requestRender(); };
$('#bottomBar').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.mode) setLook({ mode: b.dataset.mode });
  else if (b.id === 'bbMatch') { stage.matchInputCamera(); syncToolbar(); }
  else if (b.id === 'bbRotate') setLook({ autoRotate: !S.look.autoRotate });
  else if (b.id === 'bbOverlay') { setLook({ photoInPanes: !S.look.photoInPanes }); updateRef(); }
  else if (b.id === 'bbMore') document.body.classList.contains('sheet') ? closeSheet() : openSheet();
});
$('#sheetDone').onclick = closeSheet;
$('#sheetScrim').onclick = closeSheet;
stage.onUserCamera = () => { $('#matchCam').classList.remove('hl'); $('#bbMatch').classList.remove('on'); };
$('#matchCam').onclick = (e) => { e.stopPropagation(); stage.matchInputCamera(); syncToolbar(); };
$('#frameBtn').onclick = (e) => { e.stopPropagation(); stage.frameAll(); syncToolbar(); };
syncToolbar();
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  const k = e.key.toLowerCase();
  if (k === 't') setLook({ mode: 'textured' });
  else if (k === 'c') setLook({ mode: 'clay' });
  else if (k === 'w') setLook({ mode: 'wire' });
  else if (k === 'o') setLook({ wire: !S.look.wire });
  else if (k === 'n') setLook({ normals: !S.look.normals });
  else if (k === 'r') setLook({ autoRotate: !S.look.autoRotate });
  else if (k === 'm') { stage.matchInputCamera(); syncToolbar(); }
  else if (k === 'f') { stage.frameAll(); syncToolbar(); }
  else if (k === 'escape') { $('#lightbox').hidden = true; closeDrawer(); }
});

// ------------------------------------------------------------------ stats table
function renderStats() {
  const panes = S.sel.map((id) => S.panes.get(id)).filter(Boolean);
  const T = $('#statsTable');
  if (!panes.length) { T.innerHTML = ''; return; }
  const head = `<thead><tr><th></th>${panes.map((p, i) => `<th><span class="pane-dot" style="display:inline-block;background:${PANE_COLORS[i]};margin-right:6px"></span>${esc(p.run.tag)}</th>`).join('')}</tr></thead>`;
  const rows = [];
  const sec = (t) => rows.push(`<tr class="sec"><th colspan="${panes.length + 1}">${t}</th></tr>`);
  const row = (label, fn, best) => {
    const vals = panes.map((p) => { try { return fn(p); } catch { return null; } });
    let bi = -1;
    if (best && panes.length > 1) {
      const nums = vals.map((v) => (v && typeof v === 'object' ? v.n : null));
      const valid = nums.filter((n) => n != null);
      if (valid.length > 1 && new Set(valid).size > 1) { const t = best === 'min' ? Math.min(...valid) : Math.max(...valid); bi = nums.indexOf(t); }
    }
    rows.push(`<tr><th>${label}</th>${vals.map((v, i) => `<td class="${i === bi ? 'best' : ''}">${v && typeof v === 'object' ? v.html : v ?? '–'}</td>`).join('')}</tr>`);
  };
  const N = (n, html) => ({ n, html: html ?? fmtInt(n) });
  sec('Model');
  row('Showing', (p) => p.loaded ? esc(p.loaded.file.name) + (p.loaded.lite ? `<br><span class="muted">lite 1K textures · ${fmtBytes(p.view.stats?.bytes)} downloaded</span>` : '') : p.loadingKey ? '<span class="muted">loading…</span>' : '<span class="muted">not loaded</span>');
  row('File size', (p) => p.loaded ? fmtBytes(p.loaded.file.size) : bestFinal(p.run) ? fmtBytes(bestFinal(p.run).size) : null);
  row('Vertices', (p) => p.view.stats ? N(p.view.stats.vertices) : bestFinal(p.run)?.stats?.vertices ? `${fmtInt(bestFinal(p.run).stats.vertices)} <span class="muted">(header)</span>` : null);
  row('Faces (tris)', (p) => p.view.stats ? N(p.view.stats.faces) : bestFinal(p.run)?.stats?.faces ? `${fmtInt(bestFinal(p.run).stats.faces)} <span class="muted">(header)</span>` : null);
  if (panes.some((p) => p.run.objs.length)) row('Quad OBJ', (p) => { const s = p.run.objs[p.run.objs.length - 1]?.stats; return s ? `${fmtInt(s.quads)} quads · ${fmtInt(s.tris)} tris${s.ngons ? ` · ${fmtInt(s.ngons)} ngons` : ''}<br><span class="muted">${fmtPct(s.quadRatio)} quads · ${fmtInt(s.vertices)} v</span>` : null; });
  if (panes.some((p) => p.run.raws.length)) row('Raw mesh', (p) => { const f = p.run.raws[p.run.raws.length - 1]; return f ? `${f.stats?.faces ? fmtInt(f.stats.faces) + ' f · ' : ''}${fmtBytes(f.size)}` : null; });
  row('Textures', (p) => p.view.stats ? (p.view.stats.textures.map((t) => `${t.slots.join('+')} ${t.w}²`.replace(`${t.w}²`, t.w === t.h ? `${t.w}²` : `${t.w}×${t.h}`)).join('<br>') || 'none') : (bestFinal(p.run)?.stats?.maps?.join(', ') || null));
  row('Bounds', (p) => p.view.stats ? p.view.stats.bbox.map((x) => x.toFixed(3)).join(' × ') : null);
  row('Download', (p) => p.view.stats ? fmtMs(p.view.stats.dlMs) : null);
  row('Parse', (p) => p.view.stats ? fmtMs(p.view.stats.parseMs) : null);
  sec('Precomputed (full-resolution export)');
  const P = (k, f) => (p) => (p.run.pre?.[k] == null ? null : f(p.run.pre[k]));
  row('Full-res faces', P('full_res_faces', (v) => fmtInt(v)));
  row('Tri quality (mean)', P('tri_quality_mean', (v) => N(v, (+v).toFixed(3))), 'max');
  row('Slivers', P('slivers_pct', (v) => N(v, `${(+v).toFixed(2)}%`)), 'min');
  row('Boundary edges', P('boundary_edges', (v) => N(v)), 'min');
  row('Non-manifold edges', P('nonmanifold_edges', (v) => N(v)), 'min');
  if (panes.some((p) => p.run.pre?.psnr != null)) row('PSNR vs input', P('psnr', (v) => N(v, `${(+v).toFixed(2)} dB`)), 'max');
  if (panes.some((p) => p.run.pre?.hf_corr != null)) row('HF detail corr.', P('hf_corr', (v) => N(v, (+v).toFixed(3))), 'max');
  sec('Run');
  row('Notes', (p) => `<span class="muted">${esc(runNote(p.run))}</span>`);
  row('Preset', (p) => esc(p.run.meta?.preset ?? '–'));
  row('Model', (p) => esc(p.run.meta?.model ?? '–'));
  row('Seed', (p) => esc(p.run.meta?.seed ?? '–'));
  row('FOV', (p) => p.run.meta?.fov != null ? `${(+p.run.meta.fov).toFixed(2)}°` : '<span class="muted">n/a</span>');
  row('Render time', (p) => p.run.meta?.seconds != null ? N(p.run.meta.seconds, fmtDur(p.run.meta.seconds)) : null, 'min');
  row('Overrides', (p) => p.run.meta?.overrides && Object.keys(p.run.meta.overrides).length ? `<pre>${esc(JSON.stringify(p.run.meta.overrides, null, 1))}</pre>` : '<span class="muted">none</span>');
  row('Created', (p) => `${new Date(p.run.created * 1000).toLocaleString()} · ${fmtAgo(p.run.created)}`);
  if (panes.some((p) => p.run.renders?.length)) row('Renders', (p) => (p.run.renders || []).map((u, i) => `<a href="${esc(u)}" class="lb-link" data-cap="${esc(p.run.tag)}">render ${i + 1}</a>`).join(' · ') || null);
  row('meta.json', (p) => p.run.meta ? `<pre>${esc(JSON.stringify(p.run.meta, null, 1))}</pre>` : '<span class="muted">missing</span>');
  sec('Topology <span class="muted" style="text-transform:none;letter-spacing:0">(press “Topology” on a pane; welded by position)</span>');
  const T_ = (fn) => (p) => !p.topo ? null : p.topo.busy ? '<span class="muted">analyzing…</span>' : p.topo.error ? `<span class="warn">${esc(p.topo.error)}</span>` : fn(p.topo);
  row('Welded verts', T_((t) => N(t.weldedVertices)));
  row('Edges', T_((t) => N(t.edges)));
  row('Boundary edges', T_((t) => N(t.boundaryEdges)), 'min');
  row('Non-manifold edges', T_((t) => N(t.nonManifoldEdges)), 'min');
  row('Components', T_((t) => N(t.components)), 'min');
  row('Genus', T_((t) => t.genus == null ? '<span class="muted">open mesh</span>' : fmtInt(t.genus)));
  row('Degenerate tris', T_((t) => N(t.degenerate)), 'min');
  row('Valence 6', T_((t) => N(t.valence6, fmtPct(t.valence6))), 'max');
  row('Poles (≤3 or ≥9)', T_((t) => N(t.irregularPoles, fmtPct(t.irregularPoles, 2))), 'min');
  row('Tri quality (mean)', T_((t) => N(t.quality, t.quality.toFixed(3))), 'max');
  row('Slivers (q<0.3)', T_((t) => N(t.slivers, fmtPct(t.slivers, 2))), 'min');
  if (panes.some((p) => p.topo?.quadValence4 != null)) row('Quad valence 4', T_((t) => t.quadValence4 == null ? '–' : N(t.quadValence4, fmtPct(t.quadValence4))), 'max');
  row('Analysis time', T_((t) => fmtMs(t.ms)));
  sec('Files');
  row('Download', (p) => `<div class="files">${['finals', 'objs'].flatMap((k) => p.run[k]).concat([p.run.crop, p.run.mask].filter(Boolean)).map((f) => `<a href="${esc(f.url)}" download>${esc(f.url.split('/').pop())}${f.size ? ` <span class="muted">${fmtBytes(f.size)}</span>` : ''}</a>`).join('')}</div>`);
  T.innerHTML = head + `<tbody>${rows.join('')}</tbody>`;
  if (isPhone()) {
    const ci = panes.findIndex((p) => p.run.id === S.active) + 2;
    if (ci > 1) $$(`tr > *:nth-child(${ci})`, T).forEach((c) => c.classList.add('active-col'));
  }
}

phoneMQ.addEventListener('change', () => { closeSheet(); syncPanes(); stage.requestRender(); });
window.addEventListener('orientationchange', () => setTimeout(() => { layoutCols(); stage.requestRender(); }, 300));

$('#statsTable').addEventListener('click', (e) => { const a = e.target.closest('.lb-link'); if (a) { e.preventDefault(); lightbox(a.getAttribute('href'), a.dataset.cap); } });

// ------------------------------------------------------------------ about
$('#aboutBtn').onclick = () => { $('#about').hidden = false; };
$('#about').addEventListener('click', (e) => { if (e.target.id === 'about' || e.target.closest('.about-close')) $('#about').hidden = true; });

// ------------------------------------------------------------------ boot
setLive('connecting');
loadIndex();
setInterval(() => { if (S.data) { renderSidebar(); renderGallery(); } }, 60000); // refresh "x min ago"
window.__viewer = { S, stage, loadIndex };
