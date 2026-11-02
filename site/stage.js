// Three.js stage: one fixed full-window canvas, many scissored "slots" (panes + the
// input-camera reference view) that share a single camera, so geometry is uploaded once
// and every pane always shows exactly the same view.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const THREE_BASE = 'https://cdn.jsdelivr.net/npm/three@0.160.0/';
export const WIRE_FACE_LIMIT = 3_000_000;

const BGS = { dark: 0x15181d, grey: 0x5a5f66, light: 0xe9ebee };

function makeMatcap() {
  const s = 256, c = document.createElement('canvas');
  c.width = c.height = s;
  const g = c.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, s, s);
  const base = g.createRadialGradient(s * 0.38, s * 0.32, s * 0.02, s * 0.5, s * 0.5, s * 0.5);
  base.addColorStop(0, '#fbf8f2');
  base.addColorStop(0.25, '#d9d4cb');
  base.addColorStop(0.62, '#8f8a83');
  base.addColorStop(0.9, '#3b3936');
  base.addColorStop(1, '#1d1c1b');
  g.fillStyle = base;
  g.beginPath(); g.arc(s / 2, s / 2, s / 2, 0, Math.PI * 2); g.fill();
  // cool rim from below-right
  const rim = g.createRadialGradient(s * 0.5, s * 0.5, s * 0.36, s * 0.5, s * 0.5, s * 0.5);
  rim.addColorStop(0, 'rgba(150,190,255,0)');
  rim.addColorStop(1, 'rgba(150,190,255,0.28)');
  g.fillStyle = rim;
  g.beginPath(); g.arc(s / 2, s / 2, s / 2, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// three r160 only knows EXT_texture_webp; the exporter writes the ratified KHR_texture_webp.
class KHRTextureWebP {
  constructor(parser) { this.parser = parser; this.name = 'KHR_texture_webp'; }
  loadTexture(textureIndex) {
    const parser = this.parser, json = parser.json;
    const def = json.textures[textureIndex];
    const ext = def.extensions && def.extensions[this.name];
    if (!ext) return null;
    const source = json.images[ext.source];
    let loader = parser.textureLoader;
    if (source.uri) { const h = parser.options.manager.getHandler(source.uri); if (h !== null) loader = h; }
    return parser.loadTextureImage(textureIndex, ext.source, loader);
  }
}

// ------------------------------------------------------------------ download with progress
export async function fetchBuffer(url, expected, onProgress, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get('content-length')) || expected || 0;
  if (!res.body || !res.body.getReader) {
    const b = new Uint8Array(await res.arrayBuffer());
    onProgress?.(b.length, b.length);
    return b;
  }
  const reader = res.body.getReader();
  let buf = total ? new Uint8Array(total) : null;
  const chunks = [];
  let got = 0, lastT = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && got + value.length <= buf.length) buf.set(value, got);
    else {
      if (buf) { chunks.push(buf.subarray(0, got)); buf = null; }
      chunks.push(value);
    }
    got += value.length;
    const now = performance.now();
    if (now - lastT > 80) { lastT = now; onProgress?.(got, total); }
  }
  onProgress?.(got, total || got);
  if (buf) return got === buf.length ? buf : buf.subarray(0, got);
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

export async function fetchText(url, expected, onProgress, signal) {
  const b = await fetchBuffer(url, expected, onProgress, signal);
  return new TextDecoder().decode(b);
}

// ------------------------------------------------------------------ OBJ polygon edges (true quads)
function objPolygonEdges(text) {
  const pos = [];
  const keys = new Set();
  const edges = [];
  let start = 0;
  const n = text.length;
  while (start < n) {
    let end = text.indexOf('\n', start);
    if (end < 0) end = n;
    const c0 = text.charCodeAt(start), c1 = text.charCodeAt(start + 1);
    if (c0 === 118 && c1 === 32) { // "v "
      const p = text.slice(start + 2, end).trim().split(/\s+/);
      pos.push(+p[0], +p[1], +p[2]);
    } else if (c0 === 102 && c1 === 32) { // "f "
      const p = text.slice(start + 2, end).trim().split(/\s+/);
      const vs = p.map((t) => {
        let i = parseInt(t, 10);
        return i < 0 ? pos.length / 3 + i : i - 1;
      });
      for (let k = 0; k < vs.length; k++) {
        const a = vs[k], b = vs[(k + 1) % vs.length];
        const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
        if (!keys.has(key)) { keys.add(key); edges.push(a, b); }
      }
    }
    start = end + 1;
  }
  return { positions: new Float32Array(pos), edges: new Uint32Array(edges) };
}

// ------------------------------------------------------------------ topology analysis (worker)
function topoWorker() {
  self.onmessage = (ev) => {
    const { pos, idx, polyEdges, polyVerts } = ev.data;
    const t0 = performance.now();
    const nv = pos.length / 3;
    // weld coincident vertices (glTF splits at UV / normal seams)
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity, minz = Infinity, maxz = -Infinity;
    for (let i = 0; i < nv; i++) {
      const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
      if (x < minx) minx = x; if (x > maxx) maxx = x;
      if (y < miny) miny = y; if (y > maxy) maxy = y;
      if (z < minz) minz = z; if (z > maxz) maxz = z;
    }
    const diag = Math.hypot(maxx - minx, maxy - miny, maxz - minz) || 1;
    const q = 1e6 / diag;
    const map = new Map();
    const weld = new Uint32Array(nv);
    let W = 0;
    for (let i = 0; i < nv; i++) {
      const k = Math.round(pos[3 * i] * q) + ',' + Math.round(pos[3 * i + 1] * q) + ',' + Math.round(pos[3 * i + 2] * q);
      let w = map.get(k);
      if (w === undefined) { w = W++; map.set(k, w); }
      weld[i] = w;
    }
    map.clear();
    const F0 = idx ? idx.length / 3 : nv / 3;
    const tri = new Uint32Array(F0 * 3);
    let F = 0, degenerate = 0;
    for (let f = 0; f < F0; f++) {
      const a = weld[idx ? idx[3 * f] : 3 * f], b = weld[idx ? idx[3 * f + 1] : 3 * f + 1], c = weld[idx ? idx[3 * f + 2] : 3 * f + 2];
      if (a === b || b === c || a === c) { degenerate++; continue; }
      tri[3 * F] = a; tri[3 * F + 1] = b; tri[3 * F + 2] = c; F++;
    }
    // edges via sorted keys
    const keys = new Float64Array(F * 3);
    for (let f = 0; f < F; f++) {
      for (let k = 0; k < 3; k++) {
        const a = tri[3 * f + k], b = tri[3 * f + ((k + 1) % 3)];
        keys[3 * f + k] = a < b ? a * W + b : b * W + a;
      }
    }
    keys.sort();
    let E = 0, boundary = 0, nonManifold = 0;
    const valence = new Uint32Array(W);
    for (let i = 0; i < keys.length;) {
      let j = i + 1;
      while (j < keys.length && keys[j] === keys[i]) j++;
      const cnt = j - i;
      E++;
      if (cnt === 1) boundary++; else if (cnt > 2) nonManifold++;
      const a = Math.floor(keys[i] / W), b = keys[i] - a * W;
      valence[a]++; valence[b]++;
      i = j;
    }
    // components (union-find)
    const parent = new Int32Array(W);
    for (let i = 0; i < W; i++) parent[i] = i;
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (let f = 0; f < F; f++) {
      const a = find(tri[3 * f]), b = find(tri[3 * f + 1]), c = find(tri[3 * f + 2]);
      if (a !== b) parent[a] = b;
      const bb = find(b);
      if (c !== bb) parent[c] = bb;
    }
    let comps = 0, used = 0;
    const touched = new Uint8Array(W);
    for (let i = 0; i < 3 * F; i++) touched[tri[i]] = 1;
    for (let i = 0; i < W; i++) if (touched[i]) { used++; if (find(i) === i) comps++; }
    // triangle quality q = 4*sqrt(3)*A / (a^2+b^2+c^2)
    let qsum = 0, slivers = 0;
    const qh = new Uint32Array(10);
    const P = (i, o) => pos[3 * i + o];
    // need un-welded positions per welded vertex: keep first occurrence
    const rep = new Uint32Array(W);
    for (let i = nv - 1; i >= 0; i--) rep[weld[i]] = i;
    for (let f = 0; f < F; f++) {
      const ia = rep[tri[3 * f]], ib = rep[tri[3 * f + 1]], ic = rep[tri[3 * f + 2]];
      const ux = P(ib, 0) - P(ia, 0), uy = P(ib, 1) - P(ia, 1), uz = P(ib, 2) - P(ia, 2);
      const vx = P(ic, 0) - P(ia, 0), vy = P(ic, 1) - P(ia, 1), vz = P(ic, 2) - P(ia, 2);
      const wx = P(ic, 0) - P(ib, 0), wy = P(ic, 1) - P(ib, 1), wz = P(ic, 2) - P(ib, 2);
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      const A2 = Math.hypot(cx, cy, cz); // 2*area
      const s = ux * ux + uy * uy + uz * uz + vx * vx + vy * vy + vz * vz + wx * wx + wy * wy + wz * wz;
      const qq = s > 0 ? (2 * Math.sqrt(3) * A2) / s : 0;
      qsum += qq;
      if (qq < 0.3) slivers++;
      qh[Math.min(9, Math.floor(qq * 10))]++;
    }
    // valence histogram (triangle edges)
    const vh = {};
    let v6 = 0, v57 = 0, poles = 0;
    for (let i = 0; i < W; i++) {
      if (!touched[i]) continue;
      const v = valence[i];
      vh[v] = (vh[v] || 0) + 1;
      if (v === 6) v6++;
      if (v >= 5 && v <= 7) v57++;
      if (v <= 3 || v >= 9) poles++;
    }
    const out = {
      weldedVertices: used, faces: F, degenerate, edges: E, boundaryEdges: boundary, nonManifoldEdges: nonManifold,
      components: comps, euler: used - E + F,
      genus: boundary === 0 && nonManifold === 0 ? (2 * comps - (used - E + F)) / 2 : null,
      valence6: v6 / used, valence5to7: v57 / used, irregularPoles: poles / used,
      quality: qsum / F, slivers: slivers / F, qualityHist: Array.from(qh), valenceHist: vh,
    };
    // polygon (quad) valence from OBJ edges
    if (polyEdges && polyVerts) {
      const pv = new Uint32Array(polyVerts);
      for (let i = 0; i < polyEdges.length; i++) pv[polyEdges[i]]++;
      let v4 = 0, n = 0, pp = 0;
      for (let i = 0; i < polyVerts; i++) { if (!pv[i]) continue; n++; if (pv[i] === 4) v4++; if (pv[i] !== 4 && pv[i] !== 3) pp++; }
      out.quadValence4 = v4 / n; out.quadPoles = pp / n; out.polyEdges = polyEdges.length / 2;
    }
    out.ms = Math.round(performance.now() - t0);
    self.postMessage(out);
  };
}
const TOPO_SRC = `(${topoWorker.toString()})();`;

// ------------------------------------------------------------------ overlay (photo / mask) pass
const overlayVS = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const photoFS = `
uniform sampler2D map; uniform float opacity; uniform float split; uniform int mode;
varying vec2 vUv;
void main(){
  vec4 c = texture2D(map, vUv);
  float a = opacity;
  if (mode == 1) a = vUv.x < split ? 1.0 : 0.0;
  if (a <= 0.0) discard;
  gl_FragColor = vec4(c.rgb, a);
  #include <colorspace_fragment>
}`;
const outlineFS = `
uniform sampler2D map; uniform vec2 texel; uniform vec3 color; uniform int useAlpha;
varying vec2 vUv;
float m(vec2 uv){ vec4 c = texture2D(map, uv); return useAlpha == 1 ? c.a : max(c.r, max(c.g, c.b)); }
void main(){
  float c = step(0.5, m(vUv));
  float e = 0.0;
  for (int i = -2; i <= 2; i++) for (int j = -2; j <= 2; j++) {
    e = max(e, abs(step(0.5, m(vUv + vec2(float(i), float(j)) * texel)) - c));
  }
  if (e < 0.5) discard;
  gl_FragColor = vec4(color, 0.95);
  #include <colorspace_fragment>
}`;

// ------------------------------------------------------------------ Stage
export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.baseDPR = Math.min(window.devicePixelRatio || 1, 2);
    this.lowDPR = Math.max(1, this.baseDPR * 0.5);
    this.curDPR = this.baseDPR;
    this.renderer.setPixelRatio(this.baseDPR);
    this.baseFov = 30; // vertical = horizontal fov of the square view; widened per slot for portrait
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.setScissorTest(true);
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.005, 100);
    this.target = new THREE.Vector3();
    this.refCamera = new THREE.PerspectiveCamera(30, 1, 0.005, 100);
    this.panes = [];
    this.controls = new Map();
    this.ref = null; // {slot, pane, photo, mask, mode, value, outline}
    this.look = { mode: 'textured', light: 'studio', normals: true, wire: false, bg: 'dark', autoRotate: false };
    this.matched = false;
    this.inputFov = 30;
    this.dirty = true;
    this.photoInPanes = false;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envTex = pmrem.fromScene(new RoomEnvironment(this.renderer), 0.04).texture;
    pmrem.dispose();
    this.matcap = makeMatcap();
    this.texLoader = new THREE.TextureLoader();
    this.texCache = new Map();

    this.gltf = new GLTFLoader();
    const draco = new DRACOLoader();
    draco.setDecoderPath(THREE_BASE + 'examples/jsm/libs/draco/gltf/');
    this.gltf.setDRACOLoader(draco);
    this.gltf.setMeshoptDecoder(MeshoptDecoder);
    this.gltf.register((parser) => new KHRTextureWebP(parser));

    // overlay quad scene
    this.ovCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.ovGeo = new THREE.PlaneGeometry(2, 2);
    this.photoMat = new THREE.ShaderMaterial({
      uniforms: { map: { value: null }, opacity: { value: 0.5 }, split: { value: 0.5 }, mode: { value: 0 } },
      vertexShader: overlayVS, fragmentShader: photoFS, transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
    });
    this.outlineMat = new THREE.ShaderMaterial({
      uniforms: { map: { value: null }, texel: { value: new THREE.Vector2(1 / 1024, 1 / 1024) }, color: { value: new THREE.Color(0x39ff9f) }, useAlpha: { value: 0 } },
      vertexShader: overlayVS, fragmentShader: outlineFS, transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
    });
    this.ovScene = new THREE.Scene();
    this.ovMesh = new THREE.Mesh(this.ovGeo, this.photoMat);
    this.ovMesh.frustumCulled = false;
    this.ovScene.add(this.ovMesh);

    this.clock = new THREE.Clock();
    this._resize();
    window.addEventListener('resize', () => this._resize());
    window.visualViewport?.addEventListener('resize', () => this._resize());
    window.addEventListener('orientationchange', () => setTimeout(() => this._resize(), 250));
    window.addEventListener('scroll', () => this.requestRender(), { passive: true, capture: true });
    this.setInputCamera(30);
    this.matchInputCamera();
    const loop = () => {
      requestAnimationFrame(loop);
      this._tick();
    };
    requestAnimationFrame(loop);
  }

  _setDPR(d) {
    if (d === this.curDPR) return;
    this.curDPR = d;
    this.renderer.setPixelRatio(d);
    this._resize();
  }

  _interact(on) {
    clearTimeout(this._dprT);
    if (on) this._setDPR(this.lowDPR);
    else this._dprT = setTimeout(() => this._setDPR(this.baseDPR), 220);
  }

  _resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.requestRender();
  }

  requestRender() { this.dirty = true; }

  // ---------------------------------------------------------------- panes
  addPane(slot) {
    const scene = new THREE.Scene();
    const root = new THREE.Group();
    scene.add(root);
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    const fill = new THREE.HemisphereLight(0xdfe8ff, 0x2a2622, 0.35);
    scene.add(key, key.target, fill);
    const pane = { slot, scene, root, key, model: null, meshes: [], wires: [], polyLines: null, stats: null, isRaw: false, faces: 0, photo: null, mask: null };
    this.panes.push(pane);
    const c = new OrbitControls(this.camera, slot);
    c.target = this.target;
    c.enableDamping = true;
    c.dampingFactor = 0.12;
    c.rotateSpeed = 0.8;
    c.zoomSpeed = 1.1;
    c.screenSpacePanning = true;
    c.autoRotateSpeed = 2.0;
    c.addEventListener('start', () => { this.matched = false; this._interact(true); this.onUserCamera?.(); });
    c.addEventListener('end', () => this._interact(false));
    c.addEventListener('change', () => this.requestRender());
    this.controls.set(pane, c);
    this._applyScene(pane);
    this._syncAutoRotate();
    this.requestRender();
    return pane;
  }

  removePane(pane) {
    this.clearModel(pane);
    const c = this.controls.get(pane);
    c?.dispose();
    this.controls.delete(pane);
    this.panes = this.panes.filter((p) => p !== pane);
    if (this.ref?.pane === pane) this.ref.pane = null;
    this._syncAutoRotate();
    this.requestRender();
  }

  clearModel(pane) {
    if (!pane.model) return;
    pane.root.remove(pane.model);
    const mats = new Set(), texs = new Set(), geos = new Set();
    pane.model.traverse((o) => {
      if (o.geometry) geos.add(o.geometry);
      const m = o.material;
      (Array.isArray(m) ? m : m ? [m] : []).forEach((x) => mats.add(x));
      if (o.userData.orig) mats.add(o.userData.orig);
      Object.values(o.userData.mats || {}).forEach((x) => mats.add(x));
    });
    for (const m of mats) {
      for (const k of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap']) if (m[k] && m[k] !== this.matcap) texs.add(m[k]);
      m.dispose();
    }
    texs.forEach((t) => t.dispose());
    geos.forEach((g) => g.dispose());
    pane.wires.forEach((w) => w.material.dispose());
    if (pane.polyLines) { pane.polyLines.geometry.dispose(); pane.polyLines.material.dispose(); }
    pane.model = null; pane.meshes = []; pane.wires = []; pane.polyLines = null; pane.stats = null; pane.faces = 0;
    this.requestRender();
  }

  async loadGLB(pane, bytes, opts = {}) {
    const t0 = performance.now();
    const ab = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes.buffer : bytes.slice().buffer;
    const gltf = await new Promise((res, rej) => this.gltf.parse(ab, opts.base || '', res, rej));
    const obj = gltf.scene;
    this._install(pane, obj, opts);
    pane.stats.parseMs = Math.round(performance.now() - t0);
    return pane.stats;
  }

  async loadOBJ(pane, text, mtlText, base, opts = {}) {
    const t0 = performance.now();
    const loader = new OBJLoader();
    if (mtlText) {
      const ml = new MTLLoader();
      ml.setResourcePath(base);
      const mats = ml.parse(mtlText, base);
      mats.preload();
      loader.setMaterials(mats);
    }
    const obj = loader.parse(text);
    const pe = objPolygonEdges(text);
    this._install(pane, obj, opts);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pe.positions, 3));
    g.setIndex(new THREE.BufferAttribute(pe.edges, 1));
    pane.polyLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x8ecbff, transparent: true, opacity: 0.7, toneMapped: false }));
    pane.polyLines.userData.polyVerts = pe.positions.length / 3;
    pane.polyLines.userData.polyEdges = pe.edges;
    pane.polyLines.renderOrder = 2;
    pane.model.add(pane.polyLines);
    pane.stats.parseMs = Math.round(performance.now() - t0);
    pane.stats.polyEdges = pe.edges.length / 2;
    this.applyLook(pane);
    return pane.stats;
  }

  _install(pane, obj, opts) {
    this.clearModel(pane);
    pane.isRaw = !!opts.raw;
    pane.model = obj;
    pane.root.add(obj);
    const meshes = [];
    let verts = 0, faces = 0, uv = false, tangents = false;
    const textures = new Map();
    obj.updateMatrixWorld(true);
    obj.traverse((o) => {
      if (!o.isMesh) return;
      meshes.push(o);
      const g = o.geometry;
      const n = g.attributes.position.count;
      verts += n;
      faces += (g.index ? g.index.count : n) / 3;
      if (g.attributes.uv) uv = true;
      if (g.attributes.tangent) tangents = true;
      const m = Array.isArray(o.material) ? o.material[0] : o.material;
      o.userData.orig = m;
      o.userData.mats = {};
      for (const k of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap']) {
        const t = m?.[k];
        if (t?.image) {
          const key = t.uuid;
          const e = textures.get(key) || { slots: [], w: t.image.width, h: t.image.height };
          e.slots.push(k.replace('Map', '') || k);
          textures.set(key, e);
        }
      }
    });
    pane.meshes = meshes;
    pane.faces = faces;
    const box = new THREE.Box3().setFromObject(obj);
    const size = box.getSize(new THREE.Vector3());
    pane.box = box;
    pane.stats = {
      vertices: verts, faces: Math.round(faces), meshes: meshes.length, uv, tangents,
      textures: [...textures.values()].map((t) => ({ slots: [...new Set(t.slots)], w: t.w, h: t.h })),
      bbox: [size.x, size.y, size.z], center: box.getCenter(new THREE.Vector3()).toArray(),
    };
    this.applyLook(pane);
    this.requestRender();
  }

  // ---------------------------------------------------------------- look
  setLook(patch) {
    Object.assign(this.look, patch);
    for (const p of this.panes) { this.applyLook(p); this._applyScene(p); }
    this._syncAutoRotate();
    this.requestRender();
  }

  _applyScene(pane) {
    const L = this.look;
    pane.scene.background = new THREE.Color(BGS[L.bg] ?? BGS.dark);
    pane.scene.environment = L.light === 'studio' ? this.envTex : null;
    pane.key.intensity = L.light === 'studio' ? 1.2 : 0;
  }

  _material(mesh, pane) {
    const L = this.look;
    const o = mesh.userData.orig;
    const hasMap = !!o?.map && !pane.isRaw;
    const textured = L.mode === 'textured' && hasMap;
    const flat = pane.isRaw || !mesh.geometry.attributes.normal;
    const nmap = L.normals && o?.normalMap && !pane.isRaw ? o.normalMap : null;
    const wireOn = this._wireOn(pane);
    const key = [L.mode, L.light, !!nmap, textured, flat, wireOn].join('|');
    if (mesh.userData.mats[key]) return mesh.userData.mats[key];
    const clay = L.mode === 'wire' ? 0x30343b : 0xbdb7ad;
    let m;
    if (L.light === 'unlit') {
      m = new THREE.MeshBasicMaterial({ map: textured ? o.map : null, color: textured ? 0xffffff : clay, toneMapped: false });
      if (!textured) {
        // unlit clay is a flat silhouette; give a hint of shape with matcap instead
        m.dispose();
        m = new THREE.MeshMatcapMaterial({ matcap: this.matcap, color: 0xffffff, flatShading: flat, normalMap: nmap });
        if (L.mode === 'wire') m.color.set(0x555b63);
      }
    } else if (L.light === 'matcap') {
      m = new THREE.MeshMatcapMaterial({ matcap: this.matcap, map: textured ? o.map : null, normalMap: nmap, flatShading: flat, color: textured ? 0xffffff : (L.mode === 'wire' ? 0x555b63 : 0xffffff) });
      if (nmap) m.normalScale.copy(o.normalScale);
    } else if (textured) {
      m = o.clone();
      m.normalMap = nmap;
      m.envMapIntensity = 1.0;
    } else {
      m = new THREE.MeshStandardMaterial({ color: clay, roughness: 0.6, metalness: 0.0, normalMap: nmap, flatShading: flat, envMapIntensity: 0.9 });
      if (nmap) m.normalScale.copy(o.normalScale);
    }
    m.side = THREE.DoubleSide;
    if (wireOn) { m.polygonOffset = true; m.polygonOffsetFactor = 1; m.polygonOffsetUnits = 1; }
    m.needsUpdate = true;
    mesh.userData.mats[key] = m;
    return m;
  }

  _wireOn(pane) {
    return (this.look.wire || this.look.mode === 'wire') && (pane.polyLines || pane.faces <= WIRE_FACE_LIMIT);
  }

  applyLook(pane) {
    if (!pane.model) return;
    const L = this.look;
    for (const mesh of pane.meshes) mesh.material = this._material(mesh, pane);
    const wireOn = this._wireOn(pane);
    const color = L.mode === 'wire' ? 0x8ecbff : L.mode === 'clay' ? 0x1b2430 : 0x000000;
    const opacity = L.mode === 'wire' ? 0.45 : L.mode === 'clay' ? 0.45 : 0.35;
    if (pane.polyLines) {
      pane.polyLines.visible = wireOn;
      pane.polyLines.material.color.set(color);
      pane.polyLines.material.opacity = L.mode === 'wire' ? 0.85 : opacity + 0.15;
    } else if (wireOn && !pane.wires.length) {
      for (const mesh of pane.meshes) {
        const w = new THREE.Mesh(mesh.geometry, new THREE.MeshBasicMaterial({ wireframe: true, transparent: true, toneMapped: false, depthWrite: false }));
        w.renderOrder = 2;
        w.matrixAutoUpdate = false;
        mesh.add(w);
        pane.wires.push(w);
      }
    }
    for (const w of pane.wires) {
      w.visible = wireOn && !pane.polyLines;
      w.material.color.set(color);
      w.material.opacity = opacity;
    }
    this.requestRender();
  }

  // ---------------------------------------------------------------- camera
  setInputCamera(fov) {
    this.inputFov = fov;
    const d = 0.5 / Math.tan(THREE.MathUtils.degToRad(fov) / 2);
    this.refCamera.fov = fov;
    this.refCamera.aspect = 1;
    this.refCamera.position.set(0, 0, d);
    this.refCamera.up.set(0, 1, 0);
    this.refCamera.lookAt(0, 0, 0);
    this.refCamera.updateProjectionMatrix();
    if (this.matched) this.matchInputCamera();
    this.requestRender();
  }

  matchInputCamera() {
    this.baseFov = this.inputFov;
    this.camera.position.copy(this.refCamera.position);
    this.camera.up.set(0, 1, 0);
    this.target.set(0, 0, 0);
    this.camera.lookAt(this.target);
    this.camera.updateProjectionMatrix();
    this._settle();
    this.matched = true;
    this.requestRender();
  }

  frameAll() {
    const box = new THREE.Box3();
    for (const p of this.panes) if (p.box) box.union(p.box);
    if (box.isEmpty()) { this.matchInputCamera(); return; }
    const c = box.getCenter(new THREE.Vector3());
    const r = box.getSize(new THREE.Vector3()).length() / 2;
    this.baseFov = 30;
    const dist = r / Math.sin(THREE.MathUtils.degToRad(this.baseFov) / 2) * 1.02;
    const dir = new THREE.Vector3(0.45, 0.25, 1).normalize();
    this.target.copy(c);
    this.camera.position.copy(c).addScaledVector(dir, dist);
    this.camera.lookAt(c);
    this._settle();
    this.matched = false;
    this.requestRender();
  }

  viewFrom(axis) {
    const r = this.camera.position.distanceTo(this.target);
    const dirs = { front: [0, 0, 1], back: [0, 0, -1], left: [-1, 0, 0], right: [1, 0, 0], top: [0, 1, 0.0001], bottom: [0, -1, 0.0001] };
    const d = new THREE.Vector3(...dirs[axis]).normalize();
    this.camera.position.copy(this.target).addScaledVector(d, r);
    this.camera.lookAt(this.target);
    this._settle();
    this.matched = false;
    this.requestRender();
  }

  // apply a camera preset and drop any leftover damping momentum from a previous drag
  _settle() {
    for (const c of this.controls.values()) { const d = c.enableDamping; c.enableDamping = false; c.update(); c.enableDamping = d; }
  }

  _syncAutoRotate() {
    let first = true;
    for (const c of this.controls.values()) { c.autoRotate = this.look.autoRotate && first; first = false; }
  }

  // ---------------------------------------------------------------- overlays
  texture(url) {
    if (!url) return null;
    if (this.texCache.has(url)) return this.texCache.get(url);
    const t = this.texLoader.load(url, () => this.requestRender());
    t.colorSpace = THREE.SRGBColorSpace;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    this.texCache.set(url, t);
    if (this.texCache.size > 40) {
      const [k, v] = this.texCache.entries().next().value;
      v.dispose(); this.texCache.delete(k);
    }
    return t;
  }

  setRef(ref) { this.ref = ref; this.requestRender(); }

  _drawOverlay(photo, mask, ov, maskAlpha, rect) {
    const r = this.renderer;
    if (rect) {
      const H = r.domElement.clientHeight || window.innerHeight;
      const side = Math.min(rect.width, rect.height);
      const x = rect.left + (rect.width - side) / 2, y = H - rect.bottom + (rect.height - side) / 2;
      r.setViewport(x, y, side, side);
      r.setScissor(x, y, side, side);
    }
    r.autoClear = false;
    try { this._drawOverlayInner(photo, mask, ov, maskAlpha); } finally { r.autoClear = true; }
  }

  _drawOverlayInner(photo, mask, ov, maskAlpha) {
    const r = this.renderer;
    if (photo && photo.image && (ov.mode === 'wipe' || ov.value > 0)) {
      this.ovMesh.material = this.photoMat;
      this.photoMat.uniforms.map.value = photo;
      this.photoMat.uniforms.opacity.value = ov.value;
      this.photoMat.uniforms.split.value = ov.split;
      this.photoMat.uniforms.mode.value = ov.mode === 'wipe' ? 1 : 0;
      r.render(this.ovScene, this.ovCam);
    }
    if (ov.outline && mask && mask.image) {
      this.ovMesh.material = this.outlineMat;
      this.outlineMat.uniforms.map.value = mask;
      this.outlineMat.uniforms.useAlpha.value = maskAlpha ? 1 : 0;
      const w = mask.image.width || 1024;
      this.outlineMat.uniforms.texel.value.set(1 / w, 1 / w);
      r.render(this.ovScene, this.ovCam);
    }
  }

  // ---------------------------------------------------------------- render
  _slotRect(el) {
    if (!el || !el.isConnected) return null;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return null;
    if (r.bottom < 0 || r.top > window.innerHeight || r.right < 0 || r.left > window.innerWidth) return null;
    return r;
  }

  _renderInto(rect, scene, camera, keyLight, baseFov) {
    const r = this.renderer;
    const H = r.domElement.clientHeight || window.innerHeight;
    const x = rect.left, y = H - rect.bottom;
    r.setViewport(x, y, rect.width, rect.height);
    r.setScissor(x, y, rect.width, rect.height);
    // the central square of every slot shows exactly the square (input-camera) view
    const aspect = rect.width / rect.height;
    const fov = aspect >= 1 ? baseFov : THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(baseFov) / 2) / aspect));
    if (camera.aspect !== aspect || camera.fov !== fov) { camera.aspect = aspect; camera.fov = fov; camera.updateProjectionMatrix(); }
    if (keyLight) {
      keyLight.position.copy(camera.position).add(new THREE.Vector3(0.6, 0.8, 0.2).multiplyScalar(camera.position.length() * 0.5));
      keyLight.target.position.copy(this.target);
      keyLight.target.updateMatrixWorld();
    }
    r.render(scene, camera);
  }

  _tick() {
    const dt = this.clock.getDelta();
    let moving = false;
    for (const c of this.controls.values()) if (c.update(dt)) moving = true;
    if (moving) this.dirty = true;
    if (this.look.autoRotate && this.panes.length) this.dirty = true;
    if (!this.dirty) return;
    this.dirty = false;
    const r = this.renderer;
    r.setScissorTest(false);
    r.setClearColor(0x000000, 0);
    r.clear();
    r.setScissorTest(true);
    for (const p of this.panes) {
      const rect = this._slotRect(p.slot);
      if (!rect || !p.model) continue;
      this._renderInto(rect, p.scene, this.camera, p.key, this.baseFov);
      if (this.photoInPanes && this.ref) this._drawOverlay(p.photo, p.mask, this.ref, p.maskAlpha, rect);
    }
    const ref = this.ref;
    if (ref && ref.pane && ref.pane.model) {
      const rect = this._slotRect(ref.slot);
      if (rect) {
        this._renderInto(rect, ref.pane.scene, this.refCamera, ref.pane.key, this.inputFov);
        this._drawOverlay(ref.photo, ref.mask, ref, ref.maskAlpha, rect);
      }
    }
  }

  analyzeTopology(pane) {
    return new Promise((resolve, reject) => {
      if (!pane.model) return reject(new Error('no model'));
      // merge all meshes into one position/index set
      let nv = 0, ni = 0;
      for (const m of pane.meshes) { const g = m.geometry; nv += g.attributes.position.count; ni += g.index ? g.index.count : g.attributes.position.count; }
      const pos = new Float32Array(nv * 3), idx = new Uint32Array(ni);
      let vo = 0, io = 0;
      const v = new THREE.Vector3();
      for (const m of pane.meshes) {
        const g = m.geometry, pa = g.attributes.position;
        m.updateMatrixWorld(true);
        for (let i = 0; i < pa.count; i++) { v.fromBufferAttribute(pa, i).applyMatrix4(m.matrixWorld); pos[3 * (vo + i)] = v.x; pos[3 * (vo + i) + 1] = v.y; pos[3 * (vo + i) + 2] = v.z; }
        if (g.index) { const a = g.index.array; for (let i = 0; i < a.length; i++) idx[io + i] = a[i] + vo; io += a.length; }
        else { for (let i = 0; i < pa.count; i++) idx[io + i] = vo + i; io += pa.count; }
        vo += pa.count;
      }
      const w = new Worker(URL.createObjectURL(new Blob([TOPO_SRC], { type: 'text/javascript' })));
      w.onmessage = (e) => { resolve(e.data); w.terminate(); };
      w.onerror = (e) => { reject(new Error(e.message || 'worker error')); w.terminate(); };
      const msg = { pos, idx };
      const tr = [pos.buffer, idx.buffer];
      if (pane.polyLines) {
        msg.polyEdges = pane.polyLines.userData.polyEdges.slice();
        msg.polyVerts = pane.polyLines.userData.polyVerts;
        tr.push(msg.polyEdges.buffer);
      }
      w.postMessage(msg, tr);
    });
  }
}

// Fallback when WebGL is unavailable: the 2D UI keeps working, 3D calls become no-ops.
export function nullStage(err) {
  const noop = () => {};
  return {
    failed: err, look: {}, inputFov: 30, matched: false, panes: [],
    setLook: noop, requestRender: noop, setRef: noop, setInputCamera(f) { this.inputFov = f; }, matchInputCamera: noop,
    frameAll: noop, viewFrom: noop, removePane: noop, texture: () => null,
    addPane(slot) { return { slot, model: null, stats: null, faces: 0 }; },
    loadGLB: () => Promise.reject(new Error('WebGL is not available in this browser')),
    loadOBJ: () => Promise.reject(new Error('WebGL is not available in this browser')),
    analyzeTopology: () => Promise.reject(new Error('WebGL is not available')),
  };
}
