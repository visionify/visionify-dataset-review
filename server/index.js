import express from "express";
import cors from "cors";
import fs from "fs/promises";
import fsSync from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createHash } from "crypto";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 3456;

const IMG_EXTS = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"]);
function stripImageExt(name) {
  const ext = path.extname(name).toLowerCase();
  return IMG_EXTS.has(ext) ? name.slice(0, -ext.length) : name;
}

app.use(cors());
app.use(express.json());

const CONFIG_PATH = path.join(__dirname, "..", "dataset-path.json");
const DEFAULT_DATASET_PATH = process.env.DATASET_PATH || "";

function getDatasetPath() {
  try {
    const raw = fsSync.readFileSync(CONFIG_PATH, "utf8");
    const data = JSON.parse(raw);
    return data.path || DEFAULT_DATASET_PATH;
  } catch {
    return DEFAULT_DATASET_PATH;
  }
}

function setDatasetPath(newPath) {
  return fs.writeFile(CONFIG_PATH, JSON.stringify({ path: newPath }, null, 2), "utf8");
}

function getReviewDir(datasetRoot) { return path.join(datasetRoot, "review"); }
function getTagsDir(datasetRoot) { return path.join(getReviewDir(datasetRoot), "tags"); }
function getMetadataPath(datasetRoot) { return path.join(getReviewDir(datasetRoot), "metadata.json"); }
function getReviewedPath(datasetRoot) { return path.join(getReviewDir(datasetRoot), "reviewed.json"); }
function getClassColorsPath(datasetRoot) { return path.join(getReviewDir(datasetRoot), "class-colors.json"); }

async function readReviewedSet(datasetRoot) {
  try {
    const content = await fs.readFile(getReviewedPath(datasetRoot), "utf8");
    const data = JSON.parse(content);
    return new Set(Array.isArray(data) ? data : data.reviewed || []);
  } catch {
    return new Set();
  }
}

async function ensureReviewDirs(datasetRoot) {
  await fs.mkdir(getReviewDir(datasetRoot), { recursive: true });
  await fs.mkdir(getTagsDir(datasetRoot), { recursive: true });
}

// ── Dataset layout detection ──────────────────────────────────────────────────
// Supports: standard (images/train, labels/train), Roboflow (train/images, train/labels),
// and datasets where the folder name is "valid" instead of "val".

const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]);

function normalizeClassNames(data) {
  const raw = data.names;
  if (Array.isArray(raw)) return Object.fromEntries(raw.map((name, i) => [i, String(name)]));
  if (raw && typeof raw === "object" && !Array.isArray(raw))
    return Object.fromEntries(Object.entries(raw).map(([k, v]) => [parseInt(k, 10), String(v)]));
  if (data.nc != null) return Object.fromEntries(Array.from({ length: parseInt(data.nc, 10) }, (_, i) => [i, `class_${i}`]));
  return {};
}

async function dirHasImages(dir) {
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    return entries.some(e => e.isFile() && IMAGE_EXTS.has(path.extname(e.name).toLowerCase()));
  } catch { return false; }
}

async function dirExists(dir) {
  try { return (await fs.stat(dir)).isDirectory(); } catch { return false; }
}

async function findDir(datasetRoot, candidates, checkImages) {
  for (const rel of candidates) {
    const full = path.join(datasetRoot, rel);
    if (checkImages ? await dirHasImages(full) : await dirExists(full)) return rel;
  }
  return null;
}

let _cfgCache = null;
let _cfgCachePath = null;

async function parseCvatNames(datasetRoot) {
  try {
    const content = await fs.readFile(path.join(datasetRoot, "obj.names"), "utf8");
    const lines = content.split("\n").map(l => l.trim()).filter(Boolean);
    return Object.fromEntries(lines.map((name, i) => [i, name]));
  } catch { return null; }
}

async function findCvatDataDir(datasetRoot) {
  try {
    const entries = await fs.readdir(datasetRoot, { withFileTypes: true });
    const candidate = entries.find(e => e.isDirectory() && /^obj_.*data$/i.test(e.name));
    if (candidate && await dirHasImages(path.join(datasetRoot, candidate.name))) return candidate.name;
  } catch {}
  return null;
}

const DEFAULT_NAMES = { 0: "hboard", 1: "pallet", 2: "vboard", 3: "person", 4: "forklift", 5: "wooden-log" };

async function resolveConfig(datasetRoot) {
  if (_cfgCache && _cfgCachePath === datasetRoot) return _cfgCache;

  // Check for CVAT export format (obj.names + obj_*_data folder)
  const cvatNames = await parseCvatNames(datasetRoot);
  const cvatDataDir = await findCvatDataDir(datasetRoot);

  if (cvatNames && cvatDataDir) {
    const cfg = {
      names: cvatNames,
      train: cvatDataDir,
      val: null,
      test: null,
      labelsDir: { train: cvatDataDir, val: null, test: null },
    };
    _cfgCache = cfg;
    _cfgCachePath = datasetRoot;
    return cfg;
  }

  // Standard YOLO format — find and parse YAML, fall back to hardcoded defaults
  let names = DEFAULT_NAMES;
  for (const name of ["data.yaml", "dataset.yaml", "dataset_weighted.yaml"]) {
    try {
      const yamlData = yaml.load(await fs.readFile(path.join(datasetRoot, name), "utf8"));
      if (yamlData) {
        const parsed = normalizeClassNames(yamlData);
        if (Object.keys(parsed).length > 0) { names = parsed; break; }
      }
    } catch {}
  }

  // Probe filesystem for image directories, default to images/train
  const trainImgs = await findDir(datasetRoot, ["images/train", "train/images", "train"], true) || "images/train";
  const valImgs   = await findDir(datasetRoot, ["images/val", "images/valid", "valid/images", "val/images", "val", "valid"], true);
  const testImgs  = await findDir(datasetRoot, ["images/test", "test/images", "test"], true);

  // Probe for label directories, default to labels/train
  const trainLabels = await findDir(datasetRoot, ["labels/train", "train/labels"], false) || "labels/train";
  const valLabels   = await findDir(datasetRoot, ["labels/val", "labels/valid", "valid/labels", "val/labels"], false);
  const testLabels  = await findDir(datasetRoot, ["labels/test", "test/labels"], false);

  const cfg = {
    names,
    train: trainImgs,
    val: valImgs,
    test: testImgs,
    labelsDir: { train: trainLabels, val: valLabels, test: testLabels },
  };
  _cfgCache = cfg;
  _cfgCachePath = datasetRoot;
  return cfg;
}

function invalidateConfigCache() {
  _cfgCache = null;
  _cfgCachePath = null;
  classIndexCache = null;
  tagIndexCache = null;
  tagIndexCachePath = null;
  classIndexCachePath = null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function listImagesInDir(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  return entries.filter(e => e.isFile() && IMAGE_EXTS.has(path.extname(e.name).toLowerCase())).map(e => e.name).sort();
}

function activeSplits(config) {
  return ["train", "val", "test"].filter(s => config[s]);
}

async function collectAllImages(datasetRoot, config) {
  const result = [];
  for (const split of activeSplits(config)) {
    try {
      const files = await listImagesInDir(path.join(datasetRoot, config[split]));
      for (const f of files)
        result.push({ split, name: f, relPath: path.join(config[split], f).replace(/\\/g, "/"), imageRel: path.join(config[split], f).replace(/\\/g, "/") });
    } catch {}
  }
  return result;
}

async function countImages(datasetRoot, config) {
  let total = 0;
  for (const split of activeSplits(config)) {
    try { total += (await listImagesInDir(path.join(datasetRoot, config[split]))).length; } catch {}
  }
  return total;
}

async function getImagesPaginated(datasetRoot, config, split, page, limit, reviewedFilter) {
  const splits = split && split !== "all" ? [split].filter(s => config[s]) : activeSplits(config);
  const all = [];
  for (const s of splits) {
    try {
      const files = await listImagesInDir(path.join(datasetRoot, config[s]));
      for (const f of files) {
        const base = path.basename(f, path.extname(f));
        all.push({ split: s, name: f, base, key: `${s}/${base}`, imageRel: path.join(config[s], f).replace(/\\/g, "/"), relPath: path.join(config[s], f).replace(/\\/g, "/") });
      }
    } catch {}
  }
  let filtered = all;
  if (reviewedFilter === "no" || reviewedFilter === "yes") {
    const reviewed = await readReviewedSet(datasetRoot);
    filtered = reviewedFilter === "no" ? all.filter(i => !reviewed.has(i.key)) : all.filter(i => reviewed.has(i.key));
  }
  const start = (page - 1) * limit;
  return { images: filtered.slice(start, start + limit).map(({ split: s, name, imageRel, relPath }) => ({ split: s, name, imageRel, relPath })), total: filtered.length };
}

// ── Auto-tags from filenames ─────────────────────────────────────────────────

function parseImageTags(filename) {
  const base = path.basename(filename, path.extname(filename));
  const result = { task: "no-task", year: null, monthYear: null, date: null, camera: null };

  const taskRe = /task[_\-]?(\d+)/i;
  const taskMatch = base.match(taskRe);
  if (taskMatch) result.task = `task-${taskMatch[1]}`;

  const dateRe = /(202\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])/;
  const dateMatch = base.match(dateRe);
  if (dateMatch) {
    result.year = dateMatch[1];
    result.monthYear = `${dateMatch[1]}-${dateMatch[2]}`;
    result.date = `${dateMatch[1]}-${dateMatch[2]}-${dateMatch[3]}`;
  }

  const NOISE = new Set(["task", "frame", "fr", "image", "img", "pallet", "pallets", "video", "svid", "rtsp", "hd", "cam", "mp", "min", "part", "new", "version"]);
  const tokens = base.split(/[_\-\s&()]+/);
  const cleaned = [];
  const seen = new Set();
  for (const tok of tokens) {
    let t = tok.replace(/[^a-zA-Z]/g, "").toLowerCase();
    if (!t || t.length < 3 || NOISE.has(t)) continue;
    if (/^[a-f]+$/.test(t) && t.length >= 6) continue;
    if (seen.has(t)) continue;
    seen.add(t);
    cleaned.push(t);
  }
  result.camera = cleaned.length ? cleaned.join("-") : null;

  return result;
}

let tagIndexCache = null;
let tagIndexCachePath = null;

async function buildTagIndex(datasetRoot, config) {
  if (tagIndexCache && tagIndexCachePath === datasetRoot) return tagIndexCache;
  const images = await collectAllImages(datasetRoot, config);
  const byImage = {};
  const tasks = {}, months = {}, cameras = {};

  for (const img of images) {
    const tags = parseImageTags(img.name);
    byImage[`${img.split}/${img.name}`] = tags;
    const entry = { split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath };

    if (!tasks[tags.task]) tasks[tags.task] = [];
    tasks[tags.task].push(entry);

    if (tags.monthYear) {
      if (!months[tags.monthYear]) months[tags.monthYear] = [];
      months[tags.monthYear].push(entry);
    }

    if (tags.camera) {
      if (!cameras[tags.camera]) cameras[tags.camera] = [];
      cameras[tags.camera].push(entry);
    }
  }

  const idx = { byImage, tasks, months, cameras };
  tagIndexCache = idx;
  tagIndexCachePath = datasetRoot;
  return idx;
}

let classIndexCache = null;
let classIndexCachePath = null;

async function buildClassIndex(datasetRoot, config) {
  if (classIndexCache && classIndexCachePath === datasetRoot) return classIndexCache;
  const idx = {};
  for (const split of activeSplits(config)) {
    const labelsDir = config.labelsDir?.[split];
    let files = [];
    try { files = await listImagesInDir(path.join(datasetRoot, config[split])); } catch {}
    for (const f of files) {
      const imageRel = path.join(config[split], f).replace(/\\/g, "/");
      const base = path.basename(f, path.extname(f));
      let ids = [];
      if (labelsDir) {
        try {
          const content = await fs.readFile(path.join(datasetRoot, labelsDir, base + ".txt"), "utf8");
          ids = [...new Set(content.split("\n").map(l => l.trim().split(/\s+/)[0]).filter(Boolean).map(c => parseInt(c, 10)))];
        } catch {}
      }
      idx[imageRel] = ids;
    }
  }
  classIndexCache = idx;
  classIndexCachePath = datasetRoot;
  return idx;
}

function splitFromImageRel(config, imageRel) {
  for (const s of activeSplits(config)) {
    if (imageRel.startsWith(config[s])) return s;
  }
  return "train";
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get("/api/config", async (_req, res) => {
  const p = getDatasetPath();
  res.json({ datasetPath: p, configured: !!p.trim() });
});

app.post("/api/config", async (req, res) => {
  try {
    const { path: newPath } = req.body;
    if (typeof newPath !== "string") return res.status(400).json({ error: "path required" });
    await setDatasetPath(newPath.trim());
    await ensureReviewDirs(newPath.trim());
    invalidateConfigCache();
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: String(e.message) });
  }
});

app.get("/api/dataset/summary", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({ configured: false, classes: [], config: null, totalImages: 0, reviewedCount: 0, splitCounts: {}, missingLabelsCount: 0, emptyLabelsCount: 0 });
  try {
    await ensureReviewDirs(datasetRoot);
    const config = await resolveConfig(datasetRoot);
    const reviewed = await readReviewedSet(datasetRoot);
    const classes = Object.entries(config.names).map(([id, name]) => ({ id: parseInt(id, 10), name: String(name) })).sort((a, b) => a.id - b.id);
    const splitCounts = {};
    let totalImages = 0;
    for (const s of activeSplits(config)) {
      try { const c = (await listImagesInDir(path.join(datasetRoot, config[s]))).length; splitCounts[s] = c; totalImages += c; } catch { splitCounts[s] = 0; }
    }
    let missingLabelsCount = 0, emptyLabelsCount = 0;
    const images = await collectAllImages(datasetRoot, config);
    for (const img of images) {
      const labelsRel = config.labelsDir?.[img.split];
      if (!labelsRel) { missingLabelsCount++; continue; }
      const base = path.basename(img.name, path.extname(img.name));
      try {
        const content = await fs.readFile(path.join(datasetRoot, labelsRel, base + ".txt"), "utf8");
        if (!content.split("\n").filter(l => l.trim()).length) emptyLabelsCount++;
      } catch { missingLabelsCount++; }
    }
    res.json({ configured: true, classes, config: { train: config.train, val: config.val, test: config.test, names: config.names }, totalImages, reviewedCount: reviewed.size, splitCounts, missingLabelsCount, emptyLabelsCount });
  } catch (e) {
    res.status(500).json({ error: String(e.message), configured: true });
  }
});

app.get("/api/auto-tags", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({ tasks: [], months: [], cameras: [] });
  try {
    const config = await resolveConfig(datasetRoot);
    const idx = await buildTagIndex(datasetRoot, config);
    const tasks = Object.entries(idx.tasks).map(([name, imgs]) => ({ name, count: imgs.length }))
      .sort((a, b) => {
        const na = parseInt(a.name.replace("task-", ""), 10);
        const nb = parseInt(b.name.replace("task-", ""), 10);
        if (!isNaN(na) && !isNaN(nb)) return na - nb;
        return a.name.localeCompare(b.name);
      });
    const months = Object.entries(idx.months).map(([name, imgs]) => ({ name, count: imgs.length })).sort((a, b) => a.name.localeCompare(b.name));
    const cameras = Object.entries(idx.cameras).map(([name, imgs]) => ({ name, count: imgs.length })).sort((a, b) => b.count - a.count).slice(0, 50);
    res.json({ tasks, months, cameras });
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
});

app.get("/api/images", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({ images: [], total: 0 });
  try {
    const config = await resolveConfig(datasetRoot);
    const split = req.query.split || "all";
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const limit = Math.min(10000, Math.max(1, parseInt(req.query.limit, 10) || 48));
    const tagType = req.query.tagType || "";
    const tagValue = req.query.tag || "";

    if (tagType && tagValue) {
      const idx = await buildTagIndex(datasetRoot, config);
      let pool = [];
      if (tagType === "task") pool = idx.tasks[tagValue] || [];
      else if (tagType === "month") pool = idx.months[tagValue] || [];
      else if (tagType === "camera") pool = idx.cameras[tagValue] || [];

      let filtered = pool;
      if (split && split !== "all") filtered = pool.filter(img => img.split === split);
      if (req.query.reviewed === "no" || req.query.reviewed === "yes") {
        const reviewed = await readReviewedSet(datasetRoot);
        filtered = filtered.filter(img => {
          const key = `${img.split}/${path.basename(img.name, path.extname(img.name))}`;
          return req.query.reviewed === "no" ? !reviewed.has(key) : reviewed.has(key);
        });
      }
      const start = (page - 1) * limit;
      res.json({ images: filtered.slice(start, start + limit), total: filtered.length });
    } else {
      const { images, total } = await getImagesPaginated(datasetRoot, config, split, page, limit, req.query.reviewed);
      res.json({ images, total });
    }
  } catch (e) {
    res.status(500).json({ images: [], total: 0 });
  }
});

app.get("/api/class/:id/images", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({ images: [], total: 0 });
  try {
    const config = await resolveConfig(datasetRoot);
    const classId = parseInt(req.params.id, 10);
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const limit = Math.min(10000, Math.max(1, parseInt(req.query.limit, 10) || 24));
    const sort = req.query.sort || "";
    const index = await buildClassIndex(datasetRoot, config);
    const rels = Object.entries(index).filter(([, ids]) => ids.includes(classId)).map(([rel]) => rel);
    const images = [];
    for (const rel of rels) {
      const name = path.basename(rel);
      const split = splitFromImageRel(config, rel);
      const item = { split, name, imageRel: rel, relPath: rel };
      if (sort === "area_asc" || sort === "area_desc") {
        const labelsDir = config.labelsDir?.[split];
        let minArea = null;
        if (labelsDir) {
          const base = path.basename(name, path.extname(name));
          try {
            const content = await fs.readFile(path.join(datasetRoot, labelsDir, base + ".txt"), "utf8");
            for (const line of content.split("\n")) {
              const parts = line.trim().split(/\s+/);
              if (parts.length < 5) continue;
              if (parseInt(parts[0], 10) !== classId) continue;
              const bw = parseFloat(parts[3]);
              const bh = parseFloat(parts[4]);
              const area = bw * bh;
              if (minArea === null || area < minArea) minArea = area;
            }
          } catch {}
        }
        item.bboxArea = minArea ?? 0;
      }
      images.push(item);
    }
    if (sort === "area_asc") images.sort((a, b) => a.bboxArea - b.bboxArea);
    else if (sort === "area_desc") images.sort((a, b) => b.bboxArea - a.bboxArea);
    const start = (page - 1) * limit;
    res.json({ images: images.slice(start, start + limit), total: images.length });
  } catch (e) {
    res.status(500).json({ images: [], total: 0 });
  }
});

app.get("/api/class/:id/samples", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({ samples: [] });
  try {
    const config = await resolveConfig(datasetRoot);
    const classId = parseInt(req.params.id, 10);
    const limit = Math.min(12, Math.max(1, parseInt(req.query.limit, 10) || 8));
    const index = await buildClassIndex(datasetRoot, config);
    const rels = Object.entries(index).filter(([, ids]) => ids.includes(classId)).map(([rel]) => rel).slice(0, limit);
    const samples = rels.map(rel => {
      const name = path.basename(rel);
      return { split: splitFromImageRel(config, rel), name, imageRel: rel, relPath: rel };
    });
    res.json({ samples });
  } catch (e) {
    res.json({ samples: [] });
  }
});

// Serve image files — resolves the correct directory from config
app.get("/api/images/:split/:name", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(404).end();
  const config = await resolveConfig(datasetRoot);
  const { split, name } = req.params;
  const imagesRel = config[split];
  if (!imagesRel) return res.status(404).end();
  const imagePath = path.join(datasetRoot, imagesRel, name);
  if (!imagePath.startsWith(path.resolve(datasetRoot))) return res.status(403).end();
  res.sendFile(imagePath, err => { if (err) res.status(404).end(); });
});

app.get("/dataset-asset/*", (req, res) => {
  const datasetRoot = getDatasetPath();
  const sub = req.params[0];
  if (!sub) return res.status(400).end();
  const full = path.join(datasetRoot, sub);
  if (!full.startsWith(path.resolve(datasetRoot))) return res.status(403).end();
  res.sendFile(full, err => { if (err) res.status(404).end(); });
});

// Annotations (labels) — resolves label directory from config
app.get("/api/annotations/:split/:base", async (req, res) => {
  const datasetRoot = getDatasetPath();
  const config = await resolveConfig(datasetRoot);
  const { split, base: baseParam } = req.params;
  const base = stripImageExt(baseParam);
  const labelsRel = config.labelsDir?.[split];
  if (!labelsRel) return res.json([]);
  const labelPath = path.join(datasetRoot, labelsRel, base + ".txt");
  try {
    const content = await fs.readFile(labelPath, "utf8");
    const boxes = content.split("\n").filter(l => l.trim()).map(line => {
      const p = line.trim().split(/\s+/).map(Number);
      return { classId: p[0], x: p[1], y: p[2], w: p[3], h: p[4] };
    });
    res.json(boxes);
  } catch { res.json([]); }
});

app.put("/api/annotations/:split/:base", async (req, res) => {
  const datasetRoot = getDatasetPath();
  const config = await resolveConfig(datasetRoot);
  const { split, base: baseParam } = req.params;
  const base = stripImageExt(baseParam);
  const labelsRel = config.labelsDir?.[split];
  if (!labelsRel) return res.status(400).json({ error: "no labels dir for split " + split });
  const labelsDir = path.join(datasetRoot, labelsRel);
  await fs.mkdir(labelsDir, { recursive: true });
  const labelPath = path.join(labelsDir, base + ".txt");
  const boxes = req.body;
  if (!Array.isArray(boxes)) return res.status(400).json({ error: "array of boxes required" });
  await fs.writeFile(labelPath, boxes.map(b => `${b.classId} ${b.x} ${b.y} ${b.w} ${b.h}`).join("\n"), "utf8");
  res.json({ ok: true });
  try {
    const metaPath = getMetadataPath(datasetRoot);
    let meta = {};
    try { meta = JSON.parse(await fs.readFile(metaPath, "utf8")); } catch {}
    meta.lastSavedAnnotation = { split, base, at: new Date().toISOString() };
    await fs.writeFile(metaPath, JSON.stringify(meta, null, 2), "utf8");
  } catch {}
});

// Tags
function tagFilePath(datasetRoot, base) {
  return path.join(getTagsDir(datasetRoot), base + ".json");
}

app.get("/api/tags/:split/:base", async (req, res) => {
  const datasetRoot = getDatasetPath();
  const base = stripImageExt(req.params.base);
  try { res.json(JSON.parse(await fs.readFile(tagFilePath(datasetRoot, base), "utf8"))); } catch { res.json({}); }
});

app.put("/api/tags/:split/:base", async (req, res) => {
  const datasetRoot = getDatasetPath();
  await ensureReviewDirs(datasetRoot);
  const base = stripImageExt(req.params.base);
  if (typeof req.body !== "object" || req.body === null) return res.status(400).json({ error: "object required" });
  await fs.writeFile(tagFilePath(datasetRoot, base), JSON.stringify(req.body, null, 2), "utf8");
  res.json({ ok: true });
});

// Threshold for "small bbox": area as fraction of image (0.1% = 0.001). YOLO format uses normalized w,h so area = w*h.
const SMALL_BBOX_AREA_THRESHOLD = 0.001;

// Validation checks
app.get("/api/validation", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({ checks: [] });
  try {
    const config = await resolveConfig(datasetRoot);
    const images = await collectAllImages(datasetRoot, config);
    const missingLabels = [], emptyLabels = [], duplicateLabels = [], smallBboxes = [], classCounts = {};
    let totalDupLines = 0;
    let totalSmallBboxes = 0;
    for (const img of images) {
      const labelsRel = config.labelsDir?.[img.split];
      if (!labelsRel) { missingLabels.push({ split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath }); continue; }
      const base = path.basename(img.name, path.extname(img.name));
      try {
        const content = await fs.readFile(path.join(datasetRoot, labelsRel, base + ".txt"), "utf8");
        const lines = content.split("\n").filter(l => l.trim());
        if (!lines.length) emptyLabels.push({ split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath });
        const unique = new Set(lines.map(l => l.trim()));
        if (unique.size < lines.length) {
          const dupCount = lines.length - unique.size;
          totalDupLines += dupCount;
          duplicateLabels.push({ split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath, dupCount });
        }
        let smallCount = 0;
        for (const line of lines) {
          const p = line.trim().split(/\s+/).map(Number);
          if (p.length >= 5) {
            const area = p[3] * p[4];
            if (area < SMALL_BBOX_AREA_THRESHOLD) smallCount++;
            const c = parseInt(p[0], 10);
            classCounts[c] = (classCounts[c] || 0) + 1;
          }
        }
        if (smallCount > 0) {
          totalSmallBboxes += smallCount;
          smallBboxes.push({ split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath, smallCount });
        }
      } catch { missingLabels.push({ split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath }); }
    }
    // Detect duplicate images by MD5 hash
    const hashMap = {};
    for (const img of images) {
      const imgPath = path.join(datasetRoot, config[img.split], img.name);
      try {
        const buf = await fs.readFile(imgPath);
        const hash = createHash("md5").update(buf).digest("hex");
        if (!hashMap[hash]) hashMap[hash] = [];
        hashMap[hash].push({ split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath });
      } catch {}
    }
    const duplicateImages = [];
    let totalDupImages = 0;
    for (const [hash, items] of Object.entries(hashMap)) {
      if (items.length > 1) {
        totalDupImages += items.length - 1;
        for (const item of items.slice(1)) {
          duplicateImages.push({ ...item, hash, originalName: items[0].name, originalSplit: items[0].split });
        }
      }
    }

    res.json({ checks: [
      { id: "missing_labels", name: "Images without label file", count: missingLabels.length, severity: missingLabels.length ? "warning" : "ok", detail: missingLabels },
      { id: "empty_labels", name: "Label files with no objects", count: emptyLabels.length, severity: "info", detail: emptyLabels },
      { id: "duplicate_labels", name: "Images with duplicate labels", count: duplicateLabels.length, severity: duplicateLabels.length ? "warning" : "ok", detail: duplicateLabels, extra: { totalDupLines } },
      { id: "duplicate_images", name: "Duplicate images (by MD5)", count: duplicateImages.length, severity: duplicateImages.length ? "warning" : "ok", detail: duplicateImages, extra: { totalDupImages } },
      { id: "small_bboxes", name: "Bboxes smaller than 0.1% of image", count: totalSmallBboxes, severity: totalSmallBboxes ? "warning" : "ok", detail: smallBboxes, extra: { filesAffected: smallBboxes.length } },
      { id: "class_balance", name: "Class distribution", count: Object.keys(classCounts).length, severity: "ok", detail: classCounts },
    ]});
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
});

// Remove duplicate lines from all label files
app.post("/api/validation/fix-duplicates", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  try {
    const config = await resolveConfig(datasetRoot);
    const images = await collectAllImages(datasetRoot, config);
    let filesFixed = 0, linesRemoved = 0;
    for (const img of images) {
      const labelsRel = config.labelsDir?.[img.split];
      if (!labelsRel) continue;
      const base = path.basename(img.name, path.extname(img.name));
      const labelPath = path.join(datasetRoot, labelsRel, base + ".txt");
      try {
        const content = await fs.readFile(labelPath, "utf8");
        const lines = content.split("\n").filter(l => l.trim());
        const unique = [...new Set(lines.map(l => l.trim()))];
        if (unique.length < lines.length) {
          linesRemoved += lines.length - unique.length;
          filesFixed++;
          await fs.writeFile(labelPath, unique.join("\n") + "\n", "utf8");
        }
      } catch {}
    }
    invalidateConfigCache();
    res.json({ ok: true, filesFixed, linesRemoved });
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
});

// Remove duplicate images (keeps first, deletes rest + their labels)
app.post("/api/validation/delete-duplicate-images", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  try {
    const config = await resolveConfig(datasetRoot);
    const images = await collectAllImages(datasetRoot, config);
    const hashMap = {};
    for (const img of images) {
      const imgPath = path.join(datasetRoot, config[img.split], img.name);
      try {
        const buf = await fs.readFile(imgPath);
        const hash = createHash("md5").update(buf).digest("hex");
        if (!hashMap[hash]) hashMap[hash] = [];
        hashMap[hash].push(img);
      } catch {}
    }
    let deleted = 0;
    for (const items of Object.values(hashMap)) {
      if (items.length <= 1) continue;
      for (const img of items.slice(1)) {
        await tryDeleteImage(datasetRoot, config, img);
        const labelsRel = config.labelsDir?.[img.split];
        if (labelsRel) {
          const base = path.basename(img.name, path.extname(img.name));
          try { await fs.unlink(path.join(datasetRoot, labelsRel, base + ".txt")); } catch {}
        }
        deleted++;
      }
    }
    invalidateConfigCache();
    res.json({ ok: true, deleted });
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
});

// Bulk-delete images that have no label file
app.post("/api/validation/delete-missing-labels", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  try {
    const config = await resolveConfig(datasetRoot);
    const images = await collectAllImages(datasetRoot, config);
    let deleted = 0;
    for (const img of images) {
      const labelsRel = config.labelsDir?.[img.split];
      if (!labelsRel) { await tryDeleteImage(datasetRoot, config, img); deleted++; continue; }
      const base = path.basename(img.name, path.extname(img.name));
      const labelPath = path.join(datasetRoot, labelsRel, base + ".txt");
      try { await fs.access(labelPath); } catch { await tryDeleteImage(datasetRoot, config, img); deleted++; }
    }
    invalidateConfigCache();
    res.json({ ok: true, deleted });
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
});

// Bulk-remove bbox lines that are smaller than 0.1% of image area (normalized w*h < 0.001)
app.post("/api/validation/delete-small-bboxes", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  try {
    const config = await resolveConfig(datasetRoot);
    const images = await collectAllImages(datasetRoot, config);
    let removed = 0;
    let filesUpdated = 0;
    for (const img of images) {
      const labelsRel = config.labelsDir?.[img.split];
      if (!labelsRel) continue;
      const base = path.basename(img.name, path.extname(img.name));
      const labelPath = path.join(datasetRoot, labelsRel, base + ".txt");
      try {
        const content = await fs.readFile(labelPath, "utf8");
        const lines = content.split("\n").filter(l => l.trim());
        const kept = [];
        let fileRemoved = 0;
        for (const line of lines) {
          const p = line.trim().split(/\s+/).map(Number);
          if (p.length < 5) { kept.push(line); continue; }
          const area = p[3] * p[4];
          if (area >= SMALL_BBOX_AREA_THRESHOLD) kept.push(line);
          else fileRemoved++;
        }
        if (fileRemoved > 0) {
          await fs.writeFile(labelPath, kept.join("\n") + (kept.length ? "\n" : ""), "utf8");
          removed += fileRemoved;
          filesUpdated++;
        }
      } catch {}
    }
    invalidateConfigCache();
    res.json({ ok: true, removed, filesUpdated });
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
});

async function tryDeleteImage(datasetRoot, config, img) {
  const imagesRel = config[img.split];
  if (!imagesRel) return;
  try { await fs.unlink(path.join(datasetRoot, imagesRel, img.name)); } catch {}
  const key = `${img.split}/${path.basename(img.name, path.extname(img.name))}`;
  try {
    const d = JSON.parse(await fs.readFile(getReviewedPath(datasetRoot), "utf8"));
    if (Array.isArray(d.reviewed)) { d.reviewed = d.reviewed.filter(k => k !== key); await fs.writeFile(getReviewedPath(datasetRoot), JSON.stringify(d, null, 2), "utf8"); }
  } catch {}
}

// Metadata
app.patch("/api/metadata", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  await ensureReviewDirs(datasetRoot);
  const metaPath = getMetadataPath(datasetRoot);
  let meta = {};
  try { meta = JSON.parse(await fs.readFile(metaPath, "utf8")); } catch {}
  Object.assign(meta, req.body);
  await fs.writeFile(metaPath, JSON.stringify(meta, null, 2), "utf8");
  res.json(meta);
});

// Reviewed
app.get("/api/reviewed", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({ reviewed: [] });
  try { res.json({ reviewed: [...await readReviewedSet(datasetRoot)] }); } catch { res.json({ reviewed: [] }); }
});

app.patch("/api/reviewed", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  await ensureReviewDirs(datasetRoot);
  const { split, base, reviewed: mark } = req.body;
  if (!split || base == null) return res.status(400).json({ error: "split and base required" });
  const key = `${split}/${typeof base === "string" ? base.replace(/\.[^.]+$/, "") : base}`;
  const p = getReviewedPath(datasetRoot);
  let data = { reviewed: [] };
  try { data = JSON.parse(await fs.readFile(p, "utf8")); if (!Array.isArray(data.reviewed)) data.reviewed = []; } catch {}
  const set = new Set(data.reviewed);
  if (mark) set.add(key); else set.delete(key);
  data.reviewed = [...set];
  await fs.writeFile(p, JSON.stringify(data, null, 2), "utf8");
  res.json({ reviewed: data.reviewed });
});

// Class colors
app.get("/api/class-colors", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({});
  try { res.json(JSON.parse(await fs.readFile(getClassColorsPath(datasetRoot), "utf8"))); } catch { res.json({}); }
});

app.put("/api/class-colors", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  await ensureReviewDirs(datasetRoot);
  if (typeof req.body !== "object" || req.body === null) return res.status(400).json({ error: "object required" });
  await fs.writeFile(getClassColorsPath(datasetRoot), JSON.stringify(req.body, null, 2), "utf8");
  res.json(req.body);
});

// Delete image + label
app.delete("/api/images/:split/:name", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  const config = await resolveConfig(datasetRoot);
  const { split, name } = req.params;
  const imagesRel = config[split];
  if (!imagesRel) return res.status(404).json({ error: "split not found" });
  const imagePath = path.join(datasetRoot, imagesRel, name);
  if (!imagePath.startsWith(path.resolve(datasetRoot))) return res.status(403).json({ error: "forbidden" });
  const base = path.basename(name, path.extname(name));
  try { await fs.unlink(imagePath); } catch (e) { if (e.code !== "ENOENT") return res.status(500).json({ error: String(e.message) }); }
  const labelsRel = config.labelsDir?.[split];
  if (labelsRel) { try { await fs.unlink(path.join(datasetRoot, labelsRel, base + ".txt")); } catch {} }
  const key = `${split}/${base}`;
  try {
    const d = JSON.parse(await fs.readFile(getReviewedPath(datasetRoot), "utf8"));
    if (Array.isArray(d.reviewed)) { d.reviewed = d.reviewed.filter(k => k !== key); await fs.writeFile(getReviewedPath(datasetRoot), JSON.stringify(d, null, 2), "utf8"); }
  } catch {}
  invalidateConfigCache();
  res.json({ ok: true });
});

// ---------------------------------------------------------------------------
// Inference proxy — forwards to the Python inference server
// ---------------------------------------------------------------------------
const INFERENCE_URL = process.env.INFERENCE_URL || "http://localhost:3457";

async function inferenceProxy(method, inferPath, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(`${INFERENCE_URL}${inferPath}`, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || JSON.stringify(data));
  return data;
}

app.get("/api/inference/health", async (_req, res) => {
  try { res.json(await inferenceProxy("GET", "/health")); }
  catch { res.json({ status: "offline", model_loaded: false, model_path: null }); }
});

app.post("/api/inference/load", async (req, res) => {
  try { res.json(await inferenceProxy("POST", "/load", req.body)); }
  catch (e) { res.status(500).json({ error: String(e.message) }); }
});

app.post("/api/inference/predict", async (req, res) => {
  const datasetRoot = getDatasetPath();
  const { split, name, confidence, iou } = req.body;
  if (!datasetRoot || !split || !name) return res.status(400).json({ error: "split and name required" });
  try {
    const config = await resolveConfig(datasetRoot);
    const imagesRel = config[split];
    if (!imagesRel) return res.status(404).json({ error: "split not found" });
    const imagePath = path.join(datasetRoot, imagesRel, name);
    const result = await inferenceProxy("POST", "/predict", { image_path: imagePath, confidence: confidence ?? 0.25, iou: iou ?? 0.45 });
    res.json(result);
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
});

app.post("/api/inference/unload", async (_req, res) => {
  try { res.json(await inferenceProxy("POST", "/unload")); }
  catch (e) { res.status(500).json({ error: String(e.message) }); }
});

app.listen(PORT, () => {
  console.log(`Dataset API at http://localhost:${PORT}`);
});
