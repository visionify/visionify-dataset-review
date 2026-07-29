import express from "express";
import cors from "cors";
import fs from "fs/promises";
import fsSync from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createHash } from "crypto";
import yaml from "js-yaml";
import sharp from "sharp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3456;

const IMG_EXTS = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"]);
function stripImageExt(name) {
  const ext = path.extname(name).toLowerCase();
  return IMG_EXTS.has(ext) ? name.slice(0, -ext.length) : name;
}

app.use(cors());
app.use(express.json());

// DATASET_CONFIG_PATH lets a throwaway server instance point at its own dataset
// without disturbing the running app's dataset-path.json.
const CONFIG_PATH = process.env.DATASET_CONFIG_PATH || path.join(__dirname, "..", "dataset-path.json");
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

// ── Classification Detection ──────────────────────────────────────────────────

const CLS_IGNORE_DIRS = new Set([
  "review", ".git", "node_modules", ".venv", "__pycache__", ".idea", ".vscode",
  "images", "labels", ".ipynb_checkpoints", "runs", "weights", "docs",
]);

async function tryClassificationDetection(datasetRoot) {
  // Case 1: YOLO classification — train/ or training/ with subdirectories containing images
  for (const trainName of ["train", "training"]) {
    const trainDir = path.join(datasetRoot, trainName);
    if (!await dirExists(trainDir)) continue;
    try {
      const entries = await fs.readdir(trainDir, { withFileTypes: true });
      const subdirs = entries.filter(e => e.isDirectory());
      if (subdirs.length === 0) continue;
      // Check that at least one subdir has images (not standard YOLO structure like images/ or labels/)
      let hasClassFolders = false;
      for (const sd of subdirs) {
        if (sd.name === "images" || sd.name === "labels") continue;
        if (await dirHasImages(path.join(trainDir, sd.name))) { hasClassFolders = true; break; }
      }
      if (!hasClassFolders) continue;
      // Collect class folders from train split
      const classFolders = [];
      for (const sd of subdirs) {
        if (sd.name === "images" || sd.name === "labels") continue;
        if (await dirHasImages(path.join(trainDir, sd.name))) classFolders.push(sd.name);
      }
      classFolders.sort();
      const names = Object.fromEntries(classFolders.map((name, i) => [i, name]));
      // Detect val and test splits
      let valSplit = null;
      for (const vn of ["val", "valid"]) {
        if (await dirExists(path.join(datasetRoot, vn))) { valSplit = vn; break; }
      }
      let testSplit = null;
      if (await dirExists(path.join(datasetRoot, "test"))) testSplit = "test";
      return {
        type: "classification",
        names,
        classFolders,
        train: trainName,
        val: valSplit,
        test: testSplit,
        labelsDir: { train: null, val: null, test: null },
      };
    } catch {}
  }

  // Case 2: Simple classification — root has subdirectories (excluding ignored) with images
  try {
    const entries = await fs.readdir(datasetRoot, { withFileTypes: true });
    const subdirs = entries.filter(e => e.isDirectory() && !CLS_IGNORE_DIRS.has(e.name));
    const classFolders = [];
    for (const sd of subdirs) {
      if (await dirHasImages(path.join(datasetRoot, sd.name))) classFolders.push(sd.name);
    }
    if (classFolders.length >= 1) {
      classFolders.sort();
      const names = Object.fromEntries(classFolders.map((name, i) => [i, name]));
      return {
        type: "classification",
        names,
        classFolders,
        train: null,
        val: null,
        test: null,
        labelsDir: { train: null, val: null, test: null },
      };
    }
  } catch {}

  return null;
}

async function resolveConfig(datasetRoot) {
  if (_cfgCache && _cfgCachePath === datasetRoot) return _cfgCache;

  // Check for CVAT export format (obj.names + obj_*_data folder)
  const cvatNames = await parseCvatNames(datasetRoot);
  const cvatDataDir = await findCvatDataDir(datasetRoot);

  if (cvatNames && cvatDataDir) {
    const cfg = {
      type: "detection",
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

  // Standard YOLO format — find and parse YAML
  let yamlData = null;
  for (const name of ["data.yaml", "dataset.yaml", "dataset_weighted.yaml"]) {
    try {
      yamlData = yaml.load(await fs.readFile(path.join(datasetRoot, name), "utf8"));
      break;
    } catch {}
  }
  const names = yamlData ? normalizeClassNames(yamlData) : {};

  // Probe filesystem for image directories
  const trainImgs = await findDir(datasetRoot, ["images/train", "train/images", "train"], true);
  const valImgs   = await findDir(datasetRoot, ["images/val", "images/valid", "valid/images", "val/images", "val", "valid"], true);
  const testImgs  = await findDir(datasetRoot, ["images/test", "test/images", "test"], true);

  // Probe for label directories
  const trainLabels = await findDir(datasetRoot, ["labels/train", "train/labels"], false);
  const valLabels   = await findDir(datasetRoot, ["labels/val", "labels/valid", "valid/labels", "val/labels"], false);
  const testLabels  = await findDir(datasetRoot, ["labels/test", "test/labels"], false);

  if (trainImgs || valImgs || testImgs) {
    const cfg = {
      type: "detection",
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

  // Try classification detection
  const clsCfg = await tryClassificationDetection(datasetRoot);
  if (clsCfg) {
    _cfgCache = clsCfg;
    _cfgCachePath = datasetRoot;
    return clsCfg;
  }

  // Fallback: detection config
  const cfg = {
    type: "detection",
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

// ── Classification Helpers ────────────────────────────────────────────────────

function classificationSplits(config) {
  const splits = activeSplits(config);
  return splits.length > 0 ? splits : [null];
}

async function collectClassificationImages(datasetRoot, config, opts = {}) {
  const { className, split, sort, reviewed: reviewedFilter } = opts;
  const splits = classificationSplits(config);
  const items = [];

  for (const s of splits) {
    const folders = className ? [className] : config.classFolders;
    for (const cls of folders) {
      let dir;
      if (s) {
        dir = path.join(datasetRoot, config[s], cls);
      } else {
        dir = path.join(datasetRoot, cls);
      }
      try {
        const files = await listImagesInDir(dir);
        for (const f of files) {
          let imageRel;
          if (s) {
            imageRel = `${config[s]}/${cls}/${f}`;
          } else {
            imageRel = `${cls}/${f}`;
          }
          items.push({
            split: s || "all",
            className: cls,
            name: f,
            imageRel,
            relPath: imageRel,
          });
        }
      } catch {}
    }
  }

  // Filter by split
  let filtered = items;
  if (split && split !== "all") {
    filtered = filtered.filter(img => img.split === split);
  }

  // Filter by reviewed
  if (reviewedFilter === "no" || reviewedFilter === "yes") {
    const reviewedSet = await readReviewedSet(datasetRoot);
    filtered = filtered.filter(img => {
      const key = classificationReviewKey(img.imageRel);
      return reviewedFilter === "no" ? !reviewedSet.has(key) : reviewedSet.has(key);
    });
  }

  // Sort by size if requested
  if (sort === "size_asc" || sort === "size_desc") {
    for (const item of filtered) {
      try {
        const stat = await fs.stat(path.join(datasetRoot, item.imageRel));
        item.fileSize = stat.size;
      } catch {
        item.fileSize = 0;
      }
    }
    if (sort === "size_asc") filtered.sort((a, b) => a.fileSize - b.fileSize);
    else filtered.sort((a, b) => b.fileSize - a.fileSize);
  }

  return filtered;
}

function classificationReviewKey(imageRel) {
  const ext = path.extname(imageRel);
  return ext ? imageRel.slice(0, -ext.length) : imageRel;
}

async function classificationImagePath(datasetRoot, config, imageRel) {
  return path.join(datasetRoot, imageRel);
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

    if (config.type === "classification") {
      const splitCounts = {};
      let totalImages = 0;
      const classCounts = {};
      const splits = classificationSplits(config);
      for (const s of splits) {
        let splitTotal = 0;
        for (const cls of config.classFolders) {
          let dir;
          if (s) {
            dir = path.join(datasetRoot, config[s], cls);
          } else {
            dir = path.join(datasetRoot, cls);
          }
          try {
            const files = await listImagesInDir(dir);
            const count = files.length;
            splitTotal += count;
            classCounts[cls] = (classCounts[cls] || 0) + count;
          } catch {}
        }
        splitCounts[s || "all"] = splitTotal;
        totalImages += splitTotal;
      }

      // Count reviewed
      let reviewedCount = 0;
      const allImages = await collectClassificationImages(datasetRoot, config);
      for (const img of allImages) {
        const key = classificationReviewKey(img.imageRel);
        if (reviewed.has(key)) reviewedCount++;
      }

      res.json({
        configured: true,
        type: "classification",
        classes,
        config: { train: config.train, val: config.val, test: config.test, names: config.names, classFolders: config.classFolders },
        totalImages,
        reviewedCount,
        splitCounts,
        classCounts,
        missingLabelsCount: 0,
        emptyLabelsCount: 0,
      });
      return;
    }

    // Detection
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
    res.json({ configured: true, type: "detection", classes, config: { train: config.train, val: config.val, test: config.test, names: config.names }, totalImages, reviewedCount: reviewed.size, splitCounts, missingLabelsCount, emptyLabelsCount });
  } catch (e) {
    res.status(500).json({ error: String(e.message), configured: true });
  }
});

app.get("/api/auto-tags", async (_req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.json({ tasks: [], months: [], cameras: [] });
  try {
    const config = await resolveConfig(datasetRoot);

    if (config.type === "classification") {
      const allImages = await collectClassificationImages(datasetRoot, config);
      const byImage = {};
      const tasks = {}, months = {}, cameras = {};
      for (const img of allImages) {
        const tags = parseImageTags(img.name);
        byImage[img.imageRel] = tags;
        const entry = { split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath, className: img.className };
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
      const taskList = Object.entries(tasks).map(([name, imgs]) => ({ name, count: imgs.length }))
        .sort((a, b) => {
          const na = parseInt(a.name.replace("task-", ""), 10);
          const nb = parseInt(b.name.replace("task-", ""), 10);
          if (!isNaN(na) && !isNaN(nb)) return na - nb;
          return a.name.localeCompare(b.name);
        });
      const monthList = Object.entries(months).map(([name, imgs]) => ({ name, count: imgs.length })).sort((a, b) => a.name.localeCompare(b.name));
      const cameraList = Object.entries(cameras).map(([name, imgs]) => ({ name, count: imgs.length })).sort((a, b) => b.count - a.count).slice(0, 50);
      res.json({ tasks: taskList, months: monthList, cameras: cameraList });
      return;
    }

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

    // Classification branch
    if (config.type === "classification") {
      const split = req.query.split || "all";
      const page = Math.max(1, parseInt(req.query.page, 10) || 1);
      const limit = Math.min(10000, Math.max(1, parseInt(req.query.limit, 10) || 48));
      const reviewedFilter = req.query.reviewed;
      const className = req.query.className || undefined;
      const sort = req.query.sort || "";

      const items = await collectClassificationImages(datasetRoot, config, {
        className,
        split,
        sort,
        reviewed: reviewedFilter,
      });

      const start = (page - 1) * limit;
      const paged = items.slice(start, start + limit);
      res.json({ images: paged, total: items.length });
      return;
    }

    // Detection branch
    const split = req.query.split || "all";
    const page = Math.max(1, parseInt(req.query.page, 10) || 1);
    const limit = Math.min(10000, Math.max(1, parseInt(req.query.limit, 10) || 48));
    const tagType = req.query.tagType || "";
    const tagValue = req.query.tag || "";
    const classIdParam = req.query.classId;
    const classId = classIdParam != null ? parseInt(classIdParam, 10) : null;
    const sort = req.query.sort || "";

    let pool = null;

    if (tagType && tagValue) {
      const idx = await buildTagIndex(datasetRoot, config);
      if (tagType === "task") pool = idx.tasks[tagValue] || [];
      else if (tagType === "month") pool = idx.months[tagValue] || [];
      else if (tagType === "camera") pool = idx.cameras[tagValue] || [];
      else pool = [];
    }

    if (pool !== null) {
      let filtered = pool;
      if (split && split !== "all") filtered = filtered.filter(img => img.split === split);
      if (req.query.reviewed === "no" || req.query.reviewed === "yes") {
        const reviewed = await readReviewedSet(datasetRoot);
        filtered = filtered.filter(img => {
          const key = `${img.split}/${path.basename(img.name, path.extname(img.name))}`;
          return req.query.reviewed === "no" ? !reviewed.has(key) : reviewed.has(key);
        });
      }

      if (classId != null && !isNaN(classId)) {
        const classIdx = await buildClassIndex(datasetRoot, config);
        filtered = filtered.filter(img => {
          const rel = img.imageRel || img.relPath;
          return rel && classIdx[rel] && classIdx[rel].includes(classId);
        });
      }

      // Clone so we never mutate cached tagIndex objects
      filtered = filtered.map(img => ({ ...img }));

      if (classId != null && !isNaN(classId) && (sort === "area_asc" || sort === "area_desc")) {
        for (const item of filtered) {
          const imgSplit = item.split;
          const labelsDir = config.labelsDir?.[imgSplit];
          let minArea = null;
          if (labelsDir) {
            const base = path.basename(item.name, path.extname(item.name));
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
        if (sort === "area_asc") filtered.sort((a, b) => a.bboxArea - b.bboxArea);
        else filtered.sort((a, b) => b.bboxArea - a.bboxArea);
      }

      const start = (page - 1) * limit;
      res.json({ images: filtered.slice(start, start + limit), total: filtered.length });
    } else {
      let result = await getImagesPaginated(datasetRoot, config, split, page, limit, req.query.reviewed);

      if (classId != null && !isNaN(classId)) {
        const classIdx = await buildClassIndex(datasetRoot, config);
        const allSplits = split && split !== "all" ? [split].filter(s => config[s]) : activeSplits(config);
        let allFiltered = [];
        for (const s of allSplits) {
          try {
            const files = await listImagesInDir(path.join(datasetRoot, config[s]));
            for (const f of files) {
              const imageRel = path.join(config[s], f).replace(/\\/g, "/");
              if (classIdx[imageRel] && classIdx[imageRel].includes(classId)) {
                const base = path.basename(f, path.extname(f));
                allFiltered.push({ split: s, name: f, base, key: `${s}/${base}`, imageRel, relPath: imageRel });
              }
            }
          } catch {}
        }

        if (req.query.reviewed === "no" || req.query.reviewed === "yes") {
          const reviewed = await readReviewedSet(datasetRoot);
          allFiltered = req.query.reviewed === "no"
            ? allFiltered.filter(i => !reviewed.has(i.key))
            : allFiltered.filter(i => reviewed.has(i.key));
        }

        if (sort === "area_asc" || sort === "area_desc") {
          for (const item of allFiltered) {
            const labelsDir = config.labelsDir?.[item.split];
            let minArea = null;
            if (labelsDir) {
              try {
                const content = await fs.readFile(path.join(datasetRoot, labelsDir, item.base + ".txt"), "utf8");
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
          if (sort === "area_asc") allFiltered.sort((a, b) => a.bboxArea - b.bboxArea);
          else allFiltered.sort((a, b) => b.bboxArea - a.bboxArea);
        }

        const start = (page - 1) * limit;
        result = {
          images: allFiltered.slice(start, start + limit).map(({ split: s, name, imageRel, relPath, bboxArea }) => {
            const item = { split: s, name, imageRel, relPath };
            if (bboxArea != null) item.bboxArea = bboxArea;
            return item;
          }),
          total: allFiltered.length,
        };
      }

      res.json(result);
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

    // Classification branch
    if (config.type === "classification") {
      const className = config.names[classId];
      if (!className) return res.json({ images: [], total: 0 });
      const items = await collectClassificationImages(datasetRoot, config, { className, sort });
      if (sort === "size_asc" || sort === "size_desc") {
        // Already sorted by collectClassificationImages
      }
      const start = (page - 1) * limit;
      res.json({ images: items.slice(start, start + limit), total: items.length });
      return;
    }

    // Detection branch
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

    // Classification branch
    if (config.type === "classification") {
      const className = config.names[classId];
      if (!className) return res.json({ samples: [] });
      const items = await collectClassificationImages(datasetRoot, config, { className });
      const samples = items.slice(0, limit).map(img => ({
        split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath, className: img.className,
      }));
      res.json({ samples });
      return;
    }

    // Detection branch
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

  // Classification guard
  if (config.type === "classification") return res.json([]);

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

  // Classification guard
  if (config.type === "classification") return res.json({ ok: true });

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

// ---------------------------------------------------------------------------
// Crop — export a sub-region of an image (plus its labels) to a sibling dataset
// ---------------------------------------------------------------------------

// A box straddling the crop boundary is kept (clipped) only if at least this
// fraction of its original area survives the crop.
const CROP_KEEP_THRESHOLD = 0.5;

// Fill for pixels inside the crop rect but outside the crop polygon. 114 is the
// grey Ultralytics letterboxes with, so the model already reads it as "no content"
// — and the inference-time masking must use this same value.
const CROP_MASK_GREY = 114;

/** Shoelace area of a polygon given as [[x,y], ...]. Always non-negative. */
function polygonArea(pts) {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    a += pts[j][0] * pts[i][1] - pts[i][0] * pts[j][1];
  }
  return Math.abs(a) / 2;
}

/**
 * Sutherland–Hodgman: clip `pts` against an axis-aligned rect {x0,y0,x1,y1}.
 * The rect is convex, so this is exact for any (even concave) subject polygon.
 */
function clipPolygonToRect(pts, rect) {
  const edges = [
    { inside: p => p[0] >= rect.x0, isect: (a, b) => [rect.x0, a[1] + ((b[1] - a[1]) * (rect.x0 - a[0])) / (b[0] - a[0])] },
    { inside: p => p[0] <= rect.x1, isect: (a, b) => [rect.x1, a[1] + ((b[1] - a[1]) * (rect.x1 - a[0])) / (b[0] - a[0])] },
    { inside: p => p[1] >= rect.y0, isect: (a, b) => [a[0] + ((b[0] - a[0]) * (rect.y0 - a[1])) / (b[1] - a[1]), rect.y0] },
    { inside: p => p[1] <= rect.y1, isect: (a, b) => [a[0] + ((b[0] - a[0]) * (rect.y1 - a[1])) / (b[1] - a[1]), rect.y1] },
  ];
  let out = pts;
  for (const { inside, isect } of edges) {
    const input = out;
    out = [];
    for (let i = 0, j = input.length - 1; i < input.length; j = i++) {
      const cur = input[i], prev = input[j];
      const curIn = inside(cur), prevIn = inside(prev);
      if (curIn) {
        if (!prevIn) out.push(isect(prev, cur));
        out.push(cur);
      } else if (prevIn) {
        out.push(isect(prev, cur));
      }
    }
    if (out.length === 0) return [];
  }
  return out;
}

/** Bounding box of a normalized polygon. */
function polygonBounds(poly) {
  const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
}

/** Sibling output root: /data/boards -> /data/boards-cropped */
function cropOutputRoot(datasetRoot) {
  return path.resolve(datasetRoot) + "-cropped";
}

/** Label dir for a split, falling back to the conventional sibling of the images dir. */
function labelsRelFor(config, split) {
  const explicit = config.labelsDir?.[split];
  if (explicit) return explicit;
  const imagesRel = config[split];
  if (!imagesRel) return null;
  if (/(^|\/)images$/.test(imagesRel)) return imagesRel.replace(/(^|\/)images$/, "$1labels");
  if (/^images(\/|$)/.test(imagesRel)) return imagesRel.replace(/^images/, "labels");
  return `labels/${split}`;
}

/**
 * Re-map YOLO boxes into crop-local normalized coords.
 * `crop` is in source-image pixels; W/H are the source image dimensions.
 * Boxes are clipped to the crop and dropped when < CROP_KEEP_THRESHOLD survives.
 *
 * `polyPx` (optional, source pixels) is the mask polygon. Area outside it is
 * grey-filled on export, so a box's *visible* area is its overlap with the
 * polygon — not merely with the crop rect. Without this, a board sitting fully
 * inside the crop rect but under grey would keep its label and teach the model
 * to detect objects that aren't there.
 */
function remapBoxesToCrop(boxes, crop, W, H, polyPx) {
  const kept = [];
  let dropped = 0;
  for (const b of boxes) {
    // Box edges in source pixels
    const bx0 = (b.x - b.w / 2) * W, bx1 = (b.x + b.w / 2) * W;
    const by0 = (b.y - b.h / 2) * H, by1 = (b.y + b.h / 2) * H;
    const origArea = Math.max(0, bx1 - bx0) * Math.max(0, by1 - by0);
    if (origArea <= 0) { dropped++; continue; }

    const ix0 = Math.max(bx0, crop.left), ix1 = Math.min(bx1, crop.left + crop.width);
    const iy0 = Math.max(by0, crop.top), iy1 = Math.min(by1, crop.top + crop.height);
    const iw = ix1 - ix0, ih = iy1 - iy0;
    if (iw <= 0 || ih <= 0) { dropped++; continue; }

    // Visible area: exact polygon∩box when masking, else the rect intersection
    const visibleArea = polyPx
      ? polygonArea(clipPolygonToRect(polyPx, { x0: bx0, y0: by0, x1: bx1, y1: by1 }))
      : iw * ih;
    if (visibleArea / origArea < CROP_KEEP_THRESHOLD) { dropped++; continue; }

    kept.push({
      classId: b.classId,
      x: (ix0 + ix1) / 2 - crop.left,
      y: (iy0 + iy1) / 2 - crop.top,
      w: iw,
      h: ih,
    });
  }
  // Normalize against the cropped image size
  return {
    boxes: kept.map(b => ({
      classId: b.classId,
      x: b.x / crop.width,
      y: b.y / crop.height,
      w: b.w / crop.width,
      h: b.h / crop.height,
    })),
    dropped,
  };
}

/** Copy dataset-level metadata files (data.yaml etc.) into the output root, once. */
async function copyDatasetMetaFiles(datasetRoot, outRoot) {
  for (const name of ["data.yaml", "dataset.yaml", "dataset_weighted.yaml", "obj.names"]) {
    const src = path.join(datasetRoot, name);
    const dst = path.join(outRoot, name);
    try {
      await fs.access(dst);
      continue; // already copied
    } catch {}
    try { await fs.copyFile(src, dst); } catch {}
  }
}

/** Resolve source/output paths for a croppable image, or an { error, status } object. */
async function resolveCropPaths(datasetRoot, split, name) {
  if (!datasetRoot.trim()) return { error: "no dataset configured", status: 400 };
  const config = await resolveConfig(datasetRoot);
  if (config.type === "classification") return { error: "crop is only supported for detection datasets", status: 400 };

  const imagesRel = config[split];
  if (!imagesRel) return { error: `split not found: ${split}`, status: 404 };

  const srcImage = path.join(datasetRoot, imagesRel, name);
  if (!srcImage.startsWith(path.resolve(datasetRoot))) return { error: "forbidden", status: 403 };

  const outRoot = cropOutputRoot(datasetRoot);
  const labelsRel = labelsRelFor(config, split);
  const base = stripImageExt(name);
  return {
    config, imagesRel, labelsRel, base, srcImage, outRoot,
    srcLabel: labelsRel ? path.join(datasetRoot, labelsRel, base + ".txt") : null,
    outImage: path.join(outRoot, imagesRel, name),
    outLabel: labelsRel ? path.join(outRoot, labelsRel, base + ".txt") : null,
  };
}

// Has this image already been cropped into the output dataset?
app.get("/api/crop/status", async (req, res) => {
  const datasetRoot = getDatasetPath();
  const { split, name } = req.query;
  if (!split || !name) return res.status(400).json({ error: "split and name required" });
  const p = await resolveCropPaths(datasetRoot, String(split), String(name));
  if (p.error) return res.json({ exists: false, outputRoot: null });
  let exists = false;
  try { await fs.access(p.outImage); exists = true; } catch {}
  res.json({ exists, outputRoot: p.outRoot });
});

// Write the cropped image + remapped labels to the sibling dataset.
app.post("/api/crop", async (req, res) => {
  const datasetRoot = getDatasetPath();
  const { split, name, region } = req.body || {};
  if (!split || !name || !region) return res.status(400).json({ error: "split, name and region required" });

  const p = await resolveCropPaths(datasetRoot, String(split), String(name));
  if (p.error) return res.status(p.status).json({ error: p.error });

  try {
    const meta = await sharp(p.srcImage).metadata();
    const W = meta.width, H = meta.height;
    if (!W || !H) return res.status(500).json({ error: "could not read image dimensions" });

    // A polygon defines its own bounding box; a bare rect uses the given corners.
    const poly = Array.isArray(region.polygon) && region.polygon.length >= 3 ? region.polygon : null;
    const bounds = poly ? polygonBounds(poly) : region;

    // Normalized region -> integer pixel rect, clamped to the image
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const left = clamp(Math.round(Math.min(bounds.x0, bounds.x1) * W), 0, W - 1);
    const top = clamp(Math.round(Math.min(bounds.y0, bounds.y1) * H), 0, H - 1);
    const right = clamp(Math.round(Math.max(bounds.x0, bounds.x1) * W), left + 1, W);
    const bottom = clamp(Math.round(Math.max(bounds.y0, bounds.y1) * H), top + 1, H);
    const crop = { left, top, width: right - left, height: bottom - top };

    // Polygon in source pixels (for label visibility) and crop-local pixels (for the mask)
    const polyPx = poly ? poly.map(([px, py]) => [px * W, py * H]) : null;
    const polyLocal = polyPx ? polyPx.map(([px, py]) => [px - crop.left, py - crop.top]) : null;

    // Labels: read source, remap, write
    let boxes = [];
    if (p.srcLabel) {
      try {
        const content = await fs.readFile(p.srcLabel, "utf8");
        boxes = content.split("\n").filter(l => l.trim()).map(line => {
          const v = line.trim().split(/\s+/).map(Number);
          return { classId: v[0], x: v[1], y: v[2], w: v[3], h: v[4] };
        }).filter(b => Number.isFinite(b.x) && Number.isFinite(b.y) && Number.isFinite(b.w) && Number.isFinite(b.h));
      } catch {}
    }
    const { boxes: remapped, dropped } = remapBoxesToCrop(boxes, crop, W, H, polyPx);

    // Image
    await fs.mkdir(path.dirname(p.outImage), { recursive: true });
    const ext = path.extname(name).toLowerCase();
    let pipeline = sharp(p.srcImage).extract(crop);

    if (polyLocal) {
      // Overlay a grey layer with a polygon-shaped hole (even-odd fill): pixels
      // inside the polygon are untouched, everything else becomes CROP_MASK_GREY.
      // Done as a single `over` composite because sharp applies flatten *before*
      // composite in its fixed pipeline order — a dest-in + flatten pass would
      // flatten the un-masked image and leave the masked area black.
      const inner = polyLocal.map(([px, py], i) => `${i === 0 ? "M" : "L"} ${px.toFixed(2)} ${py.toFixed(2)}`).join(" ");
      const d = `M 0 0 L ${crop.width} 0 L ${crop.width} ${crop.height} L 0 ${crop.height} Z ${inner} Z`;
      const g = CROP_MASK_GREY;
      const maskSvg = Buffer.from(
        `<svg xmlns="http://www.w3.org/2000/svg" width="${crop.width}" height="${crop.height}">` +
        `<path d="${d}" fill="rgb(${g},${g},${g})" fill-rule="evenodd"/></svg>`
      );
      pipeline = pipeline.composite([{ input: maskSvg, blend: "over" }]);
    }

    if (ext === ".jpg" || ext === ".jpeg") pipeline = pipeline.jpeg({ quality: 95 });
    else if (ext === ".png") pipeline = pipeline.png();
    else if (ext === ".webp") pipeline = pipeline.webp({ quality: 95 });
    await pipeline.toFile(p.outImage);

    if (p.outLabel) {
      await fs.mkdir(path.dirname(p.outLabel), { recursive: true });
      await fs.writeFile(p.outLabel, remapped.map(b => `${b.classId} ${b.x} ${b.y} ${b.w} ${b.h}`).join("\n"), "utf8");
    }

    await copyDatasetMetaFiles(datasetRoot, p.outRoot);

    res.json({
      ok: true,
      outputRoot: p.outRoot,
      outImage: p.outImage,
      kept: remapped.length,
      dropped,
      width: crop.width,
      height: crop.height,
      masked: !!polyLocal,
    });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
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

    // Classification branch
    if (config.type === "classification") {
      const allImages = await collectClassificationImages(datasetRoot, config);

      // Duplicate images by MD5
      const hashMap = {};
      for (const img of allImages) {
        const imgPath = path.join(datasetRoot, img.imageRel);
        try {
          const buf = await fs.readFile(imgPath);
          const hash = createHash("md5").update(buf).digest("hex");
          if (!hashMap[hash]) hashMap[hash] = [];
          hashMap[hash].push({ split: img.split, name: img.name, imageRel: img.imageRel, relPath: img.relPath, className: img.className });
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

      // Class distribution
      const classCounts = {};
      for (const img of allImages) {
        classCounts[img.className] = (classCounts[img.className] || 0) + 1;
      }

      res.json({ checks: [
        { id: "duplicate_images", name: "Duplicate images (by MD5)", count: duplicateImages.length, severity: duplicateImages.length ? "warning" : "ok", detail: duplicateImages, extra: { totalDupImages } },
        { id: "class_balance", name: "Class distribution", count: Object.keys(classCounts).length, severity: "ok", detail: classCounts },
      ]});
      return;
    }

    // Detection branch
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
  const { split, base, reviewed: mark, reviewKey } = req.body;

  let key;
  if (reviewKey) {
    key = reviewKey;
  } else {
    if (!split || base == null) return res.status(400).json({ error: "split and base required" });
    key = `${split}/${typeof base === "string" ? base.replace(/\.[^.]+$/, "") : base}`;
  }

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

  // Classification support
  if (config.type === "classification") {
    const imageRel = req.query.imageRel;
    if (imageRel) {
      const imagePath = path.join(datasetRoot, imageRel);
      if (!imagePath.startsWith(path.resolve(datasetRoot))) return res.status(403).json({ error: "forbidden" });
      try { await fs.unlink(imagePath); } catch (e) { if (e.code !== "ENOENT") return res.status(500).json({ error: String(e.message) }); }
      // Remove from reviewed
      const reviewKeyVal = classificationReviewKey(imageRel);
      try {
        const d = JSON.parse(await fs.readFile(getReviewedPath(datasetRoot), "utf8"));
        if (Array.isArray(d.reviewed)) { d.reviewed = d.reviewed.filter(k => k !== reviewKeyVal); await fs.writeFile(getReviewedPath(datasetRoot), JSON.stringify(d, null, 2), "utf8"); }
      } catch {}
      invalidateConfigCache();
      return res.json({ ok: true });
    }
  }

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
// Classification endpoints
// ---------------------------------------------------------------------------

app.post("/api/classification/move", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  try {
    const config = await resolveConfig(datasetRoot);
    if (config.type !== "classification") return res.status(400).json({ error: "not a classification dataset" });
    const { imageRels, targetClassName } = req.body;
    if (!Array.isArray(imageRels) || !targetClassName) return res.status(400).json({ error: "imageRels and targetClassName required" });

    let moved = 0;
    for (const imageRel of imageRels) {
      const srcPath = path.join(datasetRoot, imageRel);
      if (!srcPath.startsWith(path.resolve(datasetRoot))) continue;
      const fileName = path.basename(imageRel);
      const parts = imageRel.split("/");
      let destRel;
      if (config.train || config.val || config.test) {
        // YOLO cls: preserve split dir, change class folder
        // imageRel = "train/cat/img.jpg" -> "train/dog/img.jpg"
        const splitDir = parts[0];
        destRel = `${splitDir}/${targetClassName}/${fileName}`;
      } else {
        // Simple cls: "cat/img.jpg" -> "dog/img.jpg"
        destRel = `${targetClassName}/${fileName}`;
      }
      const destPath = path.join(datasetRoot, destRel);
      if (!destPath.startsWith(path.resolve(datasetRoot))) continue;
      await fs.mkdir(path.dirname(destPath), { recursive: true });
      try {
        await fs.rename(srcPath, destPath);
        moved++;
      } catch {}
    }
    invalidateConfigCache();
    res.json({ ok: true, moved });
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
});

app.post("/api/classification/delete-images", async (req, res) => {
  const datasetRoot = getDatasetPath();
  if (!datasetRoot.trim()) return res.status(400).json({ error: "no dataset" });
  try {
    const config = await resolveConfig(datasetRoot);
    if (config.type !== "classification") return res.status(400).json({ error: "not a classification dataset" });
    const { imageRels } = req.body;
    if (!Array.isArray(imageRels)) return res.status(400).json({ error: "imageRels required" });

    let deleted = 0;
    for (const imageRel of imageRels) {
      const filePath = path.join(datasetRoot, imageRel);
      if (!filePath.startsWith(path.resolve(datasetRoot))) continue;
      try {
        await fs.unlink(filePath);
        deleted++;
      } catch {}
      // Remove from reviewed
      const reviewKeyVal = classificationReviewKey(imageRel);
      try {
        const d = JSON.parse(await fs.readFile(getReviewedPath(datasetRoot), "utf8"));
        if (Array.isArray(d.reviewed)) { d.reviewed = d.reviewed.filter(k => k !== reviewKeyVal); await fs.writeFile(getReviewedPath(datasetRoot), JSON.stringify(d, null, 2), "utf8"); }
      } catch {}
    }
    invalidateConfigCache();
    res.json({ ok: true, deleted });
  } catch (e) { res.status(500).json({ error: String(e.message) }); }
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
