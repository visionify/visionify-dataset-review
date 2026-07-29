import { useEffect, useState, useCallback, useRef } from "react";
import { Link, useParams, useLocation, useNavigate } from "react-router-dom";
import { api, imageBase, imageSrc } from "@/api";
import type { PredictionBox } from "@/api";
import { BBoxCanvas } from "@/components/BBoxCanvas";
import type { CropMode } from "@/components/BBoxCanvas";
import type { BBox, CropRegion, ImageItem, ClassItem } from "@/types";

const DEFAULT_CLASS_KEY = "pallet-review-default-class";
const CROP_REGION_KEY = "pallet-review-crop-region";
const LOAD_LIMIT = 5000;

export default function ImageDetailPage() {
  const { split, name } = useParams<{ split: string; name: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as { fromSplit?: string; filterReviewed?: string; classId?: string; classSort?: string; startIndex?: number; tagType?: string; tag?: string } | null;

  const fromSplit = state?.fromSplit ?? "all";
  const filterReviewed = state?.filterReviewed as "yes" | "no" | undefined;
  const filterClassId = state?.classId != null ? parseInt(state.classId, 10) : null;
  const classSort = state?.classSort || "";
  const tagType = state?.tagType || "";
  const tagValue = state?.tag || "";
  const cameFromImagesPage = state?.fromSplit != null;

  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [classColors, setClassColors] = useState<Record<number, string>>({});
  const [summary, setSummary] = useState<Awaited<ReturnType<typeof api.getSummary>> | null>(null);
  const [allImages, setAllImages] = useState<ImageItem[]>([]);
  const [listReady, setListReady] = useState(false);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [boxes, setBoxes] = useState<BBox[]>([]);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [defaultClassId, setDefaultClassIdState] = useState(0);
  const [showTags, setShowTags] = useState(false);
  const [tagList, setTagList] = useState<[string, string][]>([]);
  const [deleting, setDeleting] = useState(false);
  const [isReviewed, setIsReviewed] = useState(false);
  const [predictions, setPredictions] = useState<PredictionBox[]>([]);
  const [detecting, setDetecting] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [confidence, setConfidence] = useState(0.25);
  const [moveTarget, setMoveTarget] = useState("");
  const [moving, setMoving] = useState(false);
  const [cropRegion, setCropRegionState] = useState<CropRegion | null>(null);
  const [cropMode, setCropMode] = useState<CropMode>(null);
  const [cropping, setCropping] = useState(false);
  const [cropExists, setCropExists] = useState(false);
  const [messageKind, setMessageKind] = useState<"success" | "error">("success");
  const [cropInfo, setCropInfo] = useState<string | null>(null);
  const cropInFlight = useRef(false);
  const boxesRef = useRef(boxes);
  boxesRef.current = boxes;
  const loadedImageRef = useRef<{ split: string; name: string } | null>(null);

  const datasetType = summary?.type ?? "detection";
  const isCls = datasetType === "classification";

  const setDefaultClassId = useCallback((id: number) => {
    setDefaultClassIdState(id);
    try { localStorage.setItem(DEFAULT_CLASS_KEY, String(id)); } catch {}
  }, []);

  /**
   * Transient status. Rendered into a fixed-size slot (and, for errors, an
   * absolutely-positioned toast) so showing/hiding it never reflows the toolbar
   * and never shifts the image below it.
   */
  const notify = useCallback((text: string, kind: "success" | "error" = "success", ms = 2500) => {
    setMessage(text);
    setMessageKind(kind);
    setTimeout(() => setMessage(null), ms);
  }, []);

  // The crop region persists across images (and reloads) so a whole scene can be
  // swept with a single keystroke per image.
  const setCropRegion = useCallback((r: CropRegion | null) => {
    setCropRegionState(r);
    try {
      if (r) localStorage.setItem(CROP_REGION_KEY, JSON.stringify(r));
      else localStorage.removeItem(CROP_REGION_KEY);
    } catch {}
  }, []);

  // Load persisted default class + crop region
  useEffect(() => {
    try {
      const s = localStorage.getItem(DEFAULT_CLASS_KEY);
      if (s != null) { const n = parseInt(s, 10); if (!isNaN(n)) setDefaultClassIdState(n); }
    } catch {}
    try {
      const s = localStorage.getItem(CROP_REGION_KEY);
      if (s) {
        const r = JSON.parse(s) as CropRegion;
        const validPoly = r?.polygon == null
          || (Array.isArray(r.polygon) && r.polygon.length >= 3
              && r.polygon.every(p => Array.isArray(p) && p.length === 2 && p.every(v => typeof v === "number" && isFinite(v))));
        if ([r?.x0, r?.y0, r?.x1, r?.y1].every(v => typeof v === "number" && isFinite(v)) && validPoly) setCropRegionState(r);
      }
    } catch {}
  }, []);

  // Load summary + class colors + model health once
  useEffect(() => {
    api.getSummary().then((s) => { setClasses(s.classes ?? []); setSummary(s); }).catch(() => {});
    api.getClassColors().then((c) => setClassColors(c || {})).catch(() => setClassColors({}));
    api.inferenceHealth().then(h => setModelReady(h.model_loaded)).catch(() => setModelReady(false));
  }, []);

  // Load full image list for the split (or class) — this enables seamless navigation
  useEffect(() => {
    setListReady(false);
    const loadAll = async () => {
      const all: ImageItem[] = [];
      let page = 1;
      let hasMore = true;
      if (filterClassId != null && !isNaN(filterClassId) && !cameFromImagesPage) {
        // ClassDetailPage flow — use dedicated class images endpoint
        while (hasMore) {
          const r = await api.getClassImages(filterClassId, page, LOAD_LIMIT, classSort || undefined);
          all.push(...r.images);
          hasMore = r.images.length === LOAD_LIMIT;
          page++;
        }
      } else {
        // ImagesPage flow — use general images endpoint (with optional classId + sort)
        while (hasMore) {
          const r = await api.getImages({
            split: fromSplit, page, limit: LOAD_LIMIT, reviewed: filterReviewed,
            tagType: tagType || undefined, tag: tagValue || undefined,
            classId: (cameFromImagesPage && filterClassId != null) ? filterClassId : undefined,
            sort: (cameFromImagesPage && filterClassId != null && classSort) ? classSort : undefined,
          });
          all.push(...r.images);
          hasMore = r.images.length === LOAD_LIMIT;
          page++;
        }
      }
      setAllImages(all);
      if (split && name) {
        const idx = all.findIndex(img => img.split === split && img.name === name);
        setCurrentIdx(idx >= 0 ? idx : state?.startIndex ?? 0);
      } else {
        setCurrentIdx(state?.startIndex ?? 0);
      }
      setListReady(true);
    };
    loadAll().catch(() => setListReady(true));
  }, [fromSplit, filterReviewed, filterClassId, classSort, tagType, tagValue, cameFromImagesPage]);

  const currentImage = allImages[currentIdx] ?? (split && name ? { split, name, imageRel: "", relPath: "" } : null);
  const hasPrev = currentIdx > 0;
  const hasNext = currentIdx < allImages.length - 1;
  const backLink = (() => {
    if (filterClassId != null && !cameFromImagesPage) {
      return `/class/${filterClassId}${classSort ? `?sort=${classSort}` : ""}`;
    }
    const params = new URLSearchParams();
    if (tagType && tagValue) { params.set("tagType", tagType); params.set("tag", tagValue); }
    if (cameFromImagesPage && filterClassId != null) params.set("classId", String(filterClassId));
    if (cameFromImagesPage && classSort) params.set("sort", classSort);
    const qs = params.toString();
    return `/images/${fromSplit}${qs ? `?${qs}` : ""}`;
  })();

  const reviewKey = (img: { name: string; imageRel: string }) => {
    if (isCls) {
      const rel = img.imageRel;
      const dot = rel.lastIndexOf(".");
      return dot > 0 ? rel.slice(0, dot) : rel;
    }
    return undefined; // use default server behavior
  };

  // Load annotations + tags + reviewed status when image changes
  useEffect(() => {
    if (!currentImage) return;
    const s = currentImage.split;
    const n = currentImage.name;
    const b = imageBase(n);

    // Immediately invalidate: boxes no longer correspond to a loaded image
    loadedImageRef.current = null;
    setSelectedIndex(null);
    setIsReviewed(false);
    setPredictions([]);
    setBoxes([]);
    setCropExists(false);
    setCropInfo(null);

    let stale = false;

    if (!isCls) {
      api.cropStatus(s, n).then(r => { if (!stale) setCropExists(r.exists); }).catch(() => {});
    }

    const promises: [Promise<BBox[]>, Promise<Record<string, unknown>>, Promise<{ reviewed: string[] }>] = [
      isCls ? Promise.resolve([]) : api.getAnnotations(s, b),
      api.getTags(s, b),
      api.getReviewed(),
    ];

    Promise.all(promises).then(([ann, t, rev]) => {
      if (stale) return;
      setBoxes(ann);
      loadedImageRef.current = { split: s, name: n };
      const tagObj = (t && typeof t === "object" && !Array.isArray(t)) ? t as Record<string, unknown> : {};
      setTagList(Object.entries(tagObj).map(([k, v]) => [k, String(v ?? "")]));
      const rk = reviewKey(currentImage) ?? `${s}/${b}`;
      setIsReviewed(rev.reviewed.includes(rk));
      if (!isCls && filterClassId != null && classSort.startsWith("area") && ann.length > 0) {
        let smallestIdx = -1;
        let smallestArea = Infinity;
        ann.forEach((box: BBox, i: number) => {
          if (box.classId !== filterClassId) return;
          const area = box.w * box.h;
          if (area < smallestArea) { smallestArea = area; smallestIdx = i; }
        });
        if (smallestIdx >= 0) setSelectedIndex(smallestIdx);
      }
    }).catch(() => { if (!stale) { setBoxes([]); setTagList([]); } });

    // Update URL to match current image (without full page reload)
    const url = `/image/${encodeURIComponent(s)}/${encodeURIComponent(n)}`;
    if (location.pathname !== url)
      navigate(url, { replace: true, state: { fromSplit, filterReviewed, classId: state?.classId, classSort: state?.classSort, startIndex: currentIdx, tagType: tagType || undefined, tag: tagValue || undefined } });

    return () => { stale = true; };
  }, [currentIdx, currentImage?.split, currentImage?.name]);

  const markReviewed = useCallback(() => {
    if (!currentImage) return;
    const rk = reviewKey(currentImage);
    api.setReviewed(currentImage.split, imageBase(currentImage.name), true, rk).catch(() => {});
    setIsReviewed(true);
  }, [currentImage, isCls]);

  const saveAnnotations = useCallback(async () => {
    if (!currentImage || isCls) return;
    setSaving(true); setMessage(null);
    try {
      await api.saveAnnotations(currentImage.split, imageBase(currentImage.name), boxesRef.current);
      markReviewed();
      setMessage("Saved & reviewed.");
      setTimeout(() => setMessage(null), 2000);
    } catch (e) { setMessage(e instanceof Error ? e.message : "Save failed"); }
    finally { setSaving(false); }
  }, [currentImage, markReviewed, isCls]);

  const saveTags = useCallback(async () => {
    if (!currentImage) return;
    const obj = Object.fromEntries(tagList.filter(([k]) => k.trim() !== ""));
    setSaving(true); setMessage(null);
    try { await api.saveTags(currentImage.split, imageBase(currentImage.name), obj); setMessage("Tags saved."); setTimeout(() => setMessage(null), 2000); }
    catch (e) { setMessage(e instanceof Error ? e.message : "Save failed"); }
    finally { setSaving(false); }
  }, [currentImage, tagList]);

  const autoSaveAndMark = useCallback(() => {
    if (isCls) return; // no auto-save for classification
    const loaded = loadedImageRef.current;
    if (!loaded) return;
    const s = loaded.split;
    const b = imageBase(loaded.name);
    api.saveAnnotations(s, b, boxesRef.current).catch(() => {});
    api.setReviewed(s, b, true).catch(() => {});
    loadedImageRef.current = null;
  }, [isCls]);

  const goNext = useCallback(() => {
    if (!hasNext) return;
    autoSaveAndMark();
    setCurrentIdx(i => i + 1);
  }, [hasNext, autoSaveAndMark]);

  const goPrev = useCallback(() => {
    if (!hasPrev) return;
    autoSaveAndMark();
    setCurrentIdx(i => i - 1);
  }, [hasPrev, autoSaveAndMark]);

  const handleThumbsUp = useCallback(() => {
    markReviewed();
    if (hasNext) setCurrentIdx(i => i + 1);
    else { setMessage("Marked as reviewed."); setTimeout(() => setMessage(null), 2000); }
  }, [markReviewed, hasNext]);

  const handleDeleteImage = useCallback(async () => {
    if (!currentImage) return;
    setDeleting(true);
    try {
      if (isCls) {
        await api.classificationDeleteImages([currentImage.imageRel]);
      } else {
        await api.deleteImage(currentImage.split, currentImage.name);
      }
      const newList = allImages.filter((_, i) => i !== currentIdx);
      setAllImages(newList);
      if (newList.length === 0) { navigate(backLink, { replace: true }); return; }
      setCurrentIdx(Math.min(currentIdx, newList.length - 1));
    } catch (e) { setMessage(e instanceof Error ? e.message : "Delete failed"); }
    finally { setDeleting(false); }
  }, [currentImage, allImages, currentIdx, backLink, navigate, isCls]);

  const handleMoveImage = useCallback(async () => {
    if (!currentImage || !moveTarget) return;
    setMoving(true);
    try {
      await api.classificationMoveImages([currentImage.imageRel], moveTarget);
      setMoveTarget("");
      const newList = allImages.filter((_, i) => i !== currentIdx);
      setAllImages(newList);
      if (newList.length === 0) { navigate(backLink, { replace: true }); return; }
      setCurrentIdx(Math.min(currentIdx, newList.length - 1));
      setMessage(`Moved to "${moveTarget}".`);
      setTimeout(() => setMessage(null), 2000);
    } catch (e) { setMessage(e instanceof Error ? e.message : "Move failed"); }
    finally { setMoving(false); }
  }, [currentImage, moveTarget, allImages, currentIdx, backLink, navigate]);

  // Cycle class on selected box
  const cycleBoxClass = useCallback(() => {
    if (isCls || selectedIndex === null || !boxes[selectedIndex]) return;
    const maxClass = classes.length;
    if (!maxClass) return;
    setBoxes(prev => prev.map((b, i) => {
      if (i !== selectedIndex) return b;
      const next = (b.classId + 1) % maxClass;
      setDefaultClassId(next);
      return { ...b, classId: next };
    }));
  }, [selectedIndex, boxes, classes.length, setDefaultClassId, isCls]);

  // When a new box is created or a box class changes, update defaultClassId
  const handleBoxesChange = useCallback((newBoxes: BBox[]) => {
    if (newBoxes.length > boxes.length) {
      const last = newBoxes[newBoxes.length - 1];
      if (last) setDefaultClassId(last.classId);
    }
    setBoxes(newBoxes);
  }, [boxes.length, setDefaultClassId]);

  // When selected box class is changed via dropdown, update default
  const handleSelectedClassChange = useCallback((classId: number) => {
    if (selectedIndex === null) return;
    setBoxes(prev => prev.map((b, i) => i === selectedIndex ? { ...b, classId } : b));
    setDefaultClassId(classId);
  }, [selectedIndex, setDefaultClassId]);

  const handleAutoDetect = useCallback(async () => {
    if (!currentImage || isCls) return;
    setDetecting(true);
    try {
      const r = await api.inferencePredict(currentImage.split, currentImage.name, confidence);
      setPredictions(r.boxes);
      if (!r.boxes.length) { setMessage("No detections."); setTimeout(() => setMessage(null), 2000); }
    } catch (e) { setMessage(e instanceof Error ? e.message : "Detection failed"); setTimeout(() => setMessage(null), 3000); }
    finally { setDetecting(false); }
  }, [currentImage, confidence, isCls]);

  const acceptPrediction = useCallback((index: number) => {
    const pred = predictions[index];
    if (!pred) return;
    setBoxes(prev => [...prev, { classId: pred.classId, x: pred.x, y: pred.y, w: pred.w, h: pred.h }]);
    setPredictions(prev => prev.filter((_, i) => i !== index));
  }, [predictions]);

  // Write the current crop region (image + remapped labels) to <dataset>-cropped
  const applyCrop = useCallback(async () => {
    if (!currentImage || isCls) return;
    // Re-entrancy guard: a ref, not the `cropping` state, so a second keypress in
    // the same tick can't slip through before React re-renders.
    if (cropInFlight.current) return;
    if (!cropRegion) { notify("Set a crop region first", "error"); return; }
    if (cropMode) return;

    // Never write annotations that don't belong to the image on screen: until the
    // fetch completes, boxesRef holds [] (or the previous image's boxes), and
    // saving that would wipe the source labels. Same guard autoSaveAndMark uses.
    const loaded = loadedImageRef.current;
    if (!loaded || loaded.split !== currentImage.split || loaded.name !== currentImage.name) {
      notify("Still loading annotations — try again", "error");
      return;
    }

    cropInFlight.current = true;
    setCropping(true);
    try {
      // Persist any in-progress annotation edits first — the crop reads labels from disk
      await api.saveAnnotations(currentImage.split, imageBase(currentImage.name), boxesRef.current);
      const r = await api.cropImage(currentImage.split, currentImage.name, cropRegion);
      setCropExists(true);
      markReviewed();
      setCropInfo(`Cropped ${r.width}×${r.height} · ${r.kept} box${r.kept === 1 ? "" : "es"} kept${r.dropped ? `, ${r.dropped} dropped` : ""}`);
      notify("Cropped", "success");
    } catch (e) {
      notify(e instanceof Error ? e.message : "Crop failed", "error", 4000);
    } finally {
      cropInFlight.current = false;
      setCropping(false);
    }
  }, [currentImage, cropRegion, cropMode, isCls, markReviewed, notify]);

  const startCrop = useCallback((mode: "rect" | "polygon") => {
    setCropRegion(null);
    setCropMode(mode);
    setSelectedIndex(null);
  }, [setCropRegion]);

  const handleCropRegionChange = useCallback((r: CropRegion | null) => {
    setCropRegion(r);
    // A region drawn from scratch ends draw mode; vertex drags on an existing
    // polygon arrive with cropMode already null and must not re-trigger it.
    if (r && cropMode) setCropMode(null);
  }, [setCropRegion, cropMode]);

  const acceptAllPredictions = useCallback(() => {
    const newBoxes = predictions.map(p => ({ classId: p.classId, x: p.x, y: p.y, w: p.w, h: p.h }));
    setBoxes(prev => [...prev, ...newBoxes]);
    setPredictions([]);
  }, [predictions]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if (!isCls && e.key === "d" && !e.ctrlKey && !e.metaKey && selectedIndex !== null) {
        e.preventDefault();
        setBoxes(prev => prev.filter((_, i) => i !== selectedIndex));
        setSelectedIndex(null);
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        if (!isCls && selectedIndex !== null) {
          setBoxes(prev => prev.filter((_, i) => i !== selectedIndex));
          setSelectedIndex(null);
        } else {
          handleDeleteImage();
        }
        return;
      }
      if (!isCls && e.ctrlKey && e.key === "s") { e.preventDefault(); saveAnnotations(); return; }
      if (e.key === "t" && !e.ctrlKey && !e.metaKey && !e.altKey) { setShowTags(s => !s); return; }
      if (!isCls && e.key === "c" && !e.ctrlKey && !e.metaKey && selectedIndex !== null) { e.preventDefault(); cycleBoxClass(); return; }
      // X only ever *saves* the crop — setting/clearing the region is UI-driven.
      if (!isCls && (e.key === "x" || e.key === "X") && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
        e.preventDefault();
        applyCrop();
        return;
      }
      if (e.key === "Escape" && cropMode) { e.preventDefault(); setCropMode(null); return; }
      if (e.key === " ") { e.preventDefault(); handleThumbsUp(); return; }
      if (!isCls && e.key === "a" && !e.ctrlKey && !e.metaKey && selectedIndex === null) { e.preventDefault(); if (predictions.length) acceptAllPredictions(); else handleAutoDetect(); return; }

      if (!isCls) {
        const num = parseInt(e.key, 10);
        if (e.key >= "1" && e.key <= "9" && num >= 1 && num <= 9 && selectedIndex !== null && classes.length >= num) {
          e.preventDefault();
          handleSelectedClassChange(num - 1);
        }
        if (e.key === "0" && selectedIndex !== null && classes.length > 0) {
          e.preventDefault();
          handleSelectedClassChange(0);
        }
      }
      if (e.key === "ArrowLeft") { e.preventDefault(); goPrev(); }
      if (e.key === "ArrowRight") { e.preventDefault(); goNext(); }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isCls, selectedIndex, saveAnnotations, classes.length, goPrev, goNext, handleDeleteImage, handleThumbsUp, cycleBoxClass, handleSelectedClassChange, predictions.length, acceptAllPredictions, handleAutoDetect, applyCrop, cropMode]);

  const addTag = () => setTagList(prev => [...prev, ["", ""]]);
  const updateTag = (i: number, k: 0 | 1, v: string) =>
    setTagList(prev => prev.map((row, j) => j === i ? (k === 0 ? [v, row[1]] : [row[0], v]) : row));
  const removeTag = (i: number) => setTagList(prev => prev.filter((_, j) => j !== i));

  if (!currentImage) {
    return (
      <div>
        <Link to={backLink} className="btn btn-ghost">← Back</Link>
        <p style={{ marginTop: "1rem" }}>{listReady ? "No images found." : "Loading images…"}</p>
      </div>
    );
  }

  const classNames = Object.fromEntries(classes.map(c => [c.id, c.name]));
  const totalImages = summary?.totalImages ?? 0;
  const reviewedCount = summary?.reviewedCount ?? 0;
  const pctReviewed = totalImages ? Math.round((reviewedCount / totalImages) * 100) : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: "80vh" }}>
      {/* Compact toolbar */}
      {/* nowrap + horizontal scroll: the toolbar can never change height, so nothing
          below it ever shifts when transient controls appear or disappear. */}
      <div style={{ position: "relative", display: "flex", flexWrap: "nowrap", overflowX: "auto", alignItems: "center", gap: "0.5rem", padding: "0.4rem 0.75rem", borderBottom: "1px solid var(--color-border)", background: "var(--color-surface)", fontSize: "0.85rem" }}>
        <Link to={backLink} className="btn btn-ghost" style={{ padding: "0.3rem 0.5rem" }}>← Back</Link>

        <button className="btn btn-ghost" style={{ padding: "0.3rem 0.5rem" }} onClick={goPrev} disabled={!hasPrev}>←</button>
        <span style={{ color: "var(--color-text-muted)", fontVariantNumeric: "tabular-nums", minWidth: "5rem", textAlign: "center" }}>
          {currentIdx + 1} / {allImages.length}
        </span>
        <button className="btn btn-ghost" style={{ padding: "0.3rem 0.5rem" }} onClick={goNext} disabled={!hasNext}>→</button>

        <span style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>{pctReviewed}%</span>
        <label style={{ display: "flex", alignItems: "center", gap: "0.25rem", fontSize: "0.8rem", color: isReviewed ? "var(--color-success)" : "var(--color-text-muted)", cursor: "pointer" }} title="Mark as reviewed">
          <input type="checkbox" checked={isReviewed} onChange={e => {
            const v = e.target.checked;
            setIsReviewed(v);
            if (currentImage) {
              const rk = reviewKey(currentImage);
              api.setReviewed(currentImage.split, imageBase(currentImage.name), v, rk).catch(() => {});
            }
          }} />
          Reviewed
        </label>

        {isCls && currentImage.className && (
          <span style={{ fontSize: "0.8rem", padding: "2px 8px", borderRadius: 4, background: "rgba(99,102,241,0.12)", color: "var(--color-text)", fontWeight: 600 }}>
            {currentImage.className}
          </span>
        )}

        {!isCls && (
          <select className="input" style={{ width: "auto", padding: "0.3rem 0.4rem", maxWidth: "120px" }} value={defaultClassId} onChange={e => setDefaultClassId(parseInt(e.target.value, 10))} title="Class for new boxes">
            {classes.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        )}

        {!isCls && (
          <button className="btn btn-primary" onClick={saveAnnotations} disabled={saving} style={{ padding: "0.3rem 0.6rem" }}>
            {saving ? "…" : "Save (Ctrl+S)"}
          </button>
        )}
        <button className="btn btn-ghost" onClick={handleThumbsUp} style={{ padding: "0.3rem 0.5rem", fontSize: "1.1rem" }} title="Mark reviewed & next (Space)">
          👍
        </button>
        <button className="btn btn-ghost" style={{ padding: "0.3rem 0.5rem", color: "var(--color-danger)" }} onClick={handleDeleteImage} disabled={deleting} title="Delete image (Del)">
          {deleting ? "…" : "🗑"}
        </button>
        <button className="btn btn-ghost" onClick={() => setShowTags(s => !s)} style={{ padding: "0.3rem 0.5rem" }}>
          Tags {showTags ? "▼" : "▶"}
        </button>

        {isCls && classes.length > 1 && (
          <>
            <span style={{ borderLeft: "1px solid var(--color-border)", height: "1.2rem" }} />
            <select className="input" style={{ width: "auto", padding: "0.25rem 0.4rem", fontSize: "0.85rem" }} value={moveTarget} onChange={e => setMoveTarget(e.target.value)}>
              <option value="">Move to…</option>
              {classes.filter(c => c.name !== currentImage.className).map(c => <option key={c.id} value={c.name}>{c.name}</option>)}
            </select>
            {moveTarget && (
              <button className="btn btn-primary" style={{ padding: "0.3rem 0.6rem" }} onClick={handleMoveImage} disabled={moving}>
                {moving ? "Moving…" : "Move"}
              </button>
            )}
          </>
        )}

        {!isCls && (
          <>
            <span style={{ borderLeft: "1px solid var(--color-border)", height: "1.2rem" }} />

            {cropMode ? (
              <>
                <span style={{ fontSize: "0.8rem", color: "#b45309", fontWeight: 600, whiteSpace: "nowrap" }}>
                  {cropMode === "rect"
                    ? "Drag to set crop region"
                    : "Click to add points · click the first point or double-click to close"}
                </span>
                <button className="btn btn-ghost" onClick={() => setCropMode(null)} style={{ padding: "0.3rem 0.5rem", fontSize: "0.8rem" }}>
                  Cancel (Esc)
                </button>
              </>
            ) : cropRegion ? (
              <>
                {/* Fixed width so the "Cropping…" label can't resize the button */}
                <button className="btn btn-primary" onClick={applyCrop} disabled={cropping} style={{ padding: "0.3rem 0.6rem", background: "#f59e0b", borderColor: "#f59e0b", minWidth: "6.5rem", whiteSpace: "nowrap" }} title="Save this region + its labels to <dataset>-cropped (X)">
                  {cropping ? "Cropping…" : "✂ Crop (X)"}
                </button>
                <span style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" }}>
                  {Math.round((cropRegion.x1 - cropRegion.x0) * 100)}%×{Math.round((cropRegion.y1 - cropRegion.y0) * 100)}%
                  {cropRegion.polygon ? ` · ⬠${cropRegion.polygon.length}` : ""}
                </span>
                <button className="btn btn-ghost" onClick={() => startCrop("rect")} style={{ padding: "0.3rem 0.5rem", fontSize: "0.8rem" }} title="Clear the region and drag a new rectangle">
                  ▭
                </button>
                <button className="btn btn-ghost" onClick={() => startCrop("polygon")} style={{ padding: "0.3rem 0.5rem", fontSize: "0.8rem" }} title="Clear the region and click a new polygon">
                  ⬠
                </button>
                <button className="btn btn-ghost" onClick={() => setCropRegion(null)} style={{ padding: "0.3rem 0.5rem", fontSize: "0.8rem" }} title="Remove the crop region">
                  Clear
                </button>
                {/* Always rendered — only its visibility toggles, so it never reflows */}
                <span
                  style={{ fontSize: "0.75rem", padding: "1px 6px", borderRadius: 3, background: "rgba(245,158,11,0.18)", color: "#b45309", fontWeight: 600, whiteSpace: "nowrap", visibility: cropExists ? "visible" : "hidden" }}
                  title="This image has already been written to the cropped dataset"
                >
                  ✓ cropped
                </span>
              </>
            ) : (
              <>
                <button className="btn btn-ghost" onClick={() => startCrop("rect")} style={{ padding: "0.3rem 0.5rem", fontSize: "0.8rem", whiteSpace: "nowrap" }} title="Drag a rectangular crop region, then press X to save crops">
                  ▭ Rect crop
                </button>
                <button className="btn btn-ghost" onClick={() => startCrop("polygon")} style={{ padding: "0.3rem 0.5rem", fontSize: "0.8rem", whiteSpace: "nowrap" }} title="Click a polygon crop region — area outside it is grey-filled, matching inference-time masking">
                  ⬠ Polygon crop
                </button>
              </>
            )}

            <span style={{ borderLeft: "1px solid var(--color-border)", height: "1.2rem" }} />

            {modelReady ? (
              <>
                <button className="btn btn-ghost" onClick={handleAutoDetect} disabled={detecting} style={{ padding: "0.3rem 0.5rem" }} title="Run model (A)">
                  {detecting ? "Detecting…" : "Auto-detect (A)"}
                </button>
                {predictions.length > 0 && (
                  <>
                    <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>{predictions.length} pred</span>
                    <button className="btn btn-primary" onClick={acceptAllPredictions} style={{ padding: "0.3rem 0.5rem" }} title="Accept all predictions (A)">Accept all</button>
                    <button className="btn btn-ghost" onClick={() => setPredictions([])} style={{ padding: "0.3rem 0.5rem" }}>Clear</button>
                  </>
                )}
                <input type="range" min={0.05} max={0.95} step={0.05} value={confidence} onChange={e => setConfidence(parseFloat(e.target.value))} style={{ width: 60 }} title={`Confidence: ${Math.round(confidence * 100)}%`} />
                <span style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", minWidth: "2rem" }}>{Math.round(confidence * 100)}%</span>
              </>
            ) : (
              <Link to="/settings" className="btn btn-ghost" style={{ padding: "0.3rem 0.5rem", fontSize: "0.8rem" }}>Load model</Link>
            )}
          </>
        )}

        {/* Fixed-size status slot — content changes, footprint never does */}
        <span
          style={{ flex: "0 0 1.1rem", width: "1.1rem", textAlign: "center", fontSize: "0.95rem", fontWeight: 700, lineHeight: 1, color: messageKind === "error" ? "var(--color-danger)" : "var(--color-success)" }}
          title={message ?? ""}
        >
          {message ? (messageKind === "error" ? "✗" : "✓") : " "}
        </span>
        {/* Errors also get a toast, absolutely positioned so it floats over the image */}
        {message && messageKind === "error" && (
          <div style={{ position: "absolute", top: "100%", right: "0.75rem", marginTop: 4, zIndex: 20, background: "var(--color-danger)", color: "#fff", padding: "0.25rem 0.6rem", borderRadius: 4, fontSize: "0.8rem", boxShadow: "0 2px 8px rgba(0,0,0,0.25)" }}>
            {message}
          </div>
        )}

        <span style={{ marginLeft: "auto", color: "var(--color-text-muted)", whiteSpace: "nowrap", fontSize: "0.75rem" }}>
          {isCls
            ? "← → nav · Space approve · Del delete · T tags"
            : "A auto-detect · ← → nav · Space approve · D del box · C cycle class · X save crop"
          }
        </span>
      </div>

      {showTags && (
        <div style={{ padding: "0.4rem 0.75rem", borderBottom: "1px solid var(--color-border)", display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.5rem", background: "var(--color-bg)" }}>
          {tagList.map(([k, v], i) => (
            <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem" }}>
              <input type="text" className="input" placeholder="key" value={k} onChange={e => updateTag(i, 0, e.target.value)} style={{ width: "90px", padding: "0.25rem 0.4rem" }} />
              <input type="text" className="input" placeholder="value" value={v} onChange={e => updateTag(i, 1, e.target.value)} style={{ width: "100px", padding: "0.25rem 0.4rem" }} />
              <button type="button" className="btn btn-ghost" style={{ padding: "0.2rem 0.4rem" }} onClick={() => removeTag(i)}>×</button>
            </span>
          ))}
          <button type="button" className="btn btn-ghost" style={{ padding: "0.25rem 0.5rem" }} onClick={addTag}>+ Tag</button>
          <button type="button" className="btn btn-primary" style={{ padding: "0.25rem 0.5rem" }} onClick={saveTags} disabled={saving}>Save tags</button>
        </div>
      )}

      {/* Image area */}
      <div style={{ flex: 1, minHeight: 0, display: "flex", justifyContent: "center", alignItems: "center", padding: "0.25rem", overflow: "hidden", background: "var(--color-border)" }}>
        {isCls ? (
          <img
            src={imageSrc(currentImage, datasetType)}
            alt={currentImage.name}
            style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
          />
        ) : (
          <BBoxCanvas
            imageUrl={api.imageUrl(currentImage.split, currentImage.name)}
            boxes={boxes}
            predictions={predictions}
            classNames={classNames}
            selectedIndex={selectedIndex}
            defaultClassId={defaultClassId}
            focusedClassId={state?.classId != null ? parseInt(state.classId, 10) : null}
            classColors={classColors}
            onSelect={setSelectedIndex}
            onBoxesChange={handleBoxesChange}
            onDoubleClickBox={cycleBoxClass}
            onAcceptPrediction={acceptPrediction}
            cropRegion={cropRegion}
            cropMode={cropMode}
            onCropRegionChange={handleCropRegionChange}
            fill
          />
        )}
      </div>

      {/* Bottom status */}
      <div style={{ height: "36px", flex: "0 0 36px", borderTop: "1px solid var(--color-border)", display: "flex", alignItems: "center", padding: "0.25rem 0.75rem", gap: "0.5rem", flexWrap: "nowrap", overflowX: "auto", background: "var(--color-surface)", fontSize: "0.85rem" }}>
        {isCls ? (
          <>
            <span style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>{currentImage.name}</span>
            {currentImage.fileSize != null && (
              <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>{formatFileSize(currentImage.fileSize)}</span>
            )}
            <span style={{ marginLeft: "auto", color: "var(--color-text-muted)", fontSize: "0.75rem" }}>
              ← → navigate · Space approve & next · Del delete
            </span>
          </>
        ) : (
          <>
            {selectedIndex !== null && boxes[selectedIndex] != null ? (() => {
              const selBox = boxes[selectedIndex]!;
              const area = selBox.w * selBox.h;
              return (
                <>
                  <span>Class:</span>
                  <select className="input" style={{ width: "auto", padding: "0.2rem 0.4rem" }} value={selBox.classId} onChange={e => handleSelectedClassChange(parseInt(e.target.value, 10))}>
                    {classes.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                  <span style={{
                    fontSize: "0.8rem", padding: "1px 6px", borderRadius: 3,
                    background: area < 0.001 ? "rgba(239,68,68,0.15)" : area < 0.005 ? "rgba(234,179,8,0.15)" : "rgba(0,0,0,0.06)",
                    color: area < 0.001 ? "var(--color-danger)" : area < 0.005 ? "#b45309" : "var(--color-text-muted)",
                    fontVariantNumeric: "tabular-nums",
                  }}>
                    area: {(area * 100).toFixed(3)}%
                  </span>
                  <button type="button" className="btn btn-ghost" style={{ padding: "0.2rem 0.4rem" }} onClick={() => { setBoxes(prev => prev.filter((_, i) => i !== selectedIndex)); setSelectedIndex(null); }}>Delete box</button>
                  <span style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>C to cycle class</span>
                </>
              );
            })() : (
              <span style={{ color: cropInfo ? "var(--color-success)" : "var(--color-text-muted)", fontSize: "0.8rem", whiteSpace: "nowrap" }}>
                {cropInfo ?? "Drag to draw · Click box to select/resize · D delete box · Space = approve & next"}
              </span>
            )}
            {classSort && (
              <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "var(--color-text-muted)", background: "rgba(0,0,0,0.06)", padding: "1px 6px", borderRadius: 3 }}>
                sorted by {classSort === "area_asc" ? "smallest area" : classSort === "area_desc" ? "largest area" : classSort === "size_asc" ? "smallest file" : "largest file"}
              </span>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
