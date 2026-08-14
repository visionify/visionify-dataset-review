import { useState, useCallback, useRef, useEffect } from "react";
import { getClassColor } from "@/classColors";
import type { BBox, CropPoint, CropRegion } from "@/types";
import type { PredictionBox } from "@/api";

const HANDLE_SIZE = 8;
type ResizeHandle = "nw" | "ne" | "sw" | "se" | "n" | "s" | "e" | "w";

/** Rectangle crop, or polygon crop with grey masking outside the polygon. */
export type CropMode = "rect" | "polygon" | null;

/** Must match CROP_KEEP_THRESHOLD in server/index.js — boxes below this survival ratio are dropped. */
const CROP_KEEP_THRESHOLD = 0.5;
const CROP_COLOR = "#f59e0b";
/** Click within this many screen px of the first vertex to close the polygon. */
const CLOSE_TOLERANCE = 10;

/** Shoelace area. Mirrors polygonArea() in server/index.js. */
function polygonArea(pts: CropPoint[]): number {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    a += pts[j]![0] * pts[i]![1] - pts[i]![0] * pts[j]![1];
  }
  return Math.abs(a) / 2;
}

/** Sutherland–Hodgman clip against an axis-aligned rect. Mirrors clipPolygonToRect() in server/index.js. */
function clipPolygonToRect(pts: CropPoint[], r: { x0: number; y0: number; x1: number; y1: number }): CropPoint[] {
  const edges: { inside: (p: CropPoint) => boolean; isect: (a: CropPoint, b: CropPoint) => CropPoint }[] = [
    { inside: p => p[0] >= r.x0, isect: (a, b) => [r.x0, a[1] + ((b[1] - a[1]) * (r.x0 - a[0])) / (b[0] - a[0])] },
    { inside: p => p[0] <= r.x1, isect: (a, b) => [r.x1, a[1] + ((b[1] - a[1]) * (r.x1 - a[0])) / (b[0] - a[0])] },
    { inside: p => p[1] >= r.y0, isect: (a, b) => [a[0] + ((b[0] - a[0]) * (r.y0 - a[1])) / (b[1] - a[1]), r.y0] },
    { inside: p => p[1] <= r.y1, isect: (a, b) => [a[0] + ((b[0] - a[0]) * (r.y1 - a[1])) / (b[1] - a[1]), r.y1] },
  ];
  let out = pts;
  for (const { inside, isect } of edges) {
    const input = out;
    out = [];
    for (let i = 0, j = input.length - 1; i < input.length; j = i++) {
      const cur = input[i]!, prev = input[j]!;
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

/** Bounding box of a polygon, as a CropRegion carrying the polygon itself. */
export function regionFromPolygon(poly: CropPoint[]): CropRegion {
  const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys), polygon: poly };
}

/**
 * Fraction of a box that survives the crop (1 = fully visible, 0 = gone).
 * For a polygon crop this is the overlap with the *polygon*, not its bounding
 * box — area outside the polygon is grey-filled, so it isn't visible content.
 */
function cropKeepRatio(b: BBox, crop: CropRegion): number {
  const area = b.w * b.h;
  if (area <= 0) return 0;
  const box = { x0: b.x - b.w / 2, y0: b.y - b.h / 2, x1: b.x + b.w / 2, y1: b.y + b.h / 2 };
  if (crop.polygon && crop.polygon.length >= 3) {
    return polygonArea(clipPolygonToRect(crop.polygon, box)) / area;
  }
  const c = normalizeCrop(crop);
  const iw = Math.min(box.x1, c.x1) - Math.max(box.x0, c.x0);
  const ih = Math.min(box.y1, c.y1) - Math.max(box.y0, c.y0);
  if (iw <= 0 || ih <= 0) return 0;
  return (iw * ih) / area;
}

function normalizeCrop(c: CropRegion): CropRegion {
  return {
    x0: Math.min(c.x0, c.x1), y0: Math.min(c.y0, c.y1),
    x1: Math.max(c.x0, c.x1), y1: Math.max(c.y0, c.y1),
    ...(c.polygon ? { polygon: c.polygon } : {}),
  };
}

export interface BBoxCanvasProps {
  imageUrl: string;
  boxes: BBox[];
  predictions?: PredictionBox[];
  classNames: Record<number, string>;
  selectedIndex: number | null;
  defaultClassId: number;
  focusedClassId?: number | null;
  classColors?: Record<number, string>;
  onSelect: (i: number | null) => void;
  onBoxesChange: (boxes: BBox[]) => void;
  onDoubleClickBox?: () => void;
  onAcceptPrediction?: (index: number) => void;
  maxHeight?: string;
  fill?: boolean;
  /** Persistent crop rectangle overlay (normalized). Null = none set. */
  cropRegion?: CropRegion | null;
  /** Regions already pinned via "＋ Region" — drawn read-only, and counted when
   *  deciding whether a box survives the crop. */
  extraRegions?: CropRegion[];
  /** "rect" = drag a rectangle; "polygon" = click vertices. Null = normal annotation. */
  cropMode?: CropMode;
  onCropRegionChange?: (r: CropRegion | null) => void;
}

export function BBoxCanvas({
  imageUrl,
  boxes,
  predictions,
  classNames,
  selectedIndex,
  defaultClassId,
  focusedClassId,
  classColors,
  onSelect,
  onBoxesChange,
  onDoubleClickBox,
  onAcceptPrediction,
  maxHeight = "calc(100vh - 140px)",
  fill = false,
  cropRegion = null,
  extraRegions = [],
  cropMode = null,
  onCropRegionChange,
}: BBoxCanvasProps) {
  const focus = focusedClassId ?? null;
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ startX: number; startY: number; currentX: number; currentY: number; kind: "box" | "crop" } | null>(null);
  const [resize, setResize] = useState<{ index: number; handle: ResizeHandle; startBox: BBox; startX: number; startY: number } | null>(null);
  const [cropResize, setCropResize] = useState<{ handle: ResizeHandle; startRegion: CropRegion; startX: number; startY: number } | null>(null);
  // Vertices placed so far while drawing a polygon, plus the live cursor for the rubber-band edge
  const [polyDraft, setPolyDraft] = useState<CropPoint[]>([]);
  const [polyCursor, setPolyCursor] = useState<CropPoint | null>(null);
  const [vertexDrag, setVertexDrag] = useState<number | null>(null);

  // Leaving polygon mode abandons any half-drawn polygon
  useEffect(() => {
    if (cropMode !== "polygon") { setPolyDraft([]); setPolyCursor(null); }
  }, [cropMode]);
  const [availableSize, setAvailableSize] = useState<{ w: number; h: number } | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    if (!fill) return;
    const el = wrapperRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0]!.contentRect;
      if (width && height) setAvailableSize({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fill]);

  const colorFor = useCallback(
    (classId: number) => {
      if (classColors && classColors[classId]) return classColors[classId];
      return getClassColor(classId, focus);
    },
    [classColors, focus]
  );

  const imgRef = useCallback((el: HTMLImageElement | null) => {
    if (!el) return;
    const update = () => {
      if (el.naturalWidth && el.naturalHeight)
        setNaturalSize({ w: el.naturalWidth, h: el.naturalHeight });
      const w = el.offsetWidth;
      const h = el.offsetHeight;
      if (w && h) setImgSize({ w, h });
    };
    if (el.complete) update();
    else el.addEventListener("load", update);
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener("load", update);
      ro.disconnect();
    };
  }, []);

  const getNorm = (e: React.MouseEvent) => {
    const imgEl = (e.currentTarget as HTMLElement).querySelector("img");
    if (!imgEl) return null;
    const imgRect = imgEl.getBoundingClientRect();
    return {
      x: (e.clientX - imgRect.left) / imgRect.width,
      y: (e.clientY - imgRect.top) / imgRect.height,
    };
  };

  /** Index of the smallest box containing (normX, normY), or -1 if none. Prefer inner/smaller bbox when nested. */
  const hitSmallestContaining = (normX: number, normY: number): number => {
    const containing = boxes
      .map((b, i) => ({ b, i }))
      .filter(({ b }) => {
        const bx = b.x - b.w / 2;
        const by = b.y - b.h / 2;
        return normX >= bx && normX <= bx + b.w && normY >= by && normY <= by + b.h;
      });
    return containing.length === 0 ? -1 : containing.reduce((best, cur) => (cur.b.w * cur.b.h < best.b.w * best.b.h ? cur : best)).i;
  };

  const hitHandle = (norm: { x: number; y: number }, b: BBox, imgW: number, imgH: number): ResizeHandle | null => {
    const x = (b.x - b.w / 2) * imgW;
    const y = (b.y - b.h / 2) * imgH;
    const bw = b.w * imgW;
    const bh = b.h * imgH;
    const px = norm.x * imgW;
    const py = norm.y * imgH;
    const h = Math.max(HANDLE_SIZE / 2, 6);
    const corners: [ResizeHandle, number, number][] = [
      ["nw", x, y], ["ne", x + bw, y], ["sw", x, y + bh], ["se", x + bw, y + bh],
      ["n", x + bw / 2, y], ["s", x + bw / 2, y + bh], ["e", x + bw, y + bh / 2], ["w", x, y + bh / 2],
    ];
    for (const [handle, hx, hy] of corners) {
      if (Math.abs(px - hx) <= h && Math.abs(py - hy) <= h) return handle;
    }
    return null;
  };

  /** Handle hit-test against an axis-aligned rect given in pixel coords. */
  const hitRectHandle = (px: number, py: number, x: number, y: number, w: number, h: number): ResizeHandle | null => {
    const tol = Math.max(HANDLE_SIZE / 2, 7);
    const points: [ResizeHandle, number, number][] = [
      ["nw", x, y], ["ne", x + w, y], ["sw", x, y + h], ["se", x + w, y + h],
      ["n", x + w / 2, y], ["s", x + w / 2, y + h], ["e", x + w, y + h / 2], ["w", x, y + h / 2],
    ];
    for (const [handle, hx, hy] of points) {
      if (Math.abs(px - hx) <= tol && Math.abs(py - hy) <= tol) return handle;
    }
    return null;
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    if (!imgSize) return;
    const norm = getNorm(e);
    if (!norm || norm.x < 0 || norm.x > 1 || norm.y < 0 || norm.y > 1) return;

    // Drawing a fresh crop region takes over the canvas entirely
    if (cropMode === "rect") {
      onSelect(null);
      setDrag({ startX: norm.x, startY: norm.y, currentX: norm.x, currentY: norm.y, kind: "crop" });
      return;
    }

    if (cropMode === "polygon") {
      onSelect(null);
      const first = polyDraft[0];
      // Clicking back on the first vertex closes the polygon
      if (first && polyDraft.length >= 3) {
        const dx = (norm.x - first[0]) * imgSize.w;
        const dy = (norm.y - first[1]) * imgSize.h;
        if (Math.hypot(dx, dy) <= CLOSE_TOLERANCE) {
          onCropRegionChange?.(regionFromPolygon(polyDraft));
          setPolyDraft([]);
          setPolyCursor(null);
          return;
        }
      }
      setPolyDraft(prev => [...prev, [norm.x, norm.y]]);
      return;
    }

    // Dragging a vertex of an existing polygon
    if (cropRegion?.polygon && cropRegion.polygon.length >= 3) {
      const hit = cropRegion.polygon.findIndex(([vx, vy]) =>
        Math.hypot((norm.x - vx) * imgSize.w, (norm.y - vy) * imgSize.h) <= HANDLE_SIZE
      );
      if (hit >= 0) { setVertexDrag(hit); return; }
    }

    // Crop-region resize handles win over box interaction (rectangle crops only)
    if (cropRegion && !cropRegion.polygon) {
      const c = normalizeCrop(cropRegion);
      const handle = hitRectHandle(
        norm.x * imgSize.w, norm.y * imgSize.h,
        c.x0 * imgSize.w, c.y0 * imgSize.h, (c.x1 - c.x0) * imgSize.w, (c.y1 - c.y0) * imgSize.h
      );
      if (handle) {
        setCropResize({ handle, startRegion: c, startX: e.clientX, startY: e.clientY });
        return;
      }
    }

    if (selectedIndex !== null && boxes[selectedIndex]) {
      const handle = hitHandle(norm, boxes[selectedIndex], imgSize.w, imgSize.h);
      if (handle) {
        setResize({ index: selectedIndex, handle, startBox: { ...boxes[selectedIndex] }, startX: e.clientX, startY: e.clientY });
        return;
      }
    }

    const hit = hitSmallestContaining(norm.x, norm.y);
    if (hit >= 0) {
      onSelect(hit);
      return;
    }
    onSelect(null);
    setDrag({ startX: norm.x, startY: norm.y, currentX: norm.x, currentY: norm.y, kind: "box" });
  };

  const getNormFromClient = useCallback((clientX: number, clientY: number) => {
    const imgEl = containerRef.current?.querySelector("img");
    if (!imgEl) return null;
    const r = imgEl.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (clientX - r.left) / r.width)),
      y: Math.max(0, Math.min(1, (clientY - r.top) / r.height)),
    };
  }, []);

  const applyCropResize = useCallback((clientX: number, clientY: number) => {
    if (!cropResize) return;
    const imgEl = containerRef.current?.querySelector("img");
    if (!imgEl) return;
    const r = imgEl.getBoundingClientRect();
    const dx = (clientX - cropResize.startX) / r.width;
    const dy = (clientY - cropResize.startY) / r.height;
    const { handle, startRegion } = cropResize;
    const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
    let { x0, y0, x1, y1 } = startRegion;
    if (handle.includes("w")) x0 = clamp01(Math.min(x0 + dx, x1 - 0.01));
    if (handle.includes("e")) x1 = clamp01(Math.max(x1 + dx, x0 + 0.01));
    if (handle.includes("n")) y0 = clamp01(Math.min(y0 + dy, y1 - 0.01));
    if (handle.includes("s")) y1 = clamp01(Math.max(y1 + dy, y0 + 0.01));
    onCropRegionChange?.({ x0, y0, x1, y1 });
  }, [cropResize, onCropRegionChange]);

  /** Move one polygon vertex; the region's bbox is re-derived from the new shape. */
  const moveVertex = useCallback((index: number, clientX: number, clientY: number) => {
    const poly = cropRegion?.polygon;
    if (!poly) return;
    const n = getNormFromClient(clientX, clientY);
    if (!n) return;
    const next = poly.map((p, i): CropPoint => (i === index ? [n.x, n.y] : p));
    onCropRegionChange?.(regionFromPolygon(next));
  }, [cropRegion, getNormFromClient, onCropRegionChange]);

  const handleMouseMove = (e: React.MouseEvent) => {
    const clamped = getNormFromClient(e.clientX, e.clientY);
    if (!clamped) return;

    if (cropMode === "polygon") { setPolyCursor([clamped.x, clamped.y]); return; }
    if (vertexDrag !== null) { moveVertex(vertexDrag, e.clientX, e.clientY); return; }
    if (cropResize !== null) { applyCropResize(e.clientX, e.clientY); return; }

    if (resize !== null && imgSize) {
      const { index, handle, startBox, startX, startY } = resize;
      const imgEl = containerRef.current?.querySelector("img");
      if (!imgEl) return;
      const imgRect = imgEl.getBoundingClientRect();
      const dx = (e.clientX - startX) / imgRect.width;
      const dy = (e.clientY - startY) / imgRect.height;
      let { x, y, w, h } = startBox;
      const xMin = x - w / 2, xMax = x + w / 2, yMin = y - h / 2, yMax = y + h / 2;
      if (handle.includes("w")) { const n = Math.min(xMin + dx, xMax - 0.01); x = (n + xMax) / 2; w = xMax - n; }
      if (handle.includes("e")) { const n = Math.max(xMax + dx, xMin + 0.01); x = (xMin + n) / 2; w = n - xMin; }
      if (handle.includes("n")) { const n = Math.min(yMin + dy, yMax - 0.01); y = (n + yMax) / 2; h = yMax - n; }
      if (handle.includes("s")) { const n = Math.max(yMax + dy, yMin + 0.01); y = (yMin + n) / 2; h = n - yMin; }
      const next = boxes.map((b, i) => (i === index ? { ...b, x, y, w: Math.max(0.01, w), h: Math.max(0.01, h) } : b));
      onBoxesChange(next);
      return;
    }

    if (drag) {
      setDrag(d => d ? { ...d, currentX: clamped.x, currentY: clamped.y } : null);
    }
  };

  const finishDrag = useCallback(() => {
    setResize(null);
    setCropResize(null);
    setVertexDrag(null);
    setDrag(prev => {
      if (!prev || !imgSize) return null;
      const x0 = prev.startX, y0 = prev.startY, x1 = prev.currentX, y1 = prev.currentY;
      const xMin = Math.max(0, Math.min(x0, x1));
      const xMax = Math.min(1, Math.max(x0, x1));
      const yMin = Math.max(0, Math.min(y0, y1));
      const yMax = Math.min(1, Math.max(y0, y1));
      const w = xMax - xMin, h = yMax - yMin;
      if (prev.kind === "crop") {
        if (w >= 0.02 && h >= 0.02) onCropRegionChange?.({ x0: xMin, y0: yMin, x1: xMax, y1: yMax });
        return null;
      }
      if (w >= 0.01 && h >= 0.01) {
        const newBox: BBox = { classId: defaultClassId, x: xMin + w / 2, y: yMin + h / 2, w, h };
        onBoxesChange([...boxes, newBox]);
        onSelect(boxes.length);
      }
      return null;
    });
  }, [imgSize, defaultClassId, boxes, onBoxesChange, onSelect, onCropRegionChange]);

  const handleMouseUp = (_e: React.MouseEvent) => { finishDrag(); };

  // Track mouse even when it leaves the canvas, so edge-of-image drawing works
  useEffect(() => {
    if (!drag && !resize && !cropResize && vertexDrag === null) return;
    const onMove = (e: MouseEvent) => {
      if (vertexDrag !== null) { moveVertex(vertexDrag, e.clientX, e.clientY); return; }
      if (cropResize) { applyCropResize(e.clientX, e.clientY); return; }
      const clamped = getNormFromClient(e.clientX, e.clientY);
      if (!clamped) return;
      if (drag) setDrag(d => d ? { ...d, currentX: clamped.x, currentY: clamped.y } : null);
    };
    const onUp = () => { finishDrag(); };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [drag, resize, cropResize, vertexDrag, moveVertex, applyCropResize, getNormFromClient, finishDrag]);

  const handleDoubleClick = (e: React.MouseEvent) => {
    if (!imgSize) return;
    // While drawing, a double-click closes the polygon instead of cycling a class
    if (cropMode === "polygon") {
      if (polyDraft.length >= 3) {
        onCropRegionChange?.(regionFromPolygon(polyDraft));
        setPolyDraft([]);
        setPolyCursor(null);
      }
      return;
    }
    const norm = getNorm(e);
    if (!norm) return;
    const hit = hitSmallestContaining(norm.x, norm.y);
    if (hit >= 0) {
      onSelect(hit);
      onDoubleClickBox?.();
    }
  };

  let fittedW: number | undefined;
  let fittedH: number | undefined;
  if (fill && availableSize && naturalSize) {
    const scale = Math.min(availableSize.w / naturalSize.w, availableSize.h / naturalSize.h);
    fittedW = Math.floor(naturalSize.w * scale);
    fittedH = Math.floor(naturalSize.h * scale);
  }

  const innerStyle: React.CSSProperties = {
    position: "relative", display: "inline-block", maxWidth: "100%", maxHeight: "100%",
    cursor: cropMode ? "crosshair" : undefined,
    // Suppress text-selection flicker while click-placing polygon vertices
    userSelect: cropMode === "polygon" ? "none" : undefined,
  };

  const imgStyle: React.CSSProperties = fill && fittedW && fittedH
    ? { width: fittedW, height: fittedH, display: "block", userSelect: "none", pointerEvents: "none" }
    : { maxWidth: "100%", maxHeight, objectFit: "contain", display: "block", userSelect: "none", pointerEvents: "none" };

  const wrapperStyle: React.CSSProperties = fill
    ? { width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", minHeight: 0 }
    : {};

  return (
    <div ref={wrapperRef} style={wrapperStyle}>
      <div
        ref={containerRef}
        style={innerStyle}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        onDragStart={e => e.preventDefault()}
      >
        <img ref={imgRef} src={imageUrl} alt="" style={imgStyle} draggable={false} />
        {imgSize && (
          <svg
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
            viewBox={`0 0 ${imgSize.w} ${imgSize.h}`}
          >
            {boxes.map((b, i) => {
              const x = (b.x - b.w / 2) * imgSize.w;
              const y = (b.y - b.h / 2) * imgSize.h;
              const bw = b.w * imgSize.w;
              const bh = b.h * imgSize.h;
              const selected = i === selectedIndex;
              const color = colorFor(b.classId);
              const label = String(classNames[b.classId] ?? b.classId);
              const labelFontSize = 14;
              const labelPadX = 4;
              const labelPadY = 2;
              const labelW = label.length * labelFontSize * 0.62 + labelPadX * 2;
              const labelH = labelFontSize + labelPadY * 2;
              // Preview which boxes survive the pending crop
              // Each region is exported as its own crop, so a box is kept as long
              // as it survives in at least one of them.
              const allRegions = cropRegion ? [...extraRegions, cropRegion] : extraRegions;
              const survives = allRegions.length === 0
                || allRegions.some(r => cropKeepRatio(b, r) >= CROP_KEEP_THRESHOLD);
              return (
                <g key={i} opacity={survives ? 1 : 0.22}>
                  <rect x={x} y={y} width={bw} height={bh} fill="none" stroke={color} strokeWidth={selected ? 4 : 2} />
                  <rect x={x} y={y - labelH} width={labelW} height={labelH} fill={color} rx={2} />
                  <text x={x + labelPadX} y={y - labelPadY - 1} fill="#000" fontSize={labelFontSize} fontWeight={600} dominantBaseline="auto">
                    {label}
                  </text>
                  {selected && (
                    <>
                      {(["nw", "ne", "sw", "se", "n", "s", "e", "w"] as const).map(handle => {
                        const hx = handle.includes("e") ? x + bw : handle.includes("w") ? x : x + bw / 2;
                        const hy = handle.includes("s") ? y + bh : handle.includes("n") ? y : y + bh / 2;
                        return <circle key={handle} cx={hx} cy={hy} r={HANDLE_SIZE / 2} fill="white" stroke={color} strokeWidth={2} />;
                      })}
                    </>
                  )}
                </g>
              );
            })}
            {predictions?.map((p, i) => {
              const x = (p.x - p.w / 2) * imgSize.w;
              const y = (p.y - p.h / 2) * imgSize.h;
              const bw = p.w * imgSize.w;
              const bh = p.h * imgSize.h;
              const color = colorFor(p.classId);
              const label = `${p.className} ${Math.round(p.confidence * 100)}%`;
              const fontSize = 12;
              const padX = 3, padY = 2;
              const lw = label.length * fontSize * 0.58 + padX * 2;
              const lh = fontSize + padY * 2;
              return (
                <g key={`pred-${i}`} style={{ cursor: "pointer", pointerEvents: "all" }} onClick={() => onAcceptPrediction?.(i)}>
                  <rect x={x} y={y} width={bw} height={bh} fill={color} fillOpacity={0.08} stroke={color} strokeWidth={2} strokeDasharray="6 3" />
                  <rect x={x} y={y - lh} width={lw} height={lh} fill={color} fillOpacity={0.7} rx={2} />
                  <text x={x + padX} y={y - padY - 1} fill="#000" fontSize={fontSize} fontWeight={600} dominantBaseline="auto">{label}</text>
                </g>
              );
            })}
            {/* Regions already pinned with "＋ Region": drawn read-only, in a distinct
                colour, so it is obvious which one is being edited. */}
            {extraRegions.map((r, i) => {
              const c = normalizeCrop(r);
              const x = c.x0 * imgSize.w, y = c.y0 * imgSize.h;
              const w = (c.x1 - c.x0) * imgSize.w, h = (c.y1 - c.y0) * imgSize.h;
              const pts = r.polygon && r.polygon.length >= 3
                ? r.polygon.map(([px, py]) => [px * imgSize.w, py * imgSize.h] as const)
                : null;
              return (
                <g key={`pinned-${i}`} opacity={0.85}>
                  <rect x={x} y={y} width={w} height={h} fill="none" stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 5" />
                  {pts && <polygon points={pts.map(([px, py]) => `${px},${py}`).join(" ")} fill="none" stroke="#3b82f6" strokeWidth={2} />}
                  <text x={x + 4} y={y + 14} fill="#3b82f6" fontSize={12} fontWeight={700}>{r.name || `zone${i + 1}`}</text>
                </g>
              );
            })}
            {cropRegion?.polygon && cropRegion.polygon.length >= 3 && !(drag?.kind === "crop") && (() => {
              const poly = cropRegion.polygon!;
              const pts = poly.map(([px, py]) => [px * imgSize.w, py * imgSize.h] as const);
              const c = normalizeCrop(cropRegion);
              const x = c.x0 * imgSize.w, y = c.y0 * imgSize.h;
              const w = (c.x1 - c.x0) * imgSize.w, h = (c.y1 - c.y0) * imgSize.h;
              // Outer rect + polygon hole, even-odd: shades only the area that
              // will be grey-filled on export.
              const inner = pts.map(([px, py], i) => `${i === 0 ? "M" : "L"} ${px} ${py}`).join(" ");
              const d = `M 0 0 L ${imgSize.w} 0 L ${imgSize.w} ${imgSize.h} L 0 ${imgSize.h} Z ${inner} Z`;
              return (
                <g>
                  <path d={d} fillRule="evenodd" fill="#000" fillOpacity={0.55} />
                  {/* The bounding box the image is actually cropped to */}
                  <rect x={x} y={y} width={w} height={h} fill="none" stroke={CROP_COLOR} strokeWidth={1} strokeDasharray="3 5" opacity={0.7} />
                  <polygon points={pts.map(([px, py]) => `${px},${py}`).join(" ")} fill="none" stroke={CROP_COLOR} strokeWidth={2} />
                  {pts.map(([px, py], i) => (
                    <circle key={i} cx={px} cy={py} r={HANDLE_SIZE / 2} fill={CROP_COLOR} stroke="#fff" strokeWidth={1.5} />
                  ))}
                </g>
              );
            })()}
            {cropRegion && !cropRegion.polygon && !(drag?.kind === "crop") && (() => {
              const c = normalizeCrop(cropRegion);
              const x = c.x0 * imgSize.w, y = c.y0 * imgSize.h;
              const w = (c.x1 - c.x0) * imgSize.w, h = (c.y1 - c.y0) * imgSize.h;
              return (
                <g>
                  {/* Dim everything outside the crop */}
                  <g fill="#000" fillOpacity={0.55}>
                    <rect x={0} y={0} width={imgSize.w} height={y} />
                    <rect x={0} y={y + h} width={imgSize.w} height={Math.max(0, imgSize.h - y - h)} />
                    <rect x={0} y={y} width={x} height={h} />
                    <rect x={x + w} y={y} width={Math.max(0, imgSize.w - x - w)} height={h} />
                  </g>
                  <rect x={x} y={y} width={w} height={h} fill="none" stroke={CROP_COLOR} strokeWidth={2} strokeDasharray="8 4" />
                  {(["nw", "ne", "sw", "se", "n", "s", "e", "w"] as const).map(handle => {
                    const hx = handle.includes("e") ? x + w : handle.includes("w") ? x : x + w / 2;
                    const hy = handle.includes("s") ? y + h : handle.includes("n") ? y : y + h / 2;
                    return <rect key={handle} x={hx - HANDLE_SIZE / 2} y={hy - HANDLE_SIZE / 2} width={HANDLE_SIZE} height={HANDLE_SIZE} fill={CROP_COLOR} stroke="#fff" strokeWidth={1.5} />;
                  })}
                </g>
              );
            })()}
            {/* Polygon being drawn: placed vertices + rubber-band edge to the cursor */}
            {cropMode === "polygon" && polyDraft.length > 0 && (() => {
              const pts = polyDraft.map(([px, py]) => [px * imgSize.w, py * imgSize.h] as const);
              const path = pts.map(([px, py]) => `${px},${py}`).join(" ");
              const last = pts[pts.length - 1]!;
              const first = pts[0]!;
              return (
                <g>
                  <polyline points={path} fill={CROP_COLOR} fillOpacity={0.12} stroke={CROP_COLOR} strokeWidth={2} />
                  {polyCursor && (
                    <line x1={last[0]} y1={last[1]} x2={polyCursor[0] * imgSize.w} y2={polyCursor[1] * imgSize.h} stroke={CROP_COLOR} strokeWidth={2} strokeDasharray="5 4" />
                  )}
                  {pts.map(([px, py], i) => (
                    <circle key={i} cx={px} cy={py} r={HANDLE_SIZE / 2} fill={i === 0 ? "#fff" : CROP_COLOR} stroke={CROP_COLOR} strokeWidth={2} />
                  ))}
                  {polyDraft.length >= 3 && (
                    <circle cx={first[0]} cy={first[1]} r={CLOSE_TOLERANCE} fill="none" stroke={CROP_COLOR} strokeWidth={1} strokeDasharray="2 2" />
                  )}
                </g>
              );
            })()}
            {drag && (
              <rect
                x={Math.min(drag.startX, drag.currentX) * imgSize.w}
                y={Math.min(drag.startY, drag.currentY) * imgSize.h}
                width={Math.abs(drag.currentX - drag.startX) * imgSize.w}
                height={Math.abs(drag.currentY - drag.startY) * imgSize.h}
                fill="none" stroke={drag.kind === "crop" ? CROP_COLOR : "cyan"} strokeWidth={2} strokeDasharray="4 2"
              />
            )}
          </svg>
        )}
      </div>
    </div>
  );
}
