import type { BBox, CropRegion, ImageTags, ValidationCheck, ImageItem, ClassItem } from "./types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
  return r.json();
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
  return r.json();
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
  return r.json();
}

export interface TagGroup {
  name: string;
  count: number;
}

export interface PredictionBox {
  classId: number;
  className: string;
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
}

export interface DatasetSummary {
  configured: boolean;
  type?: "detection" | "classification";
  classes: ClassItem[];
  config: { train: string | null; val: string | null; test: string | null; names: Record<number, string>; classFolders?: string[] } | null;
  totalImages: number;
  reviewedCount?: number;
  splitCounts?: Record<string, number>;
  classCounts?: Record<string, number>;
  missingLabelsCount?: number;
  emptyLabelsCount?: number;
}

export const api = {
  getConfig: () => get<{ datasetPath: string; configured: boolean }>("/config"),
  setConfig: (path: string) => post<{ ok: boolean }>("/config", { path }),
  getSummary: () => get<DatasetSummary>("/dataset/summary"),
  getValidation: () => get<{ checks: ValidationCheck[] }>("/validation"),
  getImages: (opts: { split?: string; page?: number; limit?: number; reviewed?: "yes" | "no"; tagType?: string; tag?: string; classId?: number; sort?: string; className?: string }) => {
    const p = new URLSearchParams();
    if (opts.split) p.set("split", opts.split);
    if (opts.page != null) p.set("page", String(opts.page));
    if (opts.limit != null) p.set("limit", String(opts.limit));
    if (opts.reviewed) p.set("reviewed", opts.reviewed);
    if (opts.tagType) p.set("tagType", opts.tagType);
    if (opts.tag) p.set("tag", opts.tag);
    if (opts.classId != null) p.set("classId", String(opts.classId));
    if (opts.sort) p.set("sort", opts.sort);
    if (opts.className) p.set("className", opts.className);
    return get<{ images: ImageItem[]; total: number }>(`/images?${p}`);
  },
  getAutoTags: () => get<{ tasks: TagGroup[]; months: TagGroup[]; cameras: TagGroup[] }>("/auto-tags"),
  getReviewed: () => get<{ reviewed: string[] }>("/reviewed"),
  setReviewed: (split: string, base: string, reviewed: boolean, reviewKey?: string) =>
    patch<{ reviewed: string[] }>("/reviewed", { split, base, reviewed, ...(reviewKey ? { reviewKey } : {}) }),
  getClassColors: () => get<Record<number, string>>("/class-colors"),
  setClassColors: (colors: Record<number, string>) => put<Record<number, string>>("/class-colors", colors),
  deleteImage: (split: string, name: string) => {
    return fetch(`/api/images/${encodeURIComponent(split)}/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) => {
      if (!r.ok) throw new Error(r.statusText);
      return r.json();
    });
  },
  getClassImages: (classId: number, page: number, limit: number, sort?: string) => {
    const p = new URLSearchParams({ page: String(page), limit: String(limit) });
    if (sort) p.set("sort", sort);
    return get<{ images: ImageItem[]; total: number }>(`/class/${classId}/images?${p}`);
  },
  getClassSamples: (classId: number, limit?: number) =>
    get<{ samples: ImageItem[] }>(`/class/${classId}/samples?limit=${limit ?? 8}`),
  imageUrl: (split: string, name: string) => `/api/images/${encodeURIComponent(split)}/${encodeURIComponent(name)}`,
  assetUrl: (relPath: string) => `/dataset-asset/${relPath.replace(/^\//, "")}`,
  getAnnotations: (split: string, base: string) =>
    get<BBox[]>(`/annotations/${encodeURIComponent(split)}/${encodeURIComponent(base)}`),
  saveAnnotations: (split: string, base: string, boxes: BBox[]) =>
    put<{ ok: boolean }>(`/annotations/${encodeURIComponent(split)}/${encodeURIComponent(base)}`, boxes),
  getTags: (split: string, base: string) =>
    get<ImageTags>(`/tags/${encodeURIComponent(split)}/${encodeURIComponent(base)}`),
  saveTags: (split: string, base: string, tags: ImageTags) =>
    put<{ ok: boolean }>(`/tags/${encodeURIComponent(split)}/${encodeURIComponent(base)}`, tags),
  // Crop — export a sub-region to the sibling "<dataset>-cropped" dataset
  cropImage: (split: string, name: string, regions: CropRegion[]) =>
    post<{ ok: boolean; outputRoot: string; outImage: string; kept: number; dropped: number; width: number; height: number }>(
      "/crop", { split, name, regions }
    ),
  /** Crop every image in the dataset with these regions, in one request. */
  cropAll: (regions: CropRegion[], backgroundRatio?: number) =>
    post<{ ok: boolean; outputRoot: string; images: number; regions: number; cropped: number;
           kept: number; dropped: number; thinnedBackground: number; failureCount: number }>(
      "/crop/all", { regions, backgroundRatio }
    ),
  cropStatus: (split: string, name: string) =>
    get<{ exists: boolean; outputRoot: string | null }>(
      `/crop/status?split=${encodeURIComponent(split)}&name=${encodeURIComponent(name)}`
    ),

  patchMetadata: (data: Record<string, unknown>) => patch<Record<string, unknown>>("/metadata", data),
  fixDuplicateLabels: () => post<{ ok: boolean; filesFixed: number; linesRemoved: number }>("/validation/fix-duplicates", {}),
  deleteMissingLabelImages: () => post<{ ok: boolean; deleted: number }>("/validation/delete-missing-labels", {}),
  deleteDuplicateImages: () => post<{ ok: boolean; deleted: number }>("/validation/delete-duplicate-images", {}),
  deleteSmallBboxes: () => post<{ ok: boolean; removed: number; filesUpdated: number }>("/validation/delete-small-bboxes", {}),

  // Classification-specific
  classificationDeleteImages: (imageRels: string[]) =>
    post<{ ok: boolean; deleted: number }>("/classification/delete-images", { imageRels }),
  classificationMoveImages: (imageRels: string[], targetClassName: string) =>
    post<{ ok: boolean; moved: number }>("/classification/move", { imageRels, targetClassName }),

  // Inference
  inferenceHealth: () => get<{ status: string; model_loaded: boolean; model_path: string | null }>("/inference/health"),
  /** Fine-tune on the reviewed annotations, then predict with the result. */
  inferenceFinetune: (epochs?: number) =>
    post<{ ok: boolean; started: boolean }>("/inference/finetune", epochs ? { epochs } : {}),
  inferenceFinetuneStatus: () =>
    get<{ running: boolean; stage: string; message: string; epochs_done: number;
          epochs_total: number; elapsed: number; model_path: string | null }>("/inference/finetune/status"),
  inferenceLoad: (modelPath: string) => post<{ ok: boolean; model_path: string; classes: Record<number, string> }>("/inference/load", { model_path: modelPath }),
  inferencePredict: (split: string, name: string, confidence?: number, iou?: number) =>
    post<{ boxes: PredictionBox[]; count: number }>("/inference/predict", { split, name, confidence, iou }),
  inferenceUnload: () => post<{ ok: boolean }>("/inference/unload", {}),
};

/** Get image src URL — works for both detection and classification */
export function imageSrc(img: { split: string; name: string; imageRel: string }, datasetType?: string): string {
  if (datasetType === "classification") return `/dataset-asset/${img.imageRel}`;
  return `/api/images/${encodeURIComponent(img.split)}/${encodeURIComponent(img.name)}`;
}

export function imageBase(name: string): string {
  const lastDot = name.lastIndexOf(".");
  return lastDot > 0 ? name.slice(0, lastDot) : name;
}
