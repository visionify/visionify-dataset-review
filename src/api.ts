import type { BBox, ImageTags, ValidationCheck, ImageItem, ClassItem } from "./types";

const LOCAL_STORAGE_KEY = "dataset-review-server-url";
const DEFAULT_SERVER = "";

export function getServerUrl(): string {
  return localStorage.getItem(LOCAL_STORAGE_KEY) || DEFAULT_SERVER;
}

export function setServerUrl(url: string) {
  const trimmed = url.replace(/\/+$/, "").trim();
  if (trimmed) {
    localStorage.setItem(LOCAL_STORAGE_KEY, trimmed);
  } else {
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  }
}

function base(): string {
  return getServerUrl() + "/api";
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(base() + path);
  if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
  return r.json();
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(base() + path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(base() + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
  return r.json();
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(base() + path, {
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
  classes: ClassItem[];
  config: { train: string | null; val: string | null; test: string | null; names: Record<number, string> } | null;
  totalImages: number;
  reviewedCount?: number;
  splitCounts?: Record<string, number>;
  missingLabelsCount?: number;
  emptyLabelsCount?: number;
}

export const api = {
  getConfig: () => get<{ datasetPath: string; configured: boolean }>("/config"),
  setConfig: (path: string) => post<{ ok: boolean }>("/config", { path }),
  getSummary: () => get<DatasetSummary>("/dataset/summary"),
  getValidation: () => get<{ checks: ValidationCheck[] }>("/validation"),
  getImages: (opts: { split?: string; page?: number; limit?: number; reviewed?: "yes" | "no"; tagType?: string; tag?: string }) => {
    const p = new URLSearchParams();
    if (opts.split) p.set("split", opts.split);
    if (opts.page != null) p.set("page", String(opts.page));
    if (opts.limit != null) p.set("limit", String(opts.limit));
    if (opts.reviewed) p.set("reviewed", opts.reviewed);
    if (opts.tagType) p.set("tagType", opts.tagType);
    if (opts.tag) p.set("tag", opts.tag);
    return get<{ images: ImageItem[]; total: number }>(`/images?${p}`);
  },
  getAutoTags: () => get<{ tasks: TagGroup[]; months: TagGroup[]; cameras: TagGroup[] }>("/auto-tags"),
  getReviewed: () => get<{ reviewed: string[] }>("/reviewed"),
  setReviewed: (split: string, base: string, reviewed: boolean) =>
    patch<{ reviewed: string[] }>("/reviewed", { split, base, reviewed }),
  getClassColors: () => get<Record<number, string>>("/class-colors"),
  setClassColors: (colors: Record<number, string>) => put<Record<number, string>>("/class-colors", colors),
  deleteImage: (split: string, name: string) => {
    return fetch(`${getServerUrl()}/api/images/${encodeURIComponent(split)}/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) => {
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
  imageUrl: (split: string, name: string) => `${getServerUrl()}/api/images/${encodeURIComponent(split)}/${encodeURIComponent(name)}`,
  assetUrl: (relPath: string) => `${getServerUrl()}/dataset-asset/${relPath.replace(/^\//, "")}`,
  getAnnotations: (split: string, base: string) =>
    get<BBox[]>(`/annotations/${encodeURIComponent(split)}/${encodeURIComponent(base)}`),
  saveAnnotations: (split: string, base: string, boxes: BBox[]) =>
    put<{ ok: boolean }>(`/annotations/${encodeURIComponent(split)}/${encodeURIComponent(base)}`, boxes),
  getTags: (split: string, base: string) =>
    get<ImageTags>(`/tags/${encodeURIComponent(split)}/${encodeURIComponent(base)}`),
  saveTags: (split: string, base: string, tags: ImageTags) =>
    put<{ ok: boolean }>(`/tags/${encodeURIComponent(split)}/${encodeURIComponent(base)}`, tags),
  patchMetadata: (data: Record<string, unknown>) => patch<Record<string, unknown>>("/metadata", data),
  fixDuplicateLabels: () => post<{ ok: boolean; filesFixed: number; linesRemoved: number }>("/validation/fix-duplicates", {}),
  deleteMissingLabelImages: () => post<{ ok: boolean; deleted: number }>("/validation/delete-missing-labels", {}),
  deleteDuplicateImages: () => post<{ ok: boolean; deleted: number }>("/validation/delete-duplicate-images", {}),
  deleteSmallBboxes: () => post<{ ok: boolean; removed: number; filesUpdated: number }>("/validation/delete-small-bboxes", {}),

  // Inference
  inferenceHealth: () => get<{ status: string; model_loaded: boolean; model_path: string | null }>("/inference/health"),
  inferenceLoad: (modelPath: string) => post<{ ok: boolean; model_path: string; classes: Record<number, string> }>("/inference/load", { model_path: modelPath }),
  inferencePredict: (split: string, name: string, confidence?: number, iou?: number) =>
    post<{ boxes: PredictionBox[]; count: number }>("/inference/predict", { split, name, confidence, iou }),
  inferenceUnload: () => post<{ ok: boolean }>("/inference/unload", {}),
};

export function imageBase(name: string): string {
  const lastDot = name.lastIndexOf(".");
  return lastDot > 0 ? name.slice(0, lastDot) : name;
}
