export interface ClassItem {
  id: number;
  name: string;
}

export interface ImageItem {
  split: string;
  name: string;
  relPath: string;
  imageRel: string;
  bboxArea?: number;
  className?: string;
  fileSize?: number;
}

export interface BBox {
  classId: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Normalized (0–1) polygon vertex. */
export type CropPoint = [number, number];

/**
 * Normalized (0–1) crop region, persisted across images while sweeping a scene.
 *
 * `x0..y1` is always the region's bounding box — the rectangle the image is
 * actually cropped to. When `polygon` is present the bbox is derived from it and
 * pixels inside the bbox but outside the polygon are grey-filled on export, so
 * training data matches what the model sees at inference time.
 */
export interface CropRegion {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  polygon?: CropPoint[];
}

export interface ImageTags {
  day?: boolean;
  night?: boolean;
  camera?: string;
  client?: string;
  [key: string]: string | boolean | undefined;
}

export interface DatasetConfig {
  datasetPath: string;
  configured: boolean;
}

export interface DatasetResponse {
  configured: boolean;
  classes: ClassItem[];
  images: ImageItem[];
  samplesByClass: Record<number | string, ImageItem[]>;
  imageClassIds?: Record<string, number[]>;
  config: { train: string; val: string; test: string | null; names: Record<number, string> } | null;
  metadata: Record<string, unknown>;
  error?: string;
}

export interface ValidationCheck {
  id: string;
  name: string;
  count: number;
  severity: "ok" | "warning" | "info";
  detail: string[] | ImageItem[] | Record<string, number>;
  extra?: Record<string, unknown>;
}
