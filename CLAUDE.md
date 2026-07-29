# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A local web app for reviewing, annotating, and cleaning YOLO-format object-detection datasets. Users point it at a dataset folder, browse images/labels, edit bounding boxes, run validation checks, and optionally use a YOLO model for auto-detection. All data stays local.

## Commands

```bash
npm run dev          # Start both API server (port 3456) + Vite dev server (port 5173)
npm run dev:server   # API server only
npm run dev:vite     # Vite dev server only
npm run build        # TypeScript check + production build (tsc -b && vite build)
npm run preview      # Serve production build
```

Optional inference server (Python):
```bash
pip install fastapi uvicorn ultralytics
python server/inference.py   # port 3457
```

No test framework is configured. There are no lint or format commands.

## Architecture

**Two-process setup:**
- **Express backend** (`server/index.js`, plain JS) — port 3456. Handles all file I/O: reads YOLO datasets (images, labels, `data.yaml`), serves images, CRUD for annotations/tags/review state, validation checks, and proxies inference requests to the Python sidecar.
- **React frontend** (`src/`, TypeScript) — Vite dev server on port 5173, proxies `/api` and `/dataset-asset` to the Express backend (configured in `vite.config.ts`).
- **Optional Python inference sidecar** (`server/inference.py`, FastAPI) — port 3457. Loads YOLO `.pt` models and runs predictions. The Express server proxies `/api/inference/*` to it.

**Frontend structure:**
- `src/api.ts` — Typed API client; all backend calls go through the `api` object. Uses `/api` prefix (proxied by Vite).
- `src/types.ts` — Shared TypeScript interfaces (`BBox`, `ImageItem`, `ClassItem`, `ValidationCheck`, etc.).
- `src/components/BBoxCanvas.tsx` — Core annotation component: renders image with SVG bounding box overlay, handles draw/select/resize interactions.
- `src/components/Layout.tsx` — App shell with navigation sidebar.
- `src/pages/` — Route-based pages: `ClassesPage` (dashboard), `ImagesPage` (grid browser), `ImageDetailPage` (annotation editor), `ValidationPage`, `ConfigPage`, `SettingsPage`, `ClassDetailPage`.
- `src/App.tsx` — React Router routes. `/` is the classes dashboard; `/images/:split` for image grids; `/image/:split/:name` for annotation view.
- `src/classColors.ts` — Default color palette for bounding box classes.

**Backend key patterns:**
- Dataset path stored in `dataset-path.json` at project root (or `DATASET_PATH` env var).
- Supports two YOLO directory layouts: standard (`images/train`, `labels/train`) and Roboflow-style (`train/images`, `train/labels`). Auto-detects `valid` vs `val` folder names.
- Also supports **classification datasets** (auto-detected): YOLO-cls layout (`train/className/image.jpg`) or flat class subdirs at root. Classification has dedicated endpoints (`/api/classification/move`, `/api/classification/delete-images`).
- Review state (reviewed set, tags, class colors, metadata) stored in a `review/` directory inside the dataset folder.
- Module-level config caches (`_cfgCache`, `classIndexCache`, `tagIndexCache`) — call `invalidateConfigCache()` after any mutation that changes dataset structure.
- **Crop export** (`/api/crop`, `/api/crop/status`): writes a sub-region of an image plus its remapped labels to a sibling `<dataset>-cropped` folder, mirroring the source layout and copying `data.yaml`. The source dataset is never modified. Uses `sharp`. Crop regions are **rectangle or polygon**:
  - Polygon crops are cropped to the polygon's bounding box, then pixels outside the polygon are filled with `CROP_MASK_GREY` (114 — the Ultralytics letterbox grey), so training data matches inference-time polygon masking.
  - Masking is one `over` composite of a grey layer with an even-odd polygon hole. Do **not** rewrite it as `dest-in` + `flatten`: sharp applies `flatten` *before* `composite` regardless of call order, which flattens the unmasked image and leaves the masked area black.
  - Box survival uses exact polygon∩box area (Sutherland–Hodgman + shoelace), so boxes hidden under grey are dropped rather than kept as phantom labels. Below `CROP_KEEP_THRESHOLD` (0.5) surviving area, a box is dropped.
  - `CROP_KEEP_THRESHOLD` and the geometry helpers are mirrored in `src/components/BBoxCanvas.tsx` for the on-canvas preview; the server is authoritative, the client copy only drives box opacity.
- `DATASET_CONFIG_PATH` and `PORT` env vars let a throwaway server instance point at its own dataset — use this when testing crop/delete endpoints so you never write into the user's configured dataset.
- The `@/*` path alias maps to `src/*` (configured in both `tsconfig.json` and `vite.config.ts`).

**Frontend patterns:**
- No global state library. All state is local `useState` per page/component. Pages fetch from the API directly on mount.
- Styling uses plain CSS (`src/index.css`) with `@layer reset, base, layout, components` and CSS custom properties on `:root` (oklch colors). Fonts: DM Sans + JetBrains Mono.

## TypeScript

Strict mode enabled with `noUnusedLocals`, `noUnusedParameters`, `noUncheckedIndexedAccess`, and `noFallthroughCasesInSwitch`. Target is ES2020. The server (`server/`) is plain JavaScript (not included in `tsconfig`).
