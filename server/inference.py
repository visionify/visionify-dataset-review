"""
Lightweight inference server for YOLO models.
Uses FastAPI + ultralytics. Run with:
    pip install fastapi uvicorn ultralytics
    python server/inference.py
Listens on port 3457.
"""

import os, sys, json, shutil, tempfile, threading, time
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Missing dependencies. Install with:\n  pip install fastapi uvicorn ultralytics", file=sys.stderr)
    sys.exit(1)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_model_path = None

# Fine-tune state. One job at a time: training saturates the GPU, and a second
# run would race the first for the same output directory.
_ft_lock = threading.Lock()
_ft_state = {"running": False, "stage": "idle", "message": "", "started": 0.0,
             "epochs_done": 0, "epochs_total": 0, "result": None}


class LoadRequest(BaseModel):
    model_path: str


class FinetuneRequest(BaseModel):
    dataset_root: str
    # Keys of images the user has actually reviewed, as "<split>/<stem>".
    # Training on anything else would teach the model that its own misses are
    # correct: an untouched image has no labels, which YOLO reads as "nothing
    # here", reinforcing exactly the objects it already fails to see.
    reviewed_keys: list[str]
    epochs: int = 12
    imgsz: int = 640
    # Low by default: this is a nudge toward the corrections just made, not a
    # retrain. A high LR on a few dozen images forgets everything else.
    lr0: float = 0.001
    val_fraction: float = 0.2


class PredictRequest(BaseModel):
    image_path: str
    confidence: float = 0.25
    iou: float = 0.45


class Box(BaseModel):
    classId: int
    className: str
    x: float
    y: float
    w: float
    h: float
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_path": _model_path}


@app.post("/load")
def load_model(req: LoadRequest):
    global _model, _model_path
    p = os.path.expanduser(req.model_path)
    if not os.path.isfile(p):
        raise HTTPException(400, f"Model file not found: {p}")
    try:
        from ultralytics import YOLO
        _model = YOLO(p)
        _model_path = p
        names = _model.names or {}
        return {"ok": True, "model_path": p, "classes": names}
    except Exception as e:
        _model = None
        _model_path = None
        raise HTTPException(500, f"Failed to load model: {e}")


@app.post("/predict")
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(400, "No model loaded. POST /load first.")
    if not os.path.isfile(req.image_path):
        raise HTTPException(400, f"Image not found: {req.image_path}")
    try:
        results = _model.predict(
            source=req.image_path,
            conf=req.confidence,
            iou=req.iou,
            verbose=False,
        )
        boxes = []
        for r in results:
            img_w, img_h = r.orig_shape[1], r.orig_shape[0]
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                cls_name = _model.names.get(cls_id, str(cls_id))
                boxes.append(Box(classId=cls_id, className=cls_name, x=cx, y=cy, w=bw, h=bh, confidence=conf))
        return {"boxes": [b.model_dump() for b in boxes], "count": len(boxes)}
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


def _build_finetune_dataset(dataset_root: Path, reviewed_keys, val_fraction: float):
    """
    Stage a YOLO dataset containing ONLY reviewed images.

    Symlinks rather than copies, so a few hundred images cost nothing. Returns
    (tmpdir, data_yaml_path, n_train, n_val) or raises if there is too little
    to train on.
    """
    import yaml

    cfg = {}
    for fn in ("data.yaml", "dataset.yaml"):
        f = dataset_root / fn
        if f.exists():
            cfg = yaml.safe_load(f.read_text()) or {}
            break
    names = cfg.get("names")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    if not names:
        raise ValueError("data.yaml has no class names — cannot fine-tune")

    pairs = []
    for key in reviewed_keys:
        split, _, stem = key.partition("/")
        if not stem:
            continue
        images_rel = cfg.get(split)
        if not images_rel:
            continue
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            cand = dataset_root / images_rel / (stem + ext)
            if cand.exists():
                img = cand
                break
        if img is None:
            continue
        lbl = Path(str(dataset_root / images_rel).replace("images", "labels")) / (stem + ".txt")
        if not lbl.exists():
            continue          # reviewed but unlabelled: skip rather than teach "empty"
        pairs.append((img, lbl))

    if len(pairs) < 8:
        raise ValueError(f"only {len(pairs)} reviewed image(s) with labels — annotate a few more first")

    tmp = Path(tempfile.mkdtemp(prefix="yolo_ft_"))
    n_val = max(1, int(len(pairs) * val_fraction))
    for split, subset in (("val", pairs[:n_val]), ("train", pairs[n_val:])):
        (tmp / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in subset:
            os.symlink(img, tmp / "images" / split / img.name)
            os.symlink(lbl, tmp / "labels" / split / lbl.name)

    data_yaml = tmp / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(tmp), "train": "images/train", "val": "images/val",
        "nc": len(names), "names": {int(k): v for k, v in sorted(names.items())},
    }, sort_keys=False))
    return tmp, data_yaml, len(pairs) - n_val, n_val


def _run_finetune(req: "FinetuneRequest"):
    """Fine-tune in a worker thread and hot-swap the model if it succeeds."""
    global _model, _model_path
    tmp = None
    try:
        dataset_root = Path(req.dataset_root).expanduser().resolve()
        _ft_state.update(stage="staging", message="collecting reviewed images")
        tmp, data_yaml, n_train, n_val = _build_finetune_dataset(
            dataset_root, req.reviewed_keys, req.val_fraction)

        _ft_state.update(stage="training", epochs_total=req.epochs,
                         message=f"training on {n_train} images ({n_val} held out)")

        from ultralytics import YOLO
        base = _model_path
        if not base:
            raise ValueError("no model loaded to fine-tune from")
        m = YOLO(base)

        def _on_epoch_end(trainer):
            _ft_state["epochs_done"] = int(getattr(trainer, "epoch", 0)) + 1
        m.add_callback("on_train_epoch_end", _on_epoch_end)

        # Keep the weights WITH the dataset, not in a tempdir: they are the
        # accumulated value of the annotation session, and macOS clears /var
        # temp dirs out from under you.
        out = dataset_root / "review" / "finetune"
        out.mkdir(parents=True, exist_ok=True)
        run = time.strftime("%Y%m%d-%H%M%S")
        m.train(data=str(data_yaml), epochs=req.epochs, imgsz=req.imgsz, lr0=req.lr0,
                project=str(out), name=run, exist_ok=True, verbose=False,
                val=True, plots=False, save=True)

        best = out / run / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError("training produced no weights")

        _ft_state.update(stage="loading", message="swapping in the fine-tuned model")
        _model = YOLO(str(best))
        _model_path = str(best)
        _ft_state.update(stage="done", message=f"now predicting with {best}",
                         result={"weights": str(best), "images": n_train + n_val,
                                 "train": n_train, "val": n_val, "epochs": req.epochs})
    except Exception as e:
        _ft_state.update(stage="error", message=str(e), result=None)
    finally:
        _ft_state["running"] = False
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


@app.post("/finetune")
def finetune(req: FinetuneRequest):
    """
    Learn from the annotations made so far, then predict with the result.

    Runs in the background; poll /finetune/status. The previously loaded weights
    stay on disk untouched, so reverting is a /load away.
    """
    if _model is None:
        raise HTTPException(400, "No model loaded. POST /load first.")
    with _ft_lock:
        if _ft_state["running"]:
            raise HTTPException(409, "A fine-tune is already running")
        _ft_state.update(running=True, stage="queued", message="", started=time.time(),
                         epochs_done=0, epochs_total=req.epochs, result=None)
    threading.Thread(target=_run_finetune, args=(req,), daemon=True).start()
    return {"ok": True, "started": True}


@app.get("/finetune/status")
def finetune_status():
    s = dict(_ft_state)
    s["elapsed"] = round(time.time() - s["started"], 1) if s["started"] else 0
    s["model_path"] = _model_path
    return s


@app.post("/unload")
def unload_model():
    global _model, _model_path
    _model = None
    _model_path = None
    return {"ok": True}


if __name__ == "__main__":
    port = int(os.environ.get("INFERENCE_PORT", "3457"))
    print(f"Inference server at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
