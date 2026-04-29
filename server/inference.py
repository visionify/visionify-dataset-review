"""
Lightweight inference server for YOLO models.
Uses FastAPI + ultralytics. Run with:
    pip install fastapi uvicorn ultralytics
    python server/inference.py
Listens on port 3457.
"""

import os, sys, json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_model = None
_model_path = None


class LoadRequest(BaseModel):
    model_path: str


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
    raw = os.path.expanduser(req.model_path)
    p = raw if os.path.isabs(raw) else str(PROJECT_ROOT / raw)
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
