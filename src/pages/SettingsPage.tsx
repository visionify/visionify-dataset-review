import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "@/api";
import { getClassColor } from "@/classColors";
import type { ClassItem } from "@/types";

const DEFAULT_PALETTE = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
  "#9a6324", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
];

type ServerStatus = "checking" | "online" | "offline";

export default function SettingsPage() {
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [colors, setColors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const [inferenceStatus, setInferenceStatus] = useState<ServerStatus>("checking");
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; classes?: Record<number, string> }>({ loaded: false });
  const [modelLoading, setModelLoading] = useState(false);
  const [modelMsg, setModelMsg] = useState<string | null>(null);

  const checkInference = () => {
    setInferenceStatus("checking");
    api.inferenceHealth()
      .then(h => {
        if (h.status === "offline") {
          setInferenceStatus("offline");
        } else {
          setInferenceStatus("online");
          setModelStatus({ loaded: h.model_loaded, classes: undefined });
        }
      })
      .catch(() => setInferenceStatus("offline"));
  };

  useEffect(() => {
    api.getSummary().then((s) => setClasses(s.classes ?? [])).catch(() => {});
    api.getClassColors().then((c) => setColors(c || {})).catch(() => setColors({}));
    checkInference();
  }, []);

  const loadModel = async () => {
    setModelLoading(true); setModelMsg(null);
    try {
      const r = await api.inferenceLoad("models/model.pt");
      setModelStatus({ loaded: true, classes: r.classes });
      setModelMsg(`Loaded with ${Object.keys(r.classes).length} classes`);
      setTimeout(() => setModelMsg(null), 3000);
    } catch (e) { setModelMsg(e instanceof Error ? e.message : "Load failed"); }
    finally { setModelLoading(false); }
  };

  const unloadModel = async () => {
    try { await api.inferenceUnload(); } catch {}
    setModelStatus({ loaded: false });
    setModelMsg("Model unloaded");
    setTimeout(() => setModelMsg(null), 2000);
  };

  const getColor = (classId: number) => {
    const key = String(classId);
    if (colors[key]) return colors[key];
    return DEFAULT_PALETTE[classId % DEFAULT_PALETTE.length] ?? getClassColor(classId, null);
  };

  const setColor = (classId: number, hex: string) => {
    setColors((prev) => ({ ...prev, [String(classId)]: hex }));
  };

  const save = async () => {
    setSaving(true); setMessage(null);
    try {
      const body: Record<string, string> = {};
      Object.entries(colors).forEach(([k, v]) => { if (v) body[k] = v; });
      await api.setClassColors(body as Record<number, string>);
      setMessage("Colors saved.");
      setTimeout(() => setMessage(null), 2000);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Save failed");
    } finally { setSaving(false); }
  };

  const reset = () => setColors({});

  const statusDot = (on: boolean | null) => (
    <span style={{
      display: "inline-block", width: 10, height: 10, borderRadius: "50%",
      background: on === null ? "var(--color-text-muted)" : on ? "#22c55e" : "#ef4444",
      boxShadow: on ? "0 0 6px #22c55e80" : on === false ? "0 0 6px #ef444480" : "none",
    }} />
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <Link to="/" className="btn btn-ghost" style={{ padding: "0.35rem 0.5rem" }}>← Back</Link>
        <h1 style={{ fontSize: "1.25rem" }}>Settings</h1>
      </div>

      {/* Row 1: Inference Server + YOLO Model side by side, responsive */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1rem" }}>
        {/* ── Inference Server ─────────────────────────── */}
        <div className="card" style={{ padding: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
            <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>Inference Server</h2>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              {statusDot(inferenceStatus === "checking" ? null : inferenceStatus === "online")}
              <span style={{
                fontSize: "0.8rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em",
                color: inferenceStatus === "online" ? "#22c55e" : inferenceStatus === "offline" ? "#ef4444" : "var(--color-text-muted)",
              }}>
                {inferenceStatus === "checking" ? "Checking..." : inferenceStatus === "online" ? "Running" : "Offline"}
              </span>
            </div>
          </div>
          {inferenceStatus === "offline" && (
            <div style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", background: "oklch(0 0 0 / 0.04)", padding: "0.75rem 1rem", borderRadius: "var(--radius-sm)", display: "flex", flexDirection: "column", gap: "0.6rem" }}>
              <p style={{ fontWeight: 500, color: "var(--color-text)" }}>How to start the inference server:</p>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                <div style={{ display: "flex", gap: "0.5rem", alignItems: "baseline" }}>
                  <span style={{ fontWeight: 600, color: "var(--color-accent)", minWidth: "1.2rem" }}>1.</span>
                  <div>
                    <span>Install dependencies (one time only)</span>
                    <code style={{ display: "block", marginTop: "0.25rem", background: "oklch(0 0 0 / 0.06)", padding: "0.35rem 0.5rem", borderRadius: 4, fontSize: "0.8rem" }}>pip install fastapi uvicorn ultralytics</code>
                  </div>
                </div>
                <div style={{ display: "flex", gap: "0.5rem", alignItems: "baseline" }}>
                  <span style={{ fontWeight: 600, color: "var(--color-accent)", minWidth: "1.2rem" }}>2.</span>
                  <div>
                    <span>Place your YOLO model at</span>
                    <code style={{ display: "inline", background: "oklch(0 0 0 / 0.06)", padding: "0.1rem 0.35rem", borderRadius: 4, fontSize: "0.8rem", marginLeft: "0.25rem" }}>models/model.pt</code>
                  </div>
                </div>
                <div style={{ display: "flex", gap: "0.5rem", alignItems: "baseline" }}>
                  <span style={{ fontWeight: 600, color: "var(--color-accent)", minWidth: "1.2rem" }}>3.</span>
                  <div>
                    <span>Open a new terminal and run</span>
                    <code style={{ display: "block", marginTop: "0.25rem", background: "oklch(0 0 0 / 0.06)", padding: "0.35rem 0.5rem", borderRadius: 4, fontSize: "0.8rem" }}>python server/inference.py</code>
                  </div>
                </div>
                <div style={{ display: "flex", gap: "0.5rem", alignItems: "baseline" }}>
                  <span style={{ fontWeight: 600, color: "var(--color-accent)", minWidth: "1.2rem" }}>4.</span>
                  <span>Click <strong>Refresh status</strong> below — it should turn green</span>
                </div>
              </div>
            </div>
          )}
          {inferenceStatus === "online" && (
            <p style={{ fontSize: "0.85rem", color: "#22c55e" }}>
              Inference server is running on port 3457. You can load a model below.
            </p>
          )}
          {inferenceStatus !== "checking" && (
            <button className="btn btn-ghost" onClick={checkInference} style={{ marginTop: "0.5rem", padding: "0.3rem 0.6rem", fontSize: "0.8rem" }}>
              Refresh status
            </button>
          )}
        </div>

        {/* ── YOLO Model ──────────────────────────────── */}
        <div className="card" style={{ padding: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
            <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>YOLO Model</h2>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              {statusDot(modelStatus.loaded)}
              <span style={{
                fontSize: "0.8rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em",
                color: modelStatus.loaded ? "#22c55e" : "#ef4444",
              }}>
                {modelStatus.loaded ? "Loaded" : "Not loaded"}
              </span>
            </div>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", marginBottom: "0.75rem" }}>
            Place your model at <code style={{ background: "oklch(0 0 0 / 0.06)", padding: "0.15rem 0.4rem", borderRadius: 4 }}>models/model.pt</code>
          </p>
          {modelStatus.loaded && modelStatus.classes && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "0.75rem" }}>
              {Object.entries(modelStatus.classes).map(([id, name]) => (
                <span key={id} style={{
                  fontSize: "0.78rem", padding: "0.2rem 0.5rem", borderRadius: 20,
                  background: "oklch(0.55 0.2 265 / 0.12)", color: "var(--color-accent)", fontWeight: 500,
                }}>
                  {name}
                </span>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            {!modelStatus.loaded ? (
              <button className="btn btn-primary" onClick={loadModel} disabled={modelLoading || inferenceStatus !== "online"} style={{ fontSize: "0.85rem" }}>
                {modelLoading ? "Loading..." : "Load model"}
              </button>
            ) : (
              <button className="btn btn-ghost" onClick={unloadModel} style={{ fontSize: "0.85rem", color: "#ef4444", borderColor: "#ef444433" }}>
                Unload model
              </button>
            )}
          </div>
          {inferenceStatus !== "online" && !modelStatus.loaded && (
            <p style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", marginTop: "0.4rem" }}>
              Start the inference server first to load a model.
            </p>
          )}
          {modelMsg && <p style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "var(--color-text-muted)" }}>{modelMsg}</p>}
        </div>
      </div>

      {/* Row 2: Class Colors — full width below */}
      <div className="card" style={{ padding: "1.25rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.5rem" }}>Class Colors</h2>
        <p style={{ color: "var(--color-text-muted)", fontSize: "0.85rem", marginBottom: "1rem" }}>
          Colors used for bounding boxes in the annotator.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.6rem 2rem" }}>
          {classes.map((cls) => (
            <div key={cls.id} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <div style={{ width: 28, height: 28, borderRadius: 6, background: getColor(cls.id), border: "1px solid var(--color-border)", flexShrink: 0 }} />
              <span style={{ flex: 1, fontWeight: 500, fontSize: "0.9rem" }}>{cls.name}</span>
              <input type="color" value={getColor(cls.id)} onChange={(e) => setColor(cls.id, e.target.value)}
                style={{ width: 36, height: 28, padding: 0, border: "none", cursor: "pointer", borderRadius: 4, flexShrink: 0 }} />
              <input type="text" className="input" value={getColor(cls.id)} onChange={(e) => setColor(cls.id, e.target.value)}
                style={{ width: 90, padding: "0.3rem 0.5rem", fontSize: "0.8rem", flexShrink: 0 }} />
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
          <button className="btn btn-primary" onClick={save} disabled={saving} style={{ fontSize: "0.85rem" }}>
            {saving ? "Saving..." : "Save colors"}
          </button>
          <button className="btn btn-ghost" onClick={reset} style={{ fontSize: "0.85rem" }}>Reset</button>
        </div>
        {message && <p style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "var(--color-text-muted)" }}>{message}</p>}
      </div>

      {/* Row 3: How to Run — always visible */}
      <div className="card" style={{ padding: "1.25rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem" }}>How to Run</h2>

        <div style={{ background: "oklch(0 0 0 / 0.04)", padding: "1rem", borderRadius: "var(--radius-sm)", marginBottom: "1rem" }}>
          <p style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--color-text)", marginBottom: "0.5rem" }}>Quick start — one command runs everything:</p>
          <code style={{ display: "block", background: "oklch(0 0 0 / 0.06)", padding: "0.5rem 0.75rem", borderRadius: 4, fontSize: "0.85rem" }}>./start.sh</code>
          <p style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", marginTop: "0.4rem" }}>
            Starts both the API server (port 3456) and inference server (port 3457). Press Ctrl+C to stop both.
          </p>
        </div>

        <p style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", marginBottom: "0.75rem" }}>Or run them separately:</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.5rem" }}>
          <div>
            <h3 style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-accent)" }}>API Server (required)</h3>
            <p style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", marginBottom: "0.35rem" }}>Serves images, annotations, and review data from your local dataset.</p>
            <code style={{ display: "block", background: "oklch(0 0 0 / 0.06)", padding: "0.4rem 0.6rem", borderRadius: 4, fontSize: "0.8rem" }}>npm install && node server/index.js</code>
          </div>
          <div>
            <h3 style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-accent)" }}>Inference Server (for auto-detection)</h3>
            <p style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", marginBottom: "0.35rem" }}>Requires Python and a YOLO model at <code style={{ background: "oklch(0 0 0 / 0.06)", padding: "0.1rem 0.3rem", borderRadius: 3, fontSize: "0.8rem" }}>models/model.pt</code></p>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
              <code style={{ display: "block", background: "oklch(0 0 0 / 0.06)", padding: "0.4rem 0.6rem", borderRadius: 4, fontSize: "0.8rem" }}>pip install fastapi uvicorn ultralytics</code>
              <code style={{ display: "block", background: "oklch(0 0 0 / 0.06)", padding: "0.4rem 0.6rem", borderRadius: 4, fontSize: "0.8rem" }}>python server/inference.py</code>
            </div>
          </div>
        </div>
      </div>

      {/* Row 4: Dataset Structure — always visible */}
      <div className="card" style={{ padding: "1.25rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.75rem" }}>Dataset Structure</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", marginBottom: "1rem" }}>
          Each dataset/task should follow this folder structure. You only need to create the first two folders — the rest is handled automatically.
        </p>
        <div style={{ background: "oklch(0 0 0 / 0.04)", padding: "1rem", borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: "0.82rem", lineHeight: 1.8 }}>
          <div>your-dataset/</div>
          <div style={{ paddingLeft: "1.5rem" }}>
            <span style={{ color: "var(--color-accent)", fontWeight: 600 }}>images/train/</span>
            <span style={{ color: "var(--color-text-muted)", marginLeft: "0.75rem" }}>← paste all your images here</span>
          </div>
          <div style={{ paddingLeft: "1.5rem" }}>
            <span style={{ color: "var(--color-accent)", fontWeight: 600 }}>labels/train/</span>
            <span style={{ color: "var(--color-text-muted)", marginLeft: "0.75rem" }}>← create this empty folder, model will generate labels</span>
          </div>
          <div style={{ paddingLeft: "1.5rem", color: "var(--color-text-muted)" }}>
            <span>review/</span>
            <span style={{ marginLeft: "0.75rem" }}>← auto-created when you start reviewing</span>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", marginTop: "1rem", fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
          <p><strong style={{ color: "var(--color-text)" }}>You create:</strong> <code style={{ background: "oklch(0 0 0 / 0.06)", padding: "0.1rem 0.3rem", borderRadius: 3 }}>images/train/</code> (with images) and <code style={{ background: "oklch(0 0 0 / 0.06)", padding: "0.1rem 0.3rem", borderRadius: 3 }}>labels/train/</code> (empty)</p>
          <p><strong style={{ color: "var(--color-text)" }}>Model creates:</strong> label .txt files inside <code style={{ background: "oklch(0 0 0 / 0.06)", padding: "0.1rem 0.3rem", borderRadius: 3 }}>labels/train/</code> after running inference</p>
          <p><strong style={{ color: "var(--color-text)" }}>Auto-created:</strong> <code style={{ background: "oklch(0 0 0 / 0.06)", padding: "0.1rem 0.3rem", borderRadius: 3 }}>review/</code> folder with review state, tags, and metadata</p>
        </div>
      </div>
    </div>
  );
}
