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
        setInferenceStatus("online");
        setModelStatus({ loaded: h.model_loaded, classes: undefined });
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
    <div style={{ maxWidth: "36rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem" }}>
        <h1 style={{ fontSize: "1.25rem" }}>Settings</h1>
        <Link to="/" className="btn btn-ghost" style={{ padding: "0.35rem 0.5rem" }}>← Back</Link>
      </div>

      {/* ── Inference Server Status ─────────────────────────── */}
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
          <div style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", background: "oklch(0 0 0 / 0.04)", padding: "0.6rem 0.75rem", borderRadius: "var(--radius-sm)" }}>
            <p style={{ marginBottom: "0.35rem" }}>Start the inference server to enable auto-detection:</p>
            <code style={{ fontSize: "0.8rem" }}>pip install fastapi uvicorn ultralytics && python server/inference.py</code>
          </div>
        )}
        {inferenceStatus !== "checking" && (
          <button className="btn btn-ghost" onClick={checkInference} style={{ marginTop: "0.5rem", padding: "0.3rem 0.6rem", fontSize: "0.8rem" }}>
            Refresh status
          </button>
        )}
      </div>

      {/* ── Model Status ───────────────────────────────────── */}
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

      {/* ── Class Colors ───────────────────────────────────── */}
      <div className="card" style={{ padding: "1.25rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "0.5rem" }}>Class Colors</h2>
        <p style={{ color: "var(--color-text-muted)", fontSize: "0.85rem", marginBottom: "0.75rem" }}>
          Colors used for bounding boxes in the annotator.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          {classes.map((cls) => (
            <div key={cls.id} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <div style={{ width: 24, height: 24, borderRadius: 4, background: getColor(cls.id), border: "1px solid var(--color-border)" }} />
              <span style={{ minWidth: "6rem", fontWeight: 500, fontSize: "0.9rem" }}>{cls.name}</span>
              <input type="color" value={getColor(cls.id)} onChange={(e) => setColor(cls.id, e.target.value)}
                style={{ width: 32, height: 24, padding: 0, border: "none", cursor: "pointer", borderRadius: 4 }} />
              <input type="text" className="input" value={getColor(cls.id)} onChange={(e) => setColor(cls.id, e.target.value)}
                style={{ width: 85, padding: "0.3rem 0.4rem", fontSize: "0.8rem" }} />
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}>
          <button className="btn btn-primary" onClick={save} disabled={saving} style={{ fontSize: "0.85rem" }}>
            {saving ? "Saving..." : "Save colors"}
          </button>
          <button className="btn btn-ghost" onClick={reset} style={{ fontSize: "0.85rem" }}>Reset</button>
        </div>
        {message && <p style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "var(--color-text-muted)" }}>{message}</p>}
      </div>
    </div>
  );
}
