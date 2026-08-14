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

export default function SettingsPage() {
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [colors, setColors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [modelPath, setModelPath] = useState("");
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; path: string | null; classes?: Record<number, string> }>({ loaded: false, path: null });
  const [modelLoading, setModelLoading] = useState(false);
  const [modelMsg, setModelMsg] = useState<string | null>(null);
  const [ft, setFt] = useState<{ running: boolean; stage: string; message: string; epochs_done: number; epochs_total: number } | null>(null);

  useEffect(() => {
    api.getSummary().then((s) => setClasses(s.classes ?? [])).catch(() => {});
    api.getClassColors().then((c) => setColors(c || {})).catch(() => setColors({}));
    api.inferenceHealth().then(h => {
      setModelStatus({ loaded: h.model_loaded, path: h.model_path });
      if (h.model_path) setModelPath(h.model_path);
    }).catch(() => {});
  }, []);

  // Poll while a fine-tune runs; stop as soon as it settles.
  useEffect(() => {
    if (!ft?.running) return;
    const t = setInterval(async () => {
      try {
        const st = await api.inferenceFinetuneStatus();
        setFt(st);
        if (!st.running) {
          setModelStatus(m => ({ ...m, path: st.model_path }));
          if (st.model_path) setModelPath(st.model_path);
        }
      } catch { /* keep polling; the sidecar may be busy training */ }
    }, 2000);
    return () => clearInterval(t);
  }, [ft?.running]);

  const finetune = async () => {
    if (!window.confirm(
      "Fine-tune on the images you have marked REVIEWED, then predict with the result?\n\n" +
      "Only reviewed images are used — an unreviewed image has no labels, and training on it " +
      "would teach the model that whatever it already misses is genuinely absent.\n\n" +
      "Weights are saved under <dataset>/review/finetune/. The current model file is untouched."
    )) return;
    setModelMsg(null);
    try {
      await api.inferenceFinetune();
      setFt({ running: true, stage: "queued", message: "", epochs_done: 0, epochs_total: 0 });
    } catch (e) { setModelMsg(e instanceof Error ? e.message : "Fine-tune failed to start"); }
  };

  const loadModel = async () => {
    if (!modelPath.trim()) return;
    setModelLoading(true); setModelMsg(null);
    try {
      const r = await api.inferenceLoad(modelPath.trim());
      setModelStatus({ loaded: true, path: r.model_path, classes: r.classes });
      setModelMsg(`Model loaded with ${Object.keys(r.classes).length} classes.`);
      setTimeout(() => setModelMsg(null), 3000);
    } catch (e) { setModelMsg(e instanceof Error ? e.message : "Load failed"); }
    finally { setModelLoading(false); }
  };

  const unloadModel = async () => {
    try { await api.inferenceUnload(); } catch {}
    setModelStatus({ loaded: false, path: null });
    setModelMsg("Model unloaded.");
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
    setSaving(true);
    setMessage(null);
    try {
      const body: Record<string, string> = {};
      Object.entries(colors).forEach(([k, v]) => {
        if (v) body[k] = v;
      });
      await api.setClassColors(body as Record<number, string>);
      setMessage("Colors saved.");
      setTimeout(() => setMessage(null), 2000);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const reset = () => {
    setColors({});
  };

  return (
    <div className="card" style={{ padding: "1.5rem", maxWidth: "32rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
        <h1 style={{ fontSize: "1.25rem" }}>Settings</h1>
        <Link to="/" className="btn btn-ghost" style={{ padding: "0.35rem 0.5rem" }}>← Back</Link>
      </div>
      <p style={{ color: "var(--color-text-muted)", fontSize: "0.9rem", marginBottom: "1rem" }}>
        Set a color for each class. These colors are used for bounding boxes in the annotator. Default palette is used until you save custom colors.
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        {classes.map((cls) => (
          <div key={cls.id} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: 4,
                background: getColor(cls.id),
                border: "1px solid var(--color-border)",
              }}
            />
            <span style={{ minWidth: "8rem", fontWeight: 500 }}>{cls.name}</span>
            <input
              type="color"
              value={getColor(cls.id)}
              onChange={(e) => setColor(cls.id, e.target.value)}
              style={{ width: 40, height: 28, padding: 0, border: "none", cursor: "pointer", borderRadius: 4 }}
            />
            <input
              type="text"
              className="input"
              value={getColor(cls.id)}
              onChange={(e) => setColor(cls.id, e.target.value)}
              style={{ width: 90, padding: "0.35rem 0.5rem", fontSize: "0.85rem" }}
            />
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
        <button className="btn btn-primary" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save colors"}
        </button>
        <button className="btn btn-ghost" onClick={reset}>Reset to defaults</button>
      </div>
      {message && (
        <p style={{ marginTop: "0.75rem", fontSize: "0.9rem", color: "var(--color-text-muted)" }}>{message}</p>
      )}

      {/* Model configuration */}
      <hr style={{ margin: "1.5rem 0", border: "none", borderTop: "1px solid var(--color-border)" }} />
      <h2 style={{ fontSize: "1.1rem", marginBottom: "0.75rem" }}>Model (auto-detect)</h2>
      <p style={{ color: "var(--color-text-muted)", fontSize: "0.9rem", marginBottom: "0.75rem" }}>
        Load a YOLO .pt model to enable auto-detection in the annotation view. Requires the Python inference server to be running.
      </p>
      <div style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", marginBottom: "0.5rem", background: "oklch(0 0 0 / 0.04)", padding: "0.5rem 0.75rem", borderRadius: "var(--radius-sm)" }}>
        <strong>Setup:</strong> <code>pip install fastapi uvicorn ultralytics</code><br />
        <strong>Start:</strong> <code>python server/inference.py</code> (port 3457)
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <span style={{ width: 10, height: 10, borderRadius: "50%", background: modelStatus.loaded ? "var(--color-success)" : "var(--color-text-muted)" }} />
        <span style={{ fontSize: "0.9rem" }}>{modelStatus.loaded ? `Loaded: ${modelStatus.path}` : "No model loaded"}</span>
      </div>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
        <input
          type="text"
          className="input"
          placeholder="/path/to/model.pt"
          value={modelPath}
          onChange={e => setModelPath(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") loadModel(); }}
          style={{ flex: 1, minWidth: "200px", padding: "0.4rem 0.6rem" }}
        />
        <button className="btn btn-primary" onClick={loadModel} disabled={modelLoading || !modelPath.trim()}>
          {modelLoading ? "Loading…" : "Load model"}
        </button>
        <button
          className="btn"
          onClick={finetune}
          disabled={!modelStatus.loaded || !!ft?.running}
          title="Learn from the annotations you have reviewed so far, then use the result for auto-detect"
          style={{ marginLeft: "0.5rem" }}
        >
          {ft?.running ? "Learning…" : "🧠 Learn from my annotations"}
        </button>
        {ft && (
          <div style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: ft.stage === "error" ? "#b91c1c" : "var(--color-text-muted)" }}>
            {ft.stage}
            {ft.epochs_total ? ` · epoch ${ft.epochs_done}/${ft.epochs_total}` : ""}
            {ft.message ? ` · ${ft.message}` : ""}
          </div>
        )}
        {modelStatus.loaded && (
          <button className="btn btn-ghost" onClick={unloadModel}>Unload</button>
        )}
      </div>
      {modelStatus.loaded && modelStatus.classes && (
        <div style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
          Model classes: {Object.entries(modelStatus.classes).map(([id, name]) => `${name} (${id})`).join(", ")}
        </div>
      )}
      {modelMsg && <p style={{ marginTop: "0.5rem", fontSize: "0.9rem", color: "var(--color-text-muted)" }}>{modelMsg}</p>}
    </div>
  );
}
