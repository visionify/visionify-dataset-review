import { useEffect, useState, useCallback, useRef } from "react";
import { api, getServerUrl, setServerUrl } from "@/api";

/** Turn a file:// URL or path string into an absolute path (no file://). */
function fileUrlToPath(url: string): string {
  const trimmed = url.trim();
  if (trimmed.startsWith("file://")) {
    try {
      const decoded = decodeURIComponent(trimmed.slice(7));
      return decoded.replace(/\\/g, "/");
    } catch {
      return trimmed.replace(/\\/g, "/");
    }
  }
  return trimmed.replace(/\\/g, "/");
}

/** Get directory path: if the path is a file, return parent directory. */
function toDirectoryPath(p: string): string {
  const norm = p.replace(/\/+$/, "");
  const lastSlash = Math.max(norm.lastIndexOf("/"), norm.lastIndexOf("\\"));
  if (lastSlash > 0) {
    const maybeDir = norm.slice(0, lastSlash);
    const rest = norm.slice(lastSlash + 1);
    if (rest.includes(".")) return maybeDir;
    return norm;
  }
  return norm;
}

export default function ConfigPage() {
  const [path, setPath] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [dropMessage, setDropMessage] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [serverUrl, setServerUrlLocal] = useState(getServerUrl);
  const [serverOk, setServerOk] = useState<boolean | null>(null);

  useEffect(() => {
    api.getConfig().then((c) => { setPath(c.datasetPath || ""); setServerOk(true); }).catch(() => setServerOk(false));
  }, []);

  const saveServerUrl = () => {
    setServerUrl(serverUrl);
    setServerOk(null);
    api.getConfig().then((c) => { setPath(c.datasetPath || ""); setServerOk(true); }).catch(() => setServerOk(false));
  };

  const applyPath = useCallback(
    async (newPath: string) => {
      const trimmed = newPath.trim();
      if (!trimmed) return;
      setSaving(true);
      setMessage(null);
      setDropMessage(null);
      try {
        await api.setConfig(trimmed);
        setPath(trimmed);
        setMessage("Dataset path saved. Go to Classes or All Images to load.");
      } catch (e) {
        setMessage(e instanceof Error ? e.message : "Failed to save");
      } finally {
        setSaving(false);
      }
    },
    []
  );

  async function handleSave() {
    await applyPath(path);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    setDropMessage(null);

    const dt = e.dataTransfer;
    if (!dt) return;

    const uriList = dt.getData("text/uri-list")?.trim();
    if (uriList) {
      const first = uriList.split(/\r?\n/)[0]?.trim();
      if (first?.startsWith("file://")) {
        const fullPath = fileUrlToPath(first);
        const dirPath = toDirectoryPath(fullPath);
        setPath(dirPath);
        applyPath(dirPath);
        return;
      }
    }

    const plain = dt.getData("text/plain")?.trim();
    if (plain && (plain.includes("/") || plain.includes("\\"))) {
      const dirPath = toDirectoryPath(plain.startsWith("file://") ? fileUrlToPath(plain) : plain);
      setPath(dirPath);
      applyPath(dirPath);
      return;
    }

    const items = dt.items;
    if (items?.length) {
      const item = items[0];
      const file = item?.kind === "file" ? item.getAsFile() : null;
      if (file) {
        const withPath = file as File & { path?: string };
        if (typeof withPath.path === "string") {
          const dir = withPath.path.replace(/[\\/][^/\\]*$/, "").replace(/\\/g, "/");
          setPath(dir);
          applyPath(dir);
          return;
        }
      }
    }

    setDropMessage("Path from dropped files isn’t available in this browser. Please paste or type the full path to your dataset folder in the box below and click Save path.");
    setTimeout(() => inputRef.current?.focus(), 100);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  }

  function handleDragLeave() {
    setDragOver(false);
  }

  return (
    <div className="card" style={{ padding: "1.5rem", maxWidth: "36rem" }}>
      <h1 style={{ fontSize: "1.25rem", marginBottom: "0.75rem" }}>Server Connection</h1>
      <p style={{ color: "var(--color-text-muted)", marginBottom: "0.75rem", fontSize: "0.9rem" }}>
        Enter the URL of your local server (e.g. <code>http://localhost:3456</code>). Leave empty if running locally on the same machine.
      </p>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem" }}>
        <input
          type="text"
          className="input"
          value={serverUrl}
          onChange={(e) => setServerUrlLocal(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") saveServerUrl(); }}
          placeholder="http://localhost:3456"
          style={{ flex: 1, padding: "0.4rem 0.6rem" }}
        />
        <button className="btn btn-primary" onClick={saveServerUrl}>Connect</button>
      </div>
      {serverOk === true && <p style={{ fontSize: "0.85rem", color: "var(--color-success)" }}>Connected</p>}
      {serverOk === false && <p style={{ fontSize: "0.85rem", color: "var(--color-warning)" }}>Cannot reach server. Make sure <code>node server/index.js</code> is running.</p>}

      <hr style={{ margin: "1.5rem 0", border: "none", borderTop: "1px solid var(--color-border)" }} />
      <h1 style={{ fontSize: "1.25rem", marginBottom: "0.75rem" }}>Dataset</h1>
      <p style={{ color: "var(--color-text-muted)", marginBottom: "1rem", fontSize: "0.9rem" }}>
        Enter the <strong>full path</strong> to your YOLO dataset folder (the one that contains{" "}
        <code>data.yaml</code>, <code>images/</code>, <code>labels/</code>). The server runs on your machine, so use a path on this computer.
      </p>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        style={{
          border: `2px dashed ${dragOver ? "var(--color-accent)" : "var(--color-border)"}`,
          borderRadius: "var(--radius-md)",
          padding: "2rem",
          textAlign: "center",
          marginBottom: "1rem",
          background: dragOver ? "oklch(0.55 0.2 265 / 0.08)" : "var(--color-bg)",
          transition: "background 0.15s, border-color 0.15s",
        }}
      >
        <span style={{ color: "var(--color-text-muted)", fontSize: "0.95rem" }}>
          Drop folder here (if your OS provides a path) or paste path below
        </span>
      </div>
      {dropMessage && (
        <p style={{ marginBottom: "0.75rem", fontSize: "0.9rem", color: "var(--color-warning)" }}>
          {dropMessage}
        </p>
      )}
      <input
        ref={inputRef}
        type="text"
        className="input"
        value={path}
        onChange={(e) => setPath(e.target.value)}
        onPaste={(e) => {
          const pasted = e.clipboardData?.getData("text")?.trim();
          if (pasted && (pasted.includes("/") || pasted.includes("\\"))) {
            setPath(pasted);
            setTimeout(() => applyPath(pasted), 0);
          }
        }}
        placeholder="e.g. /home/user/datasets/mydataset or C:\datasets\mydataset"
        style={{ marginBottom: "0.5rem" }}
      />
      <p style={{ marginBottom: "1rem", fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
        Tip: In your file manager, right‑click the dataset folder → Copy path (or Copy), then paste here and click Save.
      </p>
      <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
        {saving ? "Saving…" : "Save path"}
      </button>
      {message && (
        <p style={{ marginTop: "1rem", color: message.startsWith("Dataset") ? "var(--color-success)" : "var(--color-text-muted)", fontSize: "0.9rem" }}>
          {message}
        </p>
      )}
    </div>
  );
}
