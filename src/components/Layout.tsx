import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { api } from "@/api";

export default function Layout({ children }: { children: React.ReactNode }) {
  const loc = useLocation();
  const [summary, setSummary] = useState<Awaited<ReturnType<typeof api.getSummary>> | null>(null);
  const [serverDown, setServerDown] = useState(false);

  const shouldRefetch = loc.pathname === "/" || loc.pathname === "/config" || loc.pathname.startsWith("/images");
  useEffect(() => {
    if (shouldRefetch || !summary)
      api.getSummary()
        .then((s) => { setSummary(s); setServerDown(false); })
        .catch((e) => {
          setSummary(null);
          const msg = e instanceof Error ? e.message : "";
          if (msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("ERR_CONNECTION")) {
            setServerDown(true);
          }
        });
  }, [shouldRefetch]);

  const config = summary?.config;
  const hasTrain = config?.train != null;
  const hasVal = config?.val != null;
  const hasTest = config?.test != null;

  const nav = [
    { to: "/", label: "Classes" },
    { to: "/images/all", label: "All Images" },
    ...(hasTrain ? [{ to: "/images/train", label: "Training" }] : []),
    ...(hasVal ? [{ to: "/images/val", label: "Validation" }] : []),
    ...(hasTest ? [{ to: "/images/test", label: "Test" }] : []),
    { to: "/validation", label: "Validation checks" },
    { to: "/settings", label: "Settings" },
    { to: "/config", label: "Dataset" },
  ];

  return (
    <div className="app-layout">
      <header
        style={{
          padding: "0.75rem 1.5rem",
          borderBottom: "1px solid var(--color-border)",
          background: "var(--color-surface)",
          display: "flex",
          alignItems: "center",
          gap: "1.5rem",
          flexWrap: "wrap",
        }}
      >
        <Link
          to="/"
          style={{
            fontWeight: 700,
            fontSize: "1.125rem",
            color: "var(--color-text)",
            textDecoration: "none",
          }}
        >
          YOLO Dataset Review
        </Link>
        <nav style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          {nav.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className="btn btn-ghost"
              style={{
                textDecoration: "none",
                background: loc.pathname === to ? "oklch(0 0 0 / 0.08)" : undefined,
              }}
            >
              {label}
            </Link>
          ))}
        </nav>
      </header>
      {serverDown && (
        <div style={{
          background: "#fef2f2", borderBottom: "1px solid #fecaca", padding: "0.6rem 1.5rem",
          display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem", color: "#991b1b",
        }}>
          <span style={{ fontWeight: 600 }}>Server unreachable.</span>
          <span>Make sure both servers are running on your machine. Run <code style={{ background: "#fee2e2", padding: "0.1rem 0.35rem", borderRadius: 3 }}>./start.sh</code> in the project directory.</span>
        </div>
      )}
      <main className="main-content">{children}</main>
    </div>
  );
}
