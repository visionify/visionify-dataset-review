import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "@/api";
import type { TagGroup } from "@/api";
import type { ClassItem, ImageItem } from "@/types";

export default function ClassesPage() {
  const [summary, setSummary] = useState<Awaited<ReturnType<typeof api.getSummary>> | null>(null);
  const [samplesByClass, setSamplesByClass] = useState<Record<number, ImageItem[]>>({});
  const [autoTags, setAutoTags] = useState<{ tasks: TagGroup[]; months: TagGroup[]; cameras: TagGroup[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getSummary().then((data) => {
        setSummary(data);
        if (data.configured && data.classes?.length) {
          return Promise.all(
            data.classes.map((cls) =>
              api.getClassSamples(cls.id, 8).then((r) => [cls.id, r.samples] as const)
            )
          ).then((pairs) => {
            setSamplesByClass(Object.fromEntries(pairs));
          });
        }
      }),
      api.getAutoTags().then(setAutoTags).catch(() => {}),
    ])
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p style={{ color: "var(--color-text-muted)" }}>Loading dataset…</p>;
  if (error) return <p style={{ color: "var(--color-danger)" }}>{error}</p>;
  if (!summary?.configured) {
    return (
      <div className="card" style={{ padding: "1.5rem", maxWidth: "28rem" }}>
        <p style={{ marginBottom: "1rem" }}>No dataset set. Open Dataset to choose a folder.</p>
        <Link to="/config" className="btn btn-primary">Open Dataset</Link>
      </div>
    );
  }

  const classes = summary.classes ?? [];
  const totalImages = summary.totalImages ?? 0;
  const reviewedCount = summary.reviewedCount ?? 0;
  const pctReviewed = totalImages ? Math.round((reviewedCount / totalImages) * 100) : 0;
  const splitCounts = summary.splitCounts ?? {};
  const missingLabels = summary.missingLabelsCount ?? 0;
  const emptyLabels = summary.emptyLabelsCount ?? 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem" }}>
        <h1 style={{ fontSize: "1.5rem" }}>Dashboard</h1>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <Link to="/validation" className="btn btn-ghost">Validation checks</Link>
        </div>
      </div>

      {/* Stats cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: "0.75rem" }}>
        <StatCard label="Total images" value={totalImages} />
        <StatCard label="Reviewed" value={`${reviewedCount} (${pctReviewed}%)`} color="var(--color-success)" />
        <StatCard label="Remaining" value={totalImages - reviewedCount} color="var(--color-text-muted)" />
        {Object.entries(splitCounts).map(([s, c]) => (
          <StatCard key={s} label={s === "train" ? "Training" : s === "val" ? "Validation" : s === "test" ? "Test" : s} value={c as number} />
        ))}
        <StatCard label="Missing labels" value={missingLabels} color={missingLabels > 0 ? "var(--color-warning)" : "var(--color-success)"} link="/validation" />
        <StatCard label="Empty labels" value={emptyLabels} color={emptyLabels > 0 ? "var(--color-warning)" : "var(--color-success)"} link="/validation" />
        <StatCard label="Classes" value={classes.length} />
      </div>

      {/* Progress bar */}
      <div style={{ background: "var(--color-border)", borderRadius: 6, height: 10, overflow: "hidden" }}>
        <div style={{ width: `${pctReviewed}%`, height: "100%", background: "var(--color-success)", borderRadius: 6, transition: "width 0.3s" }} />
      </div>

      <h2 style={{ fontSize: "1.2rem", marginTop: "0.5rem" }}>Classes</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "1rem" }}>
        {classes.map((cls) => (
          <ClassCard key={cls.id} cls={cls} samples={samplesByClass[cls.id] ?? []} />
        ))}
      </div>

      {autoTags && (autoTags.months.length > 0 || autoTags.tasks.length > 0 || autoTags.cameras.length > 0) && (
        <>
          {autoTags.months.length > 0 && (
            <TagRow title="Dates" tags={autoTags.months} tagType="month" color="oklch(0.65 0.15 250)" />
          )}
          {autoTags.tasks.length > 0 && (
            <TagRow title="Tasks" tags={autoTags.tasks} tagType="task" color="oklch(0.65 0.15 145)" />
          )}
          {autoTags.cameras.length > 0 && (
            <TagRow title="Cameras" tags={autoTags.cameras} tagType="camera" color="oklch(0.65 0.12 50)" />
          )}
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, color, link }: { label: string; value: string | number; color?: string; link?: string }) {
  const content = (
    <div className="card" style={{ padding: "0.75rem 1rem", display: "flex", flexDirection: "column", gap: "0.25rem", cursor: link ? "pointer" : undefined }}>
      <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>{label}</span>
      <span style={{ fontSize: "1.3rem", fontWeight: 700, color: color ?? "var(--color-text)" }}>{value}</span>
    </div>
  );
  if (link) return <Link to={link} style={{ textDecoration: "none", color: "inherit" }}>{content}</Link>;
  return content;
}

function TagRow({ title, tags, tagType, color }: { title: string; tags: TagGroup[]; tagType: string; color: string }) {
  return (
    <div>
      <h2 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>{title}</h2>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
        {tags.map((t) => (
          <Link
            key={t.name}
            to={`/images/all?tagType=${encodeURIComponent(tagType)}&tag=${encodeURIComponent(t.name)}`}
            style={{
              textDecoration: "none",
              padding: "0.3rem 0.6rem",
              borderRadius: "var(--radius-sm)",
              background: `color-mix(in oklch, ${color} 15%, transparent)`,
              border: `1px solid color-mix(in oklch, ${color} 30%, transparent)`,
              color: "var(--color-text)",
              fontSize: "0.85rem",
              display: "inline-flex",
              alignItems: "center",
              gap: "0.35rem",
              transition: "background 0.15s",
            }}
          >
            <span style={{ fontWeight: 600 }}>{t.name}</span>
            <span style={{ fontSize: "0.75rem", color: "var(--color-text-muted)" }}>{t.count}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}

function ClassCard({ cls, samples }: { cls: ClassItem; samples: ImageItem[] }) {
  const hasSamples = samples.length > 0;
  return (
    <div className="card" style={{ padding: "1rem", overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>{cls.name}</h2>
        <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>id {cls.id}</span>
      </div>
      {hasSamples ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "4px", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
            {samples.slice(0, 8).map((img, idx) => (
              <Link
                key={img.imageRel}
                to={`/image/${encodeURIComponent(img.split)}/${encodeURIComponent(img.name)}`}
                state={{ list: samples, index: idx, classId: String(cls.id) }}
                style={{ aspectRatio: "1", display: "block", background: "var(--color-border)" }}
              >
                <img src={api.imageUrl(img.split, img.name)} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} loading="lazy" />
              </Link>
            ))}
          </div>
          <div style={{ marginTop: "0.75rem", display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <Link to={`/class/${cls.id}`} className="btn btn-ghost" style={{ fontSize: "0.85rem", padding: "0.35rem 0.6rem" }}>View all</Link>
          </div>
        </>
      ) : (
        <div style={{
          padding: "1.5rem 1rem", textAlign: "center", borderRadius: "var(--radius-sm)",
          background: "oklch(0 0 0 / 0.03)", border: "1px dashed var(--color-border)",
        }}>
          <span style={{ fontSize: "0.9rem", color: "var(--color-text-muted)" }}>No detections in dataset</span>
        </div>
      )}
    </div>
  );
}
