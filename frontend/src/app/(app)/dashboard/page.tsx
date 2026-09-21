"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { FUSION_ORDER, MODEL_INFO, formatConfidence } from "@/lib/models";
import { QUESTION_ONLY, QUESTION_TYPES, RESULTS, TEST_SET_SIZE } from "@/lib/results";
import type { VqaStats } from "@/lib/types";

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="plate p-4">
      <div className="font-mono text-[11px] uppercase tracking-wider text-faint">{label}</div>
      <div className="mt-2 font-display text-3xl font-bold tabular tracking-tight">{value}</div>
      {sub && <div className="mt-1 font-mono text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

// "question_only" is the no-image reference row: drawn neutral so it reads as a floor, not a model.
function AccuracyBar({
  value,
  fusion,
}: {
  value: number | null;
  fusion: keyof typeof MODEL_INFO | "question_only";
}) {
  const info = fusion === "question_only" ? { bg: "bg-muted/60" } : MODEL_INFO[fusion];
  return (
    <div className="flex items-center gap-3">
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-raised">
        {value !== null ? (
          <div className={`h-full rounded-full ${info.bg}`} style={{ width: `${value}%` }} />
        ) : (
          <div className="h-full w-full border-y border-dashed border-line-strong" />
        )}
      </div>
      <span className={`w-14 text-right font-mono text-xs tabular ${value === null ? "text-faint" : ""}`}>
        {value !== null ? `${value.toFixed(1)}%` : "pending"}
      </span>
    </div>
  );
}

function Benchmark() {
  const base = RESULTS.elementwise;
  const cross = RESULTS.cross_attention;
  const both = base !== null && cross !== null;
  const overallGap = both ? cross.overall - base.overall : 0;
  return (
    <section className="space-y-4">
      <div>
        <h2 className="font-display text-xl font-bold tracking-tight">Model accuracy</h2>
        <p className="mt-1 text-sm text-muted">
          Measured on RSVQA-LR&apos;s held-out test split ({TEST_SET_SIZE.toLocaleString()} questions the
          models never saw during training).
        </p>
      </div>

      <div className="plate p-5">
        <div className="mb-5 flex flex-wrap gap-x-6 gap-y-2">
          {FUSION_ORDER.map((f) => {
            const r = RESULTS[f];
            return (
              <div key={f} className="min-w-40">
                <div className={`font-mono text-[11px] uppercase tracking-wider ${MODEL_INFO[f].text}`}>
                  {MODEL_INFO[f].label}
                </div>
                <div className="mt-1 font-display text-4xl font-bold tabular tracking-tight">
                  {r ? `${r.average.toFixed(1)}%` : "—"}
                </div>
                <div className="font-mono text-[11px] text-muted">
                  {r ? `average accuracy · ${r.overall.toFixed(1)}% overall` : "training run in progress"}
                </div>
              </div>
            );
          })}
        </div>

        <div className="mb-4 flex flex-wrap gap-x-5 gap-y-1 border-t border-line pt-4 font-mono text-[11px] text-muted">
          {FUSION_ORDER.map((f) => (
            <span key={f} className="flex items-center gap-1.5">
              <span className={`h-2 w-2 rounded-full ${MODEL_INFO[f].bg}`} />
              {MODEL_INFO[f].name}
            </span>
          ))}
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-muted/60" />
            No image: answer by question text alone
          </span>
        </div>

        <div className="space-y-4">
          {QUESTION_TYPES.map((t) => (
            <div key={t.key} className="grid gap-1 sm:grid-cols-[11rem_1fr] sm:items-center sm:gap-4">
              <div>
                <div className="text-sm">{t.label}</div>
                <div className="text-[11px] text-faint">
                  {t.hint} · {t.n.toLocaleString()} questions
                </div>
              </div>
              <div className="space-y-1.5">
                {FUSION_ORDER.map((f) => (
                  <AccuracyBar key={f} fusion={f} value={RESULTS[f]?.byType[t.key] ?? null} />
                ))}
                <AccuracyBar fusion="question_only" value={QUESTION_ONLY.byType[t.key]} />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-5 space-y-2 border-t border-line pt-4 text-xs leading-relaxed text-muted">
          {both ? (
            <p>
              <span className="text-text">Overall, the two models are effectively tied</span>:{" "}
              {Math.abs(overallGap).toFixed(1)} points apart on {TEST_SET_SIZE.toLocaleString()} questions.
              Cross-attention is slightly ahead on counting and slightly behind on comparison. The
              larger gap in <em>average</em> accuracy comes from rural/urban, which has only{" "}
              {QUESTION_TYPES[0].n} test questions but counts for a quarter of that average. Each model
              was trained once, so gaps this small shouldn&apos;t be read as a real difference.
            </p>
          ) : (
            <p>The cross-attention numbers appear here as soon as its evaluation finishes.</p>
          )}
          <p>
            <span className="text-text">How much does the image add?</span> A predictor that never
            looks at the image and just answers each question with its most common training answer
            already scores {QUESTION_ONLY.overall.toFixed(1)}% overall ({QUESTION_ONLY.average.toFixed(1)}%
            average), because this dataset&apos;s questions are templated. The models beat it by about
            two points overall: clearly on rural/urban ({QUESTION_ONLY.byType.rural_urban.toFixed(0)}%
            → 92% and 81%), slightly on presence and comparison, and not at all on counting.
          </p>
          <p>
            Counting is the hard case for every model of this size. The published LiT-4-RSVQA paper
            reports 64.6%, but it trained on a much larger dataset, so that comparison is indicative,
            not like-for-like.
          </p>
        </div>
      </div>
    </section>
  );
}

function Usage() {
  const [stats, setStats] = useState<VqaStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .stats()
      .then(setStats)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't load usage figures."));
  }, []);

  if (error) {
    return (
      <p role="alert" className="rounded-sm border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
        {error}
      </p>
    );
  }
  if (!stats) return <p className="font-mono text-sm text-faint">loading…</p>;

  const { agreement } = stats;
  const agreementText = agreement.compared
    ? `${Math.round((agreement.agreed / agreement.compared) * 100)}%`
    : "—";

  return (
    <section className="space-y-4">
      <div>
        <h2 className="font-display text-xl font-bold tracking-tight">Your activity</h2>
        <p className="mt-1 text-sm text-muted">Live figures from the questions you&apos;ve asked so far.</p>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Tile label="Questions asked" value={stats.total_questions.toLocaleString()} />
        <Tile label="Scenes queried" value={stats.images_queried.toLocaleString()} />
        <Tile
          label="Models agree"
          value={agreementText}
          sub={agreement.compared ? `${agreement.agreed} of ${agreement.compared} questions` : "needs both models loaded"}
        />
        <Tile
          label="Mean confidence"
          value={
            stats.by_fusion.length
              ? stats.by_fusion.map((u) => formatConfidence(u.mean_confidence)).join(" · ")
              : "—"
          }
          sub={stats.by_fusion.map((u) => MODEL_INFO[u.fusion].short.toLowerCase()).join(" · ")}
        />
      </div>

      {stats.recent.length > 0 && (
        <div className="plate">
          <div className="border-b border-line px-4 py-2.5 font-mono text-[11px] uppercase tracking-wider text-faint">
            Recent questions
          </div>
          <ul className="divide-y divide-line">
            {stats.recent.map((q) => (
              <li key={q.id}>
                <Link
                  href={`/images/${q.image_id}`}
                  className="flex items-center justify-between gap-4 px-4 py-2.5 text-sm transition hover:bg-raised"
                >
                  <span className="truncate">“{q.question}”</span>
                  <span className="flex shrink-0 gap-3 font-mono text-xs">
                    {FUSION_ORDER.map((f) => {
                      const a = q.answers.find((x) => x.fusion === f);
                      return a ? (
                        <span key={f} className={MODEL_INFO[f].text}>
                          {a.answer}
                        </span>
                      ) : null;
                    })}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

export default function DashboardPage() {
  return (
    <div className="space-y-10">
      <div>
        <h1 className="font-display text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="mt-1 text-sm text-muted">How well the models perform, and how you&apos;ve been using them.</p>
      </div>
      <Benchmark />
      <Usage />
    </div>
  );
}
