"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { FUSION_ORDER, MODEL_INFO, formatConfidence } from "@/lib/models";
import {
  COUNT_IN_RANGES,
  GUJARAT,
  QUESTION_ONLY,
  QUESTION_TYPES,
  RESULTS,
  SEEDS,
  TEST_SET_SIZE,
} from "@/lib/results";
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

function GujaratBenchmark() {
  const mark = GUJARAT.passMark;
  const passes = (key: string, value: number) =>
    key === "count" ? value > GUJARAT.countBaseline : value >= mark;
  return (
    <section className="space-y-4">
      <div>
        <h2 className="font-display text-xl font-bold tracking-tight">Accuracy in Gujarat</h2>
        <p className="mt-1 text-sm text-muted">
          What the app serves: both models fine-tuned on labelled Gujarat imagery, scored once on{" "}
          {GUJARAT.districts} districts they never saw ({GUJARAT.testTiles} tiles). Balanced accuracy
          averages the rate of correct &quot;yes&quot; and correct &quot;no&quot; answers, so always
          saying the same thing scores 50%.
        </p>
      </div>

      <div className="plate overflow-x-auto p-5">
        <table className="w-full min-w-[36rem] text-sm">
          <thead>
            <tr className="font-mono text-[11px] uppercase tracking-wider text-faint">
              <th className="pb-3 text-left font-normal">Question</th>
              {FUSION_ORDER.map((f) => (
                <th key={f} colSpan={2} className={`pb-3 text-center font-normal ${MODEL_INFO[f].text}`}>
                  {MODEL_INFO[f].name}
                </th>
              ))}
            </tr>
            <tr className="font-mono text-[10px] uppercase tracking-wider text-faint">
              <th />
              {FUSION_ORDER.map((f) => (
                <FragmentHeads key={f} />
              ))}
            </tr>
          </thead>
          <tbody>
            {GUJARAT.rows.map((row) => (
              <tr key={row.key} className="border-t border-line align-top">
                <td className="py-3 pr-3">
                  <div>{row.label}</div>
                  <div className="text-[11px] text-faint">{row.hint}</div>
                </td>
                {FUSION_ORDER.map((f) => {
                  const now = row.now[f];
                  return (
                    <td key={f} colSpan={2} className="py-3 text-center">
                      <span className="font-mono text-xs tabular text-faint">{row.old[f].toFixed(1)}%</span>
                      <span className="mx-1.5 text-faint">→</span>
                      <span
                        className={`font-mono text-sm font-semibold tabular ${
                          passes(row.key, now) ? "text-ok" : "text-text"
                        }`}
                      >
                        {now.toFixed(1)}%
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        <div className="mt-5 space-y-2 border-t border-line pt-4 text-xs leading-relaxed text-muted">
          <p>
            <span className="text-text">Old → Gujarat model.</span> Green means the pass rule fixed
            beforehand was met: at least {mark}% balanced accuracy for water, forest and rural/urban, and
            building counts placed in the right range more often than always naming the commonest range
            ({GUJARAT.countBaseline}%). <span className="text-text">It is not fully met:</span> forest
            misses for both models, and water for cross-attention. The old models, trained on the
            Netherlands, said &quot;yes&quot; to almost every presence question in Gujarat, which is why
            they sit near 50% (chance).
          </p>
          <p>
            <span className="text-text">Read these with care.</span> The tiles include extra urban and
            water examples; on tiles drawn evenly over a district the figures are lower (water about
            65–70%). The right answers come from map products made by models, not from a ground survey.
            Gujarat only. About {GUJARAT.testTiles} tiles from {GUJARAT.districts} districts leave a few
            points of noise in every figure.
          </p>
          <p>
            <span className="text-text">The original data was not harmed:</span> the Gujarat elementwise
            model scores {GUJARAT.elementwiseOnLr}% on the original RSVQA-LR test set, within the
            run-to-run spread of the figures below. The Gujarat cross-attention model has not had that
            check.
          </p>
        </div>
      </div>
    </section>
  );
}

function FragmentHeads() {
  return (
    <>
      <th className="pb-2 text-right font-normal">old</th>
      <th className="pb-2 text-left font-normal">Gujarat</th>
    </>
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
        <h2 className="font-display text-xl font-bold tracking-tight">
          Original models: accuracy on RSVQA-LR
        </h2>
        <p className="mt-1 text-sm text-muted">
          The models <em>before</em> the Gujarat fine-tuning, trained on the Netherlands (and aerial
          US cities) and measured on RSVQA-LR&apos;s held-out test split ({TEST_SET_SIZE.toLocaleString()}{" "}
          questions the models never saw during training). Each figure is the mean of {SEEDS} separate
          training runs; ± is how much those runs differed. The app now serves the Gujarat versions
          shown above.
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
                  {r
                    ? `average accuracy · ${r.overall.toFixed(1)}% overall (±${r.overallSpread.toFixed(1)})`
                    : "training run in progress"}
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

          <div className="grid gap-1 sm:grid-cols-[11rem_1fr] sm:items-center sm:gap-4">
            <div>
              <div className="text-sm">Counting, in ranges</div>
              <div className="text-[11px] text-faint">
                0 / 1–10 / 11–100 / 101–1,000 / 1,000+ · {COUNT_IN_RANGES.n.toLocaleString()} questions
              </div>
            </div>
            <div className="space-y-1.5">
              <AccuracyBar fusion="elementwise" value={COUNT_IN_RANGES.elementwise} />
              <AccuracyBar fusion="cross_attention" value={COUNT_IN_RANGES.cross_attention} />
              <AccuracyBar fusion="question_only" value={COUNT_IN_RANGES.question_only} />
            </div>
          </div>
        </div>

        <div className="mt-5 space-y-2 border-t border-line pt-4 text-xs leading-relaxed text-muted">
          {both ? (
            <p>
              <span className="text-text">
                Across {SEEDS} training runs each, the simple elementwise model comes out ahead
              </span>
              : {Math.abs(overallGap).toFixed(1)} points overall on {TEST_SET_SIZE.toLocaleString()}{" "}
              questions, and all three elementwise runs scored above all three cross-attention runs
              (at the closest by 0.06 points, so this is a consistent edge, not a large one). The richer
              cross-attention fusion is also less steady from run to run. Its lower <em>average</em>{" "}
              accuracy comes mostly from rural/urban, which has only {QUESTION_TYPES[0].n} test
              questions and swung between 54% and 89% across its runs — read that row with care. An
              earlier comparison of one run each showed them tied; that turned out to be luck of the draw.
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
            → about 89% and 74%, the three-run means), slightly on presence and comparison, and not at all on counting when it
            has to be an exact number.
          </p>
          <p>
            <span className="text-text">Counting is far better than the exact-match bar suggests.</span>{" "}
            Nobody can read &quot;158 buildings&quot; versus &quot;159&quot; from 10 m imagery, so the original
            RSVQA paper scores counts in five ranges, and so does the last row above (the app answers
            count questions this way too). Placing a count in the right range, the elementwise model
            gets {COUNT_IN_RANGES.elementwise.toFixed(1)}% and cross-attention{" "}
            {COUNT_IN_RANGES.cross_attention.toFixed(1)}%, against {COUNT_IN_RANGES.question_only.toFixed(1)}%
            for a no-image guess. So the image genuinely helps the elementwise model on counting (about{" "}
            {Math.round(COUNT_IN_RANGES.elementwise - COUNT_IN_RANGES.question_only)} points), while
            cross-attention does barely better than guessing from the question alone. That gap held up
            across training runs (elementwise on {COUNT_IN_RANGES.elementwise_runs} runs, cross-attention
            on {COUNT_IN_RANGES.cross_attention_runs}). The original RSVQA paper reported 67% on this
            task, but that isn&apos;t a like-for-like comparison: it used different models and training.
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
      <GujaratBenchmark />
      <Benchmark />
      <Usage />
    </div>
  );
}
