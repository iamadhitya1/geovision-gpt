import { FUSION_ORDER, MODEL_INFO, formatConfidence } from "@/lib/models";
import type { ConceptHighlight, VqaAnswer, VqaQuery } from "@/lib/types";

// Below this, the winning class barely beats the others — worth telling the viewer so a
// confident-looking "yes" doesn't get mistaken for a confident model.
const LOW_CONFIDENCE = 0.2;

function AnswerColumn({
  answer,
  onExplain,
  explainStatus,
}: {
  answer: VqaAnswer;
  onExplain?: () => void;
  explainStatus?: "idle" | "loading" | "shown";
}) {
  const info = MODEL_INFO[answer.fusion];
  const low = answer.confidence < LOW_CONFIDENCE;
  const top = Math.max(answer.confidence, ...answer.alternatives.map((a) => a.confidence));
  const showExplain = answer.fusion === "cross_attention" && onExplain;

  return (
    <div className={`rounded-sm border bg-panel p-4 ${info.border}/30`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="space-y-0.5">
          <div className={`font-mono text-[11px] uppercase tracking-wider ${info.text}`}>{info.name}</div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-faint">{info.short}</div>
        </div>
        {showExplain && (
          <button
            type="button"
            onClick={onExplain}
            title="Where the model's attention went while it answered. This is not a map of the answer — use the highlight button below for that."
            disabled={explainStatus === "loading"}
            className={`shrink-0 rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider transition disabled:opacity-50 ${
              explainStatus === "shown"
                ? `${info.border} ${info.text}`
                : "border-line text-muted hover:border-cross/60 hover:text-text"
            }`}
          >
            {explainStatus === "loading"
              ? "looking…"
              : explainStatus === "shown"
                ? "hide attention"
                : "show model attention"}
          </button>
        )}
      </div>

      <div className="mt-3 font-display text-4xl font-bold leading-none tracking-tight text-balance">
        {answer.answer}
      </div>

      <div className="mt-3 flex items-center gap-3">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-raised">
          <div
            className={`h-full rounded-full ${info.bg}`}
            style={{ width: `${Math.max(answer.confidence * 100, 1.5)}%` }}
          />
        </div>
        <span className="font-mono text-sm tabular">{formatConfidence(answer.confidence)}</span>
      </div>

      {answer.note && <p className="mt-2 text-xs text-muted">{answer.note}</p>}

      {low && (
        <p className="mt-2 text-xs text-muted">
          Low confidence — the model is close to guessing. The question may be outside what it was
          trained on.
        </p>
      )}

      {answer.alternatives.length > 1 && (
        <ul className="mt-4 space-y-1.5 border-t border-line pt-3">
          {answer.alternatives.slice(1).map((alt) => (
            <li key={alt.answer} className="flex items-center gap-2 font-mono text-xs text-muted">
              <span className="w-24 truncate text-text/80" title={alt.answer}>
                {alt.answer}
              </span>
              <span className="h-1 flex-1 overflow-hidden rounded-full bg-raised">
                <span
                  className={`block h-full rounded-full ${info.bg} opacity-40`}
                  style={{ width: `${(alt.confidence / top) * 100}%` }}
                />
              </span>
              <span className="w-12 text-right tabular">{formatConfidence(alt.confidence)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// Answers that mean "there isn't any" — highlighting would have nothing to point at.
const NEGATIVE_ANSWERS = new Set(["no", "none (0)"]);

/** Highlighting is offered for a question that names something it can point at. It locates
 * the thing; whether it's there at all is the VQA models' call, so for a presence or count
 * question it's offered only when at least one model doesn't say "none". A comparison's "no"
 * only says one count isn't larger, so there it's always offered. */
export interface HighlightControl {
  status: "idle" | "loading" | "shown";
  onToggle: () => void;
  /** What the highlighting step returned, to show its methods and caveats beside the image. */
  result: ConceptHighlight | null;
  error: string | null;
}

function joinLabels(labels: string[]): string {
  return labels.length === 2 ? `${labels[0]} and ${labels[1]}` : labels.join(", ");
}

function HighlightRow({
  targets,
  kind,
  answers,
  control,
}: {
  targets: string[];
  kind: "presence" | "count" | "comparison";
  answers: VqaAnswer[];
  control: HighlightControl;
}) {
  const what = joinLabels(targets);
  const nothingThere =
    kind !== "comparison" &&
    answers.length > 0 &&
    answers.every((a) => NEGATIVE_ANSWERS.has(a.answer));
  if (nothingThere) {
    return (
      <p className="font-mono text-xs text-muted">
        {answers.length > 1 ? "Both models say" : "The model says"} there&apos;s no {what} here, so
        there&apos;s nothing to highlight.
      </p>
    );
  }

  const { status, onToggle, result, error } = control;
  return (
    <div className="space-y-2 rounded-sm border border-line bg-panel/60 p-3">
      <button
        type="button"
        onClick={onToggle}
        disabled={status === "loading"}
        className={`rounded-full border px-3 py-1 font-mono text-[11px] uppercase tracking-wider transition disabled:opacity-50 ${
          status === "shown"
            ? "border-ok text-ok"
            : "border-line text-muted hover:border-ok/60 hover:text-text"
        }`}
      >
        {status === "loading"
          ? "locating…"
          : status === "shown"
            ? `hide ${what}`
            : `highlight ${what} in the image`}
      </button>
      {error && (
        <p role="alert" className="text-xs text-danger">
          {error}
        </p>
      )}
      {status === "shown" && result && (
        <ul className="space-y-2 text-xs text-muted">
          {result.layers.map((layer) => (
            <li key={layer.concept} className="space-y-0.5">
              <p>
                <span
                  aria-hidden
                  className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                  style={{ backgroundColor: layer.color }}
                />
                <span className="text-text/90">{layer.label}</span> ·{" "}
                {layer.coverage > 0
                  ? `${Math.max(1, Math.round(layer.coverage * 100))}% of the scene`
                  : "none found"}{" "}
                · from <span className="text-text/90">{layer.method}</span>
              </p>
              {layer.caveat && <p className="text-cross">{layer.caveat}</p>}
            </li>
          ))}
          {result.caveat && <li className="text-cross">{result.caveat}</li>}
          <li>
            This points at the {what}; it isn&apos;t something the answering model learned to do.
          </li>
        </ul>
      )}
    </div>
  );
}

export function AnswerComparison({
  query,
  onExplain,
  explainStatus = "idle",
  highlight,
}: {
  query: VqaQuery;
  /** Omit to hide the "show where it looked" control entirely (e.g. for past/collapsed
   * questions in history, where re-explaining isn't worth the extra API call). */
  onExplain?: () => void;
  explainStatus?: "idle" | "loading" | "shown";
  /** Omit to hide the "highlight in the image" control (same reasoning as `onExplain`). */
  highlight?: HighlightControl;
}) {
  const answers = FUSION_ORDER.map((f) => query.answers.find((a) => a.fusion === f)).filter(
    (a): a is VqaAnswer => a !== undefined,
  );
  const both = answers.length === 2;
  const agree = both && answers[0].answer === answers[1].answer;

  return (
    <div className="space-y-3">
      <div className={`grid gap-3 ${both ? "md:grid-cols-2" : ""}`}>
        {answers.map((a) => (
          <AnswerColumn
            key={a.fusion}
            answer={a}
            onExplain={a.fusion === "cross_attention" ? onExplain : undefined}
            explainStatus={explainStatus}
          />
        ))}
      </div>
      {highlight && query.highlight_targets && query.highlight_targets.length > 0 && (
        <HighlightRow
          targets={query.highlight_targets}
          kind={query.highlight_kind ?? "presence"}
          answers={answers}
          control={highlight}
        />
      )}
      {both && (
        <p
          className={`font-mono text-xs ${agree ? "text-ok" : "text-cross"}`}
          role="status"
        >
          {agree ? "● Both models give the same answer." : "● The models disagree on this one."}
        </p>
      )}
      {!both && answers.length === 1 && (
        <p className="font-mono text-xs text-muted">
          Only the {MODEL_INFO[answers[0].fusion].label.toLowerCase()} model is loaded on this server.
        </p>
      )}
    </div>
  );
}
