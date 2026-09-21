import { FUSION_ORDER, MODEL_INFO, formatConfidence } from "@/lib/models";
import type { VqaAnswer, VqaQuery } from "@/lib/types";

// Below this, the winning class barely beats the others — worth telling the viewer so a
// confident-looking "yes" doesn't get mistaken for a confident model.
const LOW_CONFIDENCE = 0.2;

function AnswerColumn({ answer }: { answer: VqaAnswer }) {
  const info = MODEL_INFO[answer.fusion];
  const low = answer.confidence < LOW_CONFIDENCE;
  const top = Math.max(answer.confidence, ...answer.alternatives.map((a) => a.confidence));

  return (
    <div className={`rounded-sm border bg-panel p-4 ${info.border}/30`}>
      <div className="flex items-baseline justify-between">
        <span className={`font-mono text-[11px] uppercase tracking-wider ${info.text}`}>{info.label}</span>
        <span className="font-mono text-[10px] uppercase tracking-wider text-faint">{info.short}</span>
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

export function AnswerComparison({ query }: { query: VqaQuery }) {
  const answers = FUSION_ORDER.map((f) => query.answers.find((a) => a.fusion === f)).filter(
    (a): a is VqaAnswer => a !== undefined,
  );
  const both = answers.length === 2;
  const agree = both && answers[0].answer === answers[1].answer;

  return (
    <div className="space-y-3">
      <div className={`grid gap-3 ${both ? "md:grid-cols-2" : ""}`}>
        {answers.map((a) => (
          <AnswerColumn key={a.fusion} answer={a} />
        ))}
      </div>
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
