"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { AnswerComparison } from "@/components/AnswerComparison";
import { AuthedImage } from "@/components/AuthedImage";
import { api, ApiError } from "@/lib/api";
import { FUSION_ORDER, MODEL_INFO, SUGGESTED_QUESTIONS, formatBytes } from "@/lib/models";
import type { SatelliteImage, VqaQuery } from "@/lib/types";

function centreOf(bounds: [number, number, number, number]): string {
  const [minX, minY, maxX, maxY] = bounds;
  const lon = (minX + maxX) / 2;
  const lat = (minY + maxY) / 2;
  return `${Math.abs(lat).toFixed(4)}°${lat >= 0 ? "N" : "S"}  ${Math.abs(lon).toFixed(4)}°${lon >= 0 ? "E" : "W"}`;
}

function Readout({ image }: { image: SatelliteImage }) {
  const m = image.metadata;
  const rows: [string, string][] = [["File", image.original_filename]];
  if (m) {
    rows.push(["Size", `${m.width_px} × ${m.height_px} px · ${m.band_count} bands`]);
    if (m.gsd_meters_approx) rows.push(["Resolution", `${m.gsd_meters_approx.toFixed(1)} m per pixel`]);
    if (m.crs) rows.push(["CRS", m.crs]);
    if (m.bounds_wgs84) rows.push(["Centre", centreOf(m.bounds_wgs84)]);
    if (m.cloud_cover_percent !== null) rows.push(["Cloud cover", `${m.cloud_cover_percent.toFixed(1)}%`]);
  }
  rows.push(["Format", `${image.image_format} · ${formatBytes(image.file_size_bytes)}`]);

  return (
    <dl className="divide-y divide-line font-mono text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-4 py-2">
          <dt className="text-faint">{k}</dt>
          <dd className="truncate text-right tabular text-text/90" title={v}>
            {v}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export default function WorkspacePage() {
  const { id } = useParams<{ id: string }>();
  const [image, setImage] = useState<SatelliteImage | null>(null);
  const [queries, setQueries] = useState<VqaQuery[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.getImage(id), api.history(id)])
      .then(([img, hist]) => {
        if (cancelled) return;
        setImage(img);
        setQueries([...hist.items].sort((a, b) => b.created_at.localeCompare(a.created_at)));
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof ApiError ? err.message : "Couldn't load this image.");
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  const ask = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || asking) return;
      setError(null);
      setAsking(true);
      try {
        const result = await api.ask(id, trimmed);
        setQueries((prev) => [result, ...prev]);
        setQuestion("");
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Something went wrong. Try again.");
      } finally {
        setAsking(false);
      }
    },
    [id, asking],
  );

  if (loadError) {
    return (
      <div className="space-y-4">
        <p role="alert" className="text-danger">
          {loadError}
        </p>
        <Link href="/images" className="text-sm text-elementwise hover:underline">
          ← Back to library
        </Link>
      </div>
    );
  }
  if (!image) return <p className="font-mono text-sm text-faint">loading…</p>;

  const ready = image.status === "ready";
  const [latest, ...earlier] = queries;

  return (
    <div className="space-y-6">
      <Link href="/images" className="text-sm text-muted hover:text-text">
        ← Library
      </Link>

      <div className="grid gap-8 lg:grid-cols-[5fr_7fr]">
        <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
          <div className="plate">
            {image.thumbnail_available ? (
              <AuthedImage
                imageId={image.id}
                alt={image.original_filename}
                className="aspect-square"
                scanning={asking}
              />
            ) : (
              <div className="flex aspect-square items-center justify-center bg-raised font-mono text-xs text-faint">
                no preview available
              </div>
            )}
          </div>
          <Readout image={image} />
        </aside>

        <section className="space-y-6">
          <div>
            <h1 className="font-display text-3xl font-bold tracking-tight">Ask about this scene</h1>
            <p className="mt-1 text-sm text-muted">
              The models pick from 1,000 learned answers, so they work best on the kinds of questions
              they were trained on: presence, counts, and rural or urban.
            </p>
          </div>

          {!ready && (
            <p role="alert" className="rounded-sm border border-cross/40 bg-cross/10 px-3 py-2 text-sm text-cross">
              {image.status === "failed"
                ? `This image couldn't be processed${image.error_message ? `: ${image.error_message}` : "."}`
                : "This image is still being processed. Reload in a moment."}
            </p>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              void ask(question);
            }}
            className="space-y-3"
          >
            <div className="flex gap-2">
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                maxLength={500}
                disabled={!ready || asking}
                placeholder="e.g. Is there a water area?"
                aria-label="Your question"
                className="min-w-0 flex-1 rounded-sm border border-line bg-panel px-3 py-2.5 text-sm placeholder:text-faint focus:border-elementwise focus:outline-none disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={!ready || asking || !question.trim()}
                className="rounded-sm bg-elementwise px-5 py-2.5 text-sm font-semibold text-ink transition hover:brightness-110 disabled:opacity-40"
              >
                {asking ? "Analysing…" : "Ask"}
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  disabled={!ready || asking}
                  onClick={() => void ask(q)}
                  className="rounded-full border border-line px-3 py-1 text-xs text-muted transition hover:border-elementwise/60 hover:text-text disabled:opacity-40"
                >
                  {q}
                </button>
              ))}
            </div>
          </form>

          {error && (
            <p role="alert" className="rounded-sm border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
              {error}
            </p>
          )}

          {latest ? (
            <div key={latest.id} className="rise space-y-3">
              <h2 className="font-mono text-[11px] uppercase tracking-wider text-faint">Latest answer</h2>
              <p className="font-display text-xl font-medium tracking-tight">“{latest.question}”</p>
              <AnswerComparison query={latest} />
            </div>
          ) : (
            <div className="rounded-sm border border-dashed border-line p-6 text-sm text-muted">
              <p>Answers from both models will appear here, side by side:</p>
              <ul className="mt-3 space-y-2">
                {FUSION_ORDER.map((f) => (
                  <li key={f} className="flex gap-3">
                    <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${MODEL_INFO[f].bg}`} />
                    <span>
                      <span className={MODEL_INFO[f].text}>{MODEL_INFO[f].label}</span> —{" "}
                      {MODEL_INFO[f].blurb}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {earlier.length > 0 && (
            <div className="space-y-2">
              <h2 className="font-mono text-[11px] uppercase tracking-wider text-faint">
                Earlier questions · {earlier.length}
              </h2>
              <div className="divide-y divide-line border-y border-line">
                {earlier.map((q) => (
                  <details key={q.id} className="group py-3">
                    <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-sm">
                      <span className="truncate">“{q.question}”</span>
                      <span className="flex shrink-0 gap-2 font-mono text-xs">
                        {FUSION_ORDER.map((f) => {
                          const a = q.answers.find((x) => x.fusion === f);
                          return a ? (
                            <span key={f} className={MODEL_INFO[f].text}>
                              {a.answer}
                            </span>
                          ) : null;
                        })}
                      </span>
                    </summary>
                    <div className="mt-3">
                      <AnswerComparison query={q} />
                    </div>
                  </details>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
