"use client";

import { useEffect, useRef } from "react";
import { Rectangle, Tooltip } from "react-leaflet";
import { FUSION_ORDER, MODEL_INFO, SUGGESTED_QUESTIONS, formatConfidence } from "@/lib/models";
import type { AreaAnalysis, AreaCapabilities, FusionAreaResult, FusionType, ImagerySource } from "@/lib/types";

// Colours for what a cell answered. Deliberately not the two model colours (teal and amber), which
// already mean "elementwise" and "cross-attention" everywhere else in the app.
const RANGE_COLOURS: Record<string, string> = {
  "none (0)": "#4b3a7a",
  "1 to 10": "#3b6fa0",
  "11 to 100": "#2ea39a",
  "101 to 1,000": "#8bc34a",
  "more than 1,000": "#f2d23c",
};

export function answerColour(answer: string): string {
  const a = answer.toLowerCase();
  if (a === "yes" || a === "urban") return "#ff7a59";
  if (a === "no" || a === "rural") return "#4a8fd4";
  return RANGE_COLOURS[answer] ?? "#9aa4b2";
}

/** How many windows a square of this side needs: 256 px (2.56 km) steps of half a window. */
export function windowsFor(sizeKm: number): number {
  const sizePx = Math.round((sizeKm * 1000) / 10);
  const perSide = sizePx <= 256 ? 1 : Math.ceil((sizePx - 256) / 128) + 1;
  return perSide * perSide;
}

const SIZE_OPTIONS_KM = [5, 10, 15, 20];

/** The coarse result grid, drawn on the map: each cell coloured by the answer most of the windows
 * over it gave, fainter where they disagreed. */
export function AreaCells({
  result,
  fusion,
  onPick,
}: {
  result: AreaAnalysis;
  fusion: FusionType;
  /** Called with the centre of the square that was clicked. */
  onPick?: (latitude: number, longitude: number) => void;
}) {
  const view = result.results.find((r) => r.fusion === fusion);
  if (!view) return null;
  return (
    <>
      {view.cells.map((cell) => {
        const [west, south, east, north] = cell.bounds_wgs84;
        const colour = answerColour(cell.answer);
        return (
          <Rectangle
            key={`${cell.row}-${cell.col}`}
            bounds={[
              [south, west],
              [north, east],
            ]}
            pathOptions={{
              color: colour,
              weight: 0.5,
              fillColor: colour,
              fillOpacity: 0.15 + 0.45 * cell.agreement,
              className: onPick ? "cursor-pointer" : undefined,
            }}
            eventHandlers={{ click: () => onPick?.((south + north) / 2, (west + east) / 2) }}
          >
            <Tooltip sticky>
              <strong>{cell.answer}</strong>
              <br />
              {Math.round(cell.agreement * 100)}% of {cell.windows} windows agreed ·{" "}
              {formatConfidence(cell.mean_confidence)} sure
              {onPick && (
                <>
                  <br />
                  <em>Click to open this spot with the highlight options</em>
                </>
              )}
            </Tooltip>
          </Rectangle>
        );
      })}
    </>
  );
}

function SharesLine({ view }: { view: FusionAreaResult }) {
  if (view.shares.length === 0) return <span className="text-faint">no answers</span>;
  return (
    <>
      {view.shares.map((s, i) => (
        <span key={s.answer}>
          {i > 0 && <span className="text-faint"> · </span>}
          <span style={{ color: answerColour(s.answer) }}>■</span> {s.answer}{" "}
          <span className="tabular">{Math.round(s.share * 100)}%</span>
        </span>
      ))}
    </>
  );
}

const SOURCE_LABEL: Record<ImagerySource, string> = {
  earth_engine: "Google Earth Engine — latest clear Sentinel-2",
  sentinel_cloudless: "Fixed 2020 Sentinel-2 mosaic",
};

interface AreaPanelProps {
  capabilities: AreaCapabilities | null;
  sizeKm: number;
  onSizeKm: (km: number) => void;
  question: string;
  onQuestion: (q: string) => void;
  source: ImagerySource | "auto";
  onSource: (s: ImagerySource | "auto") => void;
  dateFrom: string;
  dateTo: string;
  onDates: (from: string, to: string) => void;
  busy: boolean;
  canRun: boolean;
  onRun: () => void;
  result: AreaAnalysis | null;
  fusion: FusionType;
  onFusion: (f: FusionType) => void;
}

export function AreaPanel(props: AreaPanelProps) {
  const { capabilities, result } = props;
  const sizes = SIZE_OPTIONS_KM.filter((km) => !capabilities || km <= capabilities.max_size_km);
  const hasEarthEngine = capabilities?.sources.includes("earth_engine") ?? false;
  const effective: ImagerySource | null =
    props.source === "auto" ? (capabilities?.sources[0] ?? null) : props.source;
  const windows = windowsFor(props.sizeKm);
  const models = capabilities?.fusions.length ?? 2;
  const shown = result?.results.find((r) => r.fusion === props.fusion) ?? result?.results[0];
  const present = [...new Set(shown?.cells.map((c) => c.answer) ?? [])];

  // The controls are tall, so when a result arrives scroll the panel down to it.
  const panelRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (result && panelRef.current && resultsRef.current) {
      panelRef.current.scrollTo({ top: resultsRef.current.offsetTop - 8, behavior: "smooth" });
    }
  }, [result]);

  return (
    <div
      ref={panelRef}
      className="relative flex max-h-[calc(100vh-17rem)] min-h-0 w-80 max-w-full flex-col gap-2 overflow-y-auto rounded-sm border border-line-strong bg-ink/95 p-3 text-xs backdrop-blur"
    >
      <div>
        <div className="mb-1 text-[11px] uppercase tracking-wide text-faint">Question</div>
        <input
          value={props.question}
          onChange={(e) => props.onQuestion(e.target.value)}
          aria-label="Question about the area"
          className="w-full rounded-sm border border-line bg-transparent px-2 py-1.5 placeholder:text-faint focus:border-line-strong focus:outline-none"
          placeholder="e.g. Is there a water area?"
        />
        <div className="mt-1.5 flex flex-wrap gap-1">
          {SUGGESTED_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => props.onQuestion(q)}
              className="rounded-sm border border-line px-1.5 py-0.5 text-[11px] text-muted transition hover:border-line-strong hover:text-text"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-faint" htmlFor="area-size">
          Area
        </label>
        <select
          id="area-size"
          value={props.sizeKm}
          onChange={(e) => props.onSizeKm(Number(e.target.value))}
          className="rounded-sm border border-line bg-ink px-2 py-1 focus:outline-none"
        >
          {sizes.map((km) => (
            <option key={km} value={km}>
              {km} × {km} km
            </option>
          ))}
        </select>
        <span className="text-faint tabular">
          {windows} window{windows === 1 ? "" : "s"} × {models} models
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-faint" htmlFor="area-source">
          Imagery
        </label>
        {capabilities && capabilities.sources.length > 1 ? (
          <select
            id="area-source"
            value={props.source}
            onChange={(e) => props.onSource(e.target.value as ImagerySource | "auto")}
            className="rounded-sm border border-line bg-ink px-2 py-1 focus:outline-none"
          >
            <option value="auto">Best available</option>
            {capabilities.sources.map((s) => (
              <option key={s} value={s}>
                {SOURCE_LABEL[s]}
              </option>
            ))}
          </select>
        ) : (
          <span className="text-muted">
            {effective ? SOURCE_LABEL[effective] : "Checking…"}
            {!hasEarthEngine && capabilities && (
              <span className="block text-faint">Google Earth Engine isn&apos;t set up on this server.</span>
            )}
          </span>
        )}
        {effective === "earth_engine" && (
          <div className="flex items-center gap-1.5">
            <input
              type="date"
              value={props.dateFrom}
              onChange={(e) => props.onDates(e.target.value, props.dateTo)}
              aria-label="From date"
              className="min-w-0 flex-1 rounded-sm border border-line bg-ink px-1.5 py-1 focus:outline-none"
            />
            <span className="text-faint">to</span>
            <input
              type="date"
              value={props.dateTo}
              onChange={(e) => props.onDates(props.dateFrom, e.target.value)}
              aria-label="To date"
              className="min-w-0 flex-1 rounded-sm border border-line bg-ink px-1.5 py-1 focus:outline-none"
            />
          </div>
        )}
        {effective === "earth_engine" && (
          <span className="text-faint">Leave the dates empty for the last 90 days.</span>
        )}
      </div>

      <button
        type="button"
        onClick={props.onRun}
        disabled={!props.canRun}
        className="rounded-sm bg-elementwise px-4 py-2.5 text-sm font-semibold text-ink shadow-lg transition hover:brightness-110 disabled:opacity-50"
      >
        {props.busy ? "Sliding the window…" : "Analyse area"}
      </button>
      {props.busy && (
        <p className="text-faint">
          Asking {windows * models} questions, one per window and model. A large area can take a minute or
          two.
        </p>
      )}

      {result && shown && (
        <div ref={resultsRef} className="flex flex-col gap-2 border-t border-line pt-2">
          <div className="flex overflow-hidden rounded-sm border border-line">
            {FUSION_ORDER.filter((f) => result.results.some((r) => r.fusion === f)).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => props.onFusion(f)}
                className={`flex-1 px-2 py-1 transition ${
                  props.fusion === f ? "bg-raised text-text" : "text-muted hover:text-text"
                }`}
              >
                {MODEL_INFO[f].name}
              </button>
            ))}
          </div>

          {result.results.map((r) => (
            <div key={r.fusion} className="leading-relaxed">
              <span className={MODEL_INFO[r.fusion].text}>{MODEL_INFO[r.fusion].name}:</span>{" "}
              <SharesLine view={r} />
              {r.failed_windows > 0 && (
                <span className="text-danger"> · {r.failed_windows} windows failed</span>
              )}
            </div>
          ))}

          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {present.map((a) => (
              <span key={a}>
                <span style={{ color: answerColour(a) }}>■</span> {a}
              </span>
            ))}
          </div>

          <p className="text-text">
            Click any square on the map to open that spot (about 2.5 km) in the workspace, where you can
            ask questions and highlight roads, buildings, water and more.
          </p>
          {result.note && <p className="text-muted">{result.note}</p>}
          <p className="text-faint">
            {result.captured ? `Imagery: ${result.captured}. ` : ""}A rough map: each cell is about{" "}
            {((result.stride_px * 10) / 1000).toFixed(2)} km wide and shows what the 2.56 km windows over
            it answered, fainter where they disagreed. It is not an outline, and the models haven&apos;t
            been measured on large areas.
          </p>
        </div>
      )}
    </div>
  );
}
