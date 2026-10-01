import type { AttentionHeatmap } from "@/lib/types";

/** Where the cross-attention model looked when answering — one semi-transparent cell per
 * image patch, opacity scaled to that patch's share of attention. Interpretability only:
 * shows attention weight, not a verified location for any object in the scene. */
export function AttentionOverlay({ heatmap }: { heatmap: AttentionHeatmap }) {
  const max = Math.max(...heatmap.weights, 1e-8);

  return (
    <div
      className="absolute inset-0 grid"
      style={{ gridTemplateColumns: `repeat(${heatmap.grid_size}, 1fr)` }}
      aria-hidden
    >
      {heatmap.weights.map((w, i) => (
        <div
          key={i}
          style={{ backgroundColor: "#f0b13e", opacity: (w / max) * 0.75 }}
        />
      ))}
    </div>
  );
}
