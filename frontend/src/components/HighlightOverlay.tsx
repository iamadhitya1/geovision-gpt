import type { ConceptHighlight } from "@/lib/types";

/** The server draws the highlight itself: a transparent PNG with everything outside the
 * concept dimmed and the concept outlined, so the matching areas stand out in the
 * foreground. It's the same shape as the image, so an `inset-0` overlay lines up exactly. */
export function HighlightOverlay({ highlight }: { highlight: ConceptHighlight }) {
  return (
    // eslint-disable-next-line @next/next/no-img-element -- data: URL, next/image can't optimise it
    <img
      src={`data:image/png;base64,${highlight.overlay_png_base64}`}
      alt=""
      aria-hidden
      className="pointer-events-none absolute inset-0 h-full w-full object-cover"
    />
  );
}
