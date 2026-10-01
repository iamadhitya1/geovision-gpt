"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

/** Thumbnails sit behind bearer auth, which a plain <img src> can't send — so fetch the
 * bytes with the token and hand the browser a local object URL instead. */
export function AuthedImage({
  imageId,
  alt,
  className,
  scanning = false,
  overlay,
}: {
  imageId: string;
  alt: string;
  className?: string;
  scanning?: boolean;
  /** Rendered absolutely inside this component's own relative wrapper, e.g. an attention
   * heatmap — the source images are square and object-cover fills a square container
   * exactly, so an `inset-0` overlay lines up with the rendered image with no extra math. */
  overlay?: ReactNode;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    api
      .thumbnailBlob(imageId)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setSrc(url);
      })
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [imageId]);

  return (
    <div className={`relative overflow-hidden bg-raised ${scanning ? "scanning" : ""} ${className ?? ""}`}>
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element -- blob: URL, next/image can't optimise it
        <img src={src} alt={alt} className="h-full w-full object-cover" />
      ) : (
        <div className="flex h-full w-full items-center justify-center font-mono text-xs text-faint">
          {failed ? "no preview" : "loading…"}
        </div>
      )}
      {src && overlay}
    </div>
  );
}
