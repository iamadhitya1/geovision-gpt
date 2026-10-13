"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { SatelliteImage } from "@/lib/types";

// Leaflet touches `window` on import, so it can only be loaded in the browser.
const MapView = dynamic(() => import("@/components/MapView"), {
  ssr: false,
  loading: () => <div className="plate h-[28rem] animate-pulse" />,
});

export default function MapPage() {
  const [images, setImages] = useState<SatelliteImage[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listImages()
      .then((page) => setImages(page.items))
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't load your images."));
  }, []);

  const located = images?.filter((i) => i.metadata?.bounds_wgs84).length ?? 0;
  const unlocated = (images?.length ?? 0) - located;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-bold tracking-tight">Map</h1>
          <p className="mt-1 text-sm text-muted">
            Browse anywhere, pick a place, and ask the models about it. Your own georeferenced
            scenes show up as footprints.
          </p>
        </div>
        {images && (
          <div className="font-mono text-sm text-muted tabular">
            {located} on map
            {unlocated > 0 && <span className="text-faint"> · {unlocated} without coordinates</span>}
          </div>
        )}
      </div>

      {error && (
        <p role="alert" className="rounded-sm border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
          {error}
        </p>
      )}

      {images && <MapView images={images} />}
    </div>
  );
}
