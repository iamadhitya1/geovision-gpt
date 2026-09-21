"use client";

import "leaflet/dist/leaflet.css";
import L from "leaflet";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ImageOverlay, MapContainer, Rectangle, TileLayer, Tooltip, useMap } from "react-leaflet";
import { api } from "@/lib/api";
import type { SatelliteImage } from "@/lib/types";

type Bounds = [[number, number], [number, number]]; // [[south, west], [north, east]]

const BASEMAPS = {
  dark: {
    label: "Dark",
    // CARTO's dark tiles now demand an API key (they render a watermark without one).
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles © Esri — Esri, HERE, Garmin, OpenStreetMap contributors",
  },
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
  },
} as const;

function toBounds(image: SatelliteImage): Bounds | null {
  const b = image.metadata?.bounds_wgs84;
  if (!b) return null;
  const [west, south, east, north] = b;
  return [
    [south, west],
    [north, east],
  ];
}

function FitToImages({ bounds }: { bounds: Bounds[] }) {
  const map = useMap();
  useEffect(() => {
    if (bounds.length === 0) return;
    map.fitBounds(L.latLngBounds(bounds.flat() as [number, number][]), { padding: [60, 60], maxZoom: 14 });
  }, [map, bounds]);
  return null;
}

export default function MapView({ images }: { images: SatelliteImage[] }) {
  const [basemap, setBasemap] = useState<keyof typeof BASEMAPS>("dark");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [previews, setPreviews] = useState<Record<string, string>>({});

  const located = useMemo(
    () =>
      images.flatMap((image) => {
        const bounds = toBounds(image);
        return bounds ? [{ image, bounds }] : [];
      }),
    [images],
  );
  const allBounds = useMemo(() => located.map((l) => l.bounds), [located]);
  const selected = located.find((l) => l.image.id === selectedId) ?? null;

  // Thumbnails are behind bearer auth, so each one is fetched with the token and drawn
  // from a local object URL. Only images with a preview are requested.
  useEffect(() => {
    let cancelled = false;
    const urls: string[] = [];
    (async () => {
      for (const { image } of located) {
        if (!image.thumbnail_available) continue;
        try {
          const blob = await api.thumbnailBlob(image.id);
          if (cancelled) return;
          const url = URL.createObjectURL(blob);
          urls.push(url);
          setPreviews((prev) => ({ ...prev, [image.id]: url }));
        } catch {
          // A missing preview just leaves the footprint outline on the map.
        }
      }
    })();
    return () => {
      cancelled = true;
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [located]);

  const base = BASEMAPS[basemap];

  return (
    <div className="plate relative h-[calc(100vh-11rem)] min-h-[28rem]">
      <MapContainer
        center={[52.1, 5.3]}
        zoom={7}
        className="h-full w-full bg-ink"
        scrollWheelZoom
        attributionControl
      >
        <TileLayer key={basemap} url={base.url} attribution={base.attribution} />
        <FitToImages bounds={allBounds} />
        {located.map(({ image, bounds }) => {
          const active = image.id === selectedId;
          return (
            <div key={image.id}>
              {previews[image.id] && <ImageOverlay url={previews[image.id]} bounds={bounds} opacity={0.92} />}
              <Rectangle
                bounds={bounds}
                pathOptions={{
                  color: active ? "#f0b13e" : "#5fd4c4",
                  weight: active ? 2.5 : 1.5,
                  fillOpacity: 0,
                }}
                eventHandlers={{ click: () => setSelectedId(image.id) }}
              >
                <Tooltip sticky>{image.original_filename}</Tooltip>
              </Rectangle>
            </div>
          );
        })}
      </MapContainer>

      <div className="absolute right-3 top-3 z-[1000] flex overflow-hidden rounded-sm border border-line bg-ink/90 backdrop-blur">
        {(Object.keys(BASEMAPS) as (keyof typeof BASEMAPS)[]).map((key) => (
          <button
            key={key}
            onClick={() => setBasemap(key)}
            className={`px-3 py-1.5 text-xs transition ${
              basemap === key ? "bg-raised text-text" : "text-muted hover:text-text"
            }`}
          >
            {BASEMAPS[key].label}
          </button>
        ))}
      </div>

      {selected && (
        <div className="rise absolute bottom-4 left-4 z-[1000] w-72 rounded-sm border border-line-strong bg-ink/95 p-4 backdrop-blur">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium" title={selected.image.original_filename}>
                {selected.image.original_filename}
              </div>
              <div className="mt-1 font-mono text-[11px] text-muted tabular">
                {selected.image.metadata?.width_px}×{selected.image.metadata?.height_px} px
                {selected.image.metadata?.gsd_meters_approx
                  ? ` · ${selected.image.metadata.gsd_meters_approx.toFixed(1)} m/px`
                  : ""}
              </div>
            </div>
            <button
              onClick={() => setSelectedId(null)}
              aria-label="Close"
              className="text-muted hover:text-text"
            >
              ✕
            </button>
          </div>
          <Link
            href={`/images/${selected.image.id}`}
            className="mt-3 block rounded-sm bg-elementwise px-3 py-2 text-center text-sm font-semibold text-ink transition hover:brightness-110"
          >
            Ask questions about this scene
          </Link>
        </div>
      )}
    </div>
  );
}
