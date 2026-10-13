"use client";

import "leaflet/dist/leaflet.css";
import L from "leaflet";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  ImageOverlay,
  MapContainer,
  Rectangle,
  TileLayer,
  Tooltip,
  ZoomControl,
  useMap,
  useMapEvents,
} from "react-leaflet";
import { api, ApiError } from "@/lib/api";
import type { SatelliteImage } from "@/lib/types";

type Bounds = [[number, number], [number, number]]; // [[south, west], [north, east]]

// The scene "Analyse this place" captures: 256 px at ~10 m per pixel, so 2.56 km across.
// This is the size the models were trained on — see the capture endpoint in the backend.
const SCENE_SIDE_M = 2560;
// Below this zoom the square is only a few pixels wide, so there's no real "place" to pick.
const MIN_ANALYSE_ZOOM = 9;

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
  sentinel: {
    label: "Sentinel-2",
    // The same imagery a captured scene is cut from: ~10 m per pixel, like the training data.
    url: "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
    attribution:
      '<a href="https://s2maps.eu" target="_blank" rel="noopener">Sentinel-2 cloudless</a> by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2020)',
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

/** The square a capture would cover if taken now, as map bounds around the centre. */
function sceneBounds(lat: number, lng: number): Bounds {
  const half = SCENE_SIDE_M / 2;
  const dLat = half / 111_320;
  const dLng = half / (111_320 * Math.cos((lat * Math.PI) / 180));
  return [
    [lat - dLat, lng - dLng],
    [lat + dLat, lng + dLng],
  ];
}

/** Reports where the map is looking, so the capture square and button follow the view. */
function ViewTracker({ onChange }: { onChange: (view: { lat: number; lng: number; zoom: number }) => void }) {
  const map = useMapEvents({
    moveend: () => {
      const c = map.getCenter();
      onChange({ lat: c.lat, lng: c.lng, zoom: map.getZoom() });
    },
  });
  useEffect(() => {
    const c = map.getCenter();
    onChange({ lat: c.lat, lng: c.lng, zoom: map.getZoom() });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- report the starting view once
  }, []);
  return null;
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
  const router = useRouter();
  const [map, setMap] = useState<L.Map | null>(null);
  const [view, setView] = useState<{ lat: number; lng: number; zoom: number } | null>(null);
  const [place, setPlace] = useState("");
  const [searching, setSearching] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

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
  const canAnalyse = view !== null && view.zoom >= MIN_ANALYSE_ZOOM && !capturing;

  // OpenStreetMap's Nominatim geocoder: one request per submitted search (never per keystroke),
  // which is what its usage policy asks for.
  async function search(event: React.FormEvent) {
    event.preventDefault();
    const query = place.trim();
    if (!query || !map || searching) return;
    setSearching(true);
    setNotice(null);
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(query)}`,
        { headers: { Accept: "application/json" } },
      );
      const hits: { lat: string; lon: string; display_name: string }[] = await res.json();
      if (hits.length === 0) {
        setNotice(`Couldn't find "${query}". Try a city or a more specific name.`);
      } else {
        map.flyTo([Number(hits[0].lat), Number(hits[0].lon)], 13, { duration: 1.2 });
      }
    } catch {
      setNotice("The place search isn't reachable right now. Pan and zoom the map instead.");
    } finally {
      setSearching(false);
    }
  }

  async function analyse() {
    if (!view || !canAnalyse) return;
    setCapturing(true);
    setNotice(null);
    try {
      const image = await api.captureScene(view.lat, view.lng);
      router.push(`/images/${image.id}`);
    } catch (err) {
      setNotice(err instanceof ApiError ? err.message : "Couldn't fetch that scene. Try again.");
      setCapturing(false);
    }
  }

  return (
    <div className="plate relative h-[calc(100vh-11rem)] min-h-[28rem]">
      <MapContainer
        center={[52.1, 5.3]}
        zoom={7}
        className="h-full w-full bg-ink"
        scrollWheelZoom
        zoomControl={false}
        attributionControl
        ref={setMap}
      >
        <TileLayer
          key={basemap}
          url={base.url}
          attribution={base.attribution}
          // The Sentinel-2 mosaic stops at zoom 14; past that Leaflet stretches its last level.
          maxNativeZoom={basemap === "sentinel" ? 14 : undefined}
          maxZoom={18}
        />
        {/* Bottom-left: the default top-left spot is covered by the place-search box. */}
        <ZoomControl position="bottomleft" />
        <ViewTracker onChange={setView} />
        <FitToImages bounds={allBounds} />
        {view && view.zoom >= MIN_ANALYSE_ZOOM && (
          <Rectangle
            bounds={sceneBounds(view.lat, view.lng)}
            pathOptions={{ color: "#ffffff", weight: 1.5, dashArray: "6 6", fillOpacity: 0 }}
            interactive={false}
          />
        )}
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

      <form
        onSubmit={search}
        className="absolute left-3 top-3 z-[1000] flex w-72 max-w-[calc(100%-1.5rem)] overflow-hidden rounded-sm border border-line bg-ink/90 backdrop-blur"
      >
        <input
          value={place}
          onChange={(e) => setPlace(e.target.value)}
          placeholder="Find a place, e.g. Ahmedabad"
          aria-label="Find a place"
          className="min-w-0 flex-1 bg-transparent px-3 py-1.5 text-xs placeholder:text-faint focus:outline-none"
        />
        <button
          type="submit"
          disabled={searching || !place.trim()}
          className="px-3 py-1.5 text-xs text-muted transition hover:text-text disabled:opacity-40"
        >
          {searching ? "…" : "Go"}
        </button>
      </form>

      <div className="absolute bottom-4 right-4 z-[1000] flex max-w-xs flex-col items-end gap-2">
        {notice && (
          <p role="alert" className="rounded-sm border border-danger/40 bg-ink/95 px-3 py-2 text-xs text-danger">
            {notice}
          </p>
        )}
        <button
          type="button"
          onClick={() => void analyse()}
          disabled={!canAnalyse}
          className="rounded-sm bg-elementwise px-4 py-2.5 text-sm font-semibold text-ink shadow-lg transition hover:brightness-110 disabled:opacity-50"
        >
          {capturing
            ? "Fetching the scene…"
            : view && view.zoom < MIN_ANALYSE_ZOOM
              ? "Zoom in to pick a place"
              : "Analyse this place"}
        </button>
        <p className="rounded-sm bg-ink/80 px-2 py-1 text-right font-mono text-[10px] leading-snug text-muted backdrop-blur">
          Analyses the dashed 2.5 km square, from Sentinel-2 imagery at ~10 m per pixel — the scale
          the models were trained on.
        </p>
      </div>

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
