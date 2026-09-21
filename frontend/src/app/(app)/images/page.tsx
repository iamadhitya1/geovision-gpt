"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { AuthedImage } from "@/components/AuthedImage";
import { api, ApiError } from "@/lib/api";
import { formatBytes } from "@/lib/models";
import type { SatelliteImage } from "@/lib/types";

const STATUS_STYLE: Record<SatelliteImage["status"], string> = {
  ready: "border-ok/40 text-ok",
  processing: "border-cross/40 text-cross",
  failed: "border-danger/40 text-danger",
};

function UploadZone({ onUploaded }: { onUploaded: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function upload(files: File[]) {
    if (files.length === 0) return;
    setError(null);
    setProgress({ done: 0, total: files.length });
    const failures: string[] = [];
    for (const [i, file] of files.entries()) {
      try {
        await api.uploadImage(file);
      } catch (err) {
        failures.push(`${file.name}: ${err instanceof ApiError ? err.message : "upload failed"}`);
      }
      setProgress({ done: i + 1, total: files.length });
      onUploaded();
    }
    setProgress(null);
    if (failures.length) setError(failures.join(" · "));
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(Array.from(e.dataTransfer.files));
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-sm border border-dashed px-6 py-8 text-center transition ${
          dragging ? "border-elementwise bg-elementwise/5" : "border-line-strong hover:border-elementwise/60"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          className="hidden"
          onChange={(e) => {
            void upload(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
        {progress ? (
          <p className="font-mono text-sm text-elementwise tabular">
            uploading {progress.done} / {progress.total}
          </p>
        ) : (
          <>
            <p className="text-sm">
              <span className="text-elementwise">Drop satellite images here</span>, or click to browse
            </p>
            <p className="text-xs text-muted">GeoTIFF, PNG or JPEG · georeferenced files also appear on the map</p>
          </>
        )}
      </div>
      {error && (
        <p role="alert" className="mt-2 rounded-sm border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
          {error}
        </p>
      )}
    </div>
  );
}

function ImageCard({ image, onDelete }: { image: SatelliteImage; onDelete: () => void }) {
  const m = image.metadata;
  return (
    <div className="plate rise group">
      <Link href={`/images/${image.id}`} className="block">
        {image.thumbnail_available ? (
          <AuthedImage imageId={image.id} alt={image.original_filename} className="aspect-square" />
        ) : (
          <div className="flex aspect-square items-center justify-center bg-raised font-mono text-xs text-faint">
            no preview
          </div>
        )}
        <div className="space-y-2 p-3">
          <div className="flex items-start justify-between gap-2">
            <span className="truncate text-sm font-medium" title={image.original_filename}>
              {image.original_filename}
            </span>
            <span
              className={`shrink-0 rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${STATUS_STYLE[image.status]}`}
            >
              {image.status}
            </span>
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-muted tabular">
            {m && (
              <>
                <span>
                  {m.width_px}×{m.height_px}
                </span>
                {m.gsd_meters_approx && <span>{m.gsd_meters_approx.toFixed(1)} m/px</span>}
                {m.crs && <span>{m.crs}</span>}
              </>
            )}
            <span>{formatBytes(image.file_size_bytes)}</span>
          </div>
        </div>
      </Link>
      <button
        onClick={onDelete}
        aria-label={`Delete ${image.original_filename}`}
        className="absolute right-2 top-2 rounded-sm bg-ink/80 px-2 py-1 font-mono text-[11px] text-muted opacity-0 backdrop-blur transition hover:text-danger group-hover:opacity-100 focus-visible:opacity-100"
      >
        delete
      </button>
    </div>
  );
}

export default function LibraryPage() {
  const [images, setImages] = useState<SatelliteImage[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const page = await api.listImages();
      setImages(page.items);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't load your images.");
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial data fetch
    void load();
  }, [load]);

  async function remove(image: SatelliteImage) {
    if (!window.confirm(`Delete ${image.original_filename}? Its question history goes with it.`)) return;
    try {
      await api.deleteImage(image.id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't delete that image.");
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-bold tracking-tight">Image library</h1>
          <p className="mt-1 text-sm text-muted">
            Pick a scene to ask questions about, or upload a new one.
          </p>
        </div>
        {images && (
          <div className="font-mono text-sm text-muted tabular">
            {images.length} {images.length === 1 ? "image" : "images"}
          </div>
        )}
      </div>

      <UploadZone onUploaded={() => void load()} />

      {error && (
        <p role="alert" className="rounded-sm border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
          {error}
        </p>
      )}

      {images === null && !error && <p className="font-mono text-sm text-faint">loading…</p>}

      {images?.length === 0 && (
        <p className="py-12 text-center text-sm text-muted">
          Nothing here yet. Upload a satellite image above to get started.
        </p>
      )}

      {images && images.length > 0 && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
          {images.map((image) => (
            <ImageCard key={image.id} image={image} onDelete={() => void remove(image)} />
          ))}
        </div>
      )}
    </div>
  );
}
