// Mirrors backend/app/application/dto.py — keep the two in step.

export type Role = "researcher" | "admin";
export type FusionType = "elementwise" | "cross_attention";
export type ImageStatus = "processing" | "ready" | "failed";

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: Role;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface AuthResponse {
  user: User;
  tokens: TokenPair;
}

export interface ImageMetadata {
  width_px: number;
  height_px: number;
  band_count: number;
  dtype: string;
  crs: string | null;
  pixel_size_x: number | null;
  pixel_size_y: number | null;
  gsd_meters_approx: number | null;
  bounds_native: [number, number, number, number] | null;
  bounds_wgs84: [number, number, number, number] | null;
  cloud_cover_percent: number | null;
}

export interface SatelliteImage {
  id: string;
  owner_id: string;
  original_filename: string;
  image_format: "geotiff" | "png" | "jpeg";
  status: ImageStatus;
  file_size_bytes: number;
  metadata: ImageMetadata | null;
  error_message: string | null;
  thumbnail_available: boolean;
  created_at: string;
  updated_at: string;
}

export interface AnswerAlternative {
  answer: string;
  confidence: number;
}

export interface VqaAnswer {
  fusion: FusionType;
  answer: string;
  confidence: number;
  alternatives: AnswerAlternative[];
  /** A plain-English caveat for this answer, e.g. why a count is given as a range. */
  note?: string | null;
}

export interface VqaQuery {
  id: string;
  image_id: string;
  question: string;
  answers: VqaAnswer[];
  created_at: string;
  /** What in the question can be highlighted on the image: one thing, two for a comparison. */
  highlight_targets?: string[];
  /** A "no" to a presence/count question means nothing to point at; to a comparison it doesn't. */
  highlight_kind?: "presence" | "count" | "comparison";
}

export interface Page<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

/** One thing a question named, and how it was found. `method` says which technique (a colour
 * index, a zero-shot segmenter, or OpenStreetMap map data — never the trained VQA model, which
 * can't locate anything) and `caveat` says how far to trust it. */
export interface HighlightLayer {
  concept: string;
  label: string;
  color: string; // "#rrggbb", the colour it is outlined in on the overlay
  coverage: number; // 0-1, fraction of the image it covers
  method: string;
  caveat?: string | null;
}

/** Where the thing(s) a question names appear in the image: one overlay PNG with everything
 * else dimmed and each layer outlined in its own colour. */
export interface ConceptHighlight {
  layers: HighlightLayer[];
  overlay_png_base64: string; // transparent PNG, same shape as the image
  caveat?: string | null; // a note about the whole highlight, e.g. size/shape words ignored
}

export interface AttentionHeatmap {
  fusion: FusionType;
  answer: string;
  confidence: number;
  grid_size: number;
  weights: number[]; // grid_size * grid_size, row-major, sums to ~1.0
}

export interface VqaStats {
  total_questions: number;
  sample_size: number;
  images_queried: number;
  by_fusion: { fusion: FusionType; answered: number; mean_confidence: number }[];
  agreement: { compared: number; agreed: number };
  recent: VqaQuery[];
}
