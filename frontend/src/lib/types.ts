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
}

export interface VqaQuery {
  id: string;
  image_id: string;
  question: string;
  answers: VqaAnswer[];
  created_at: string;
}

export interface Page<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}
