import type {
  AuthResponse,
  FusionType,
  Page,
  SatelliteImage,
  TokenPair,
  User,
  VqaQuery,
  VqaStats,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";
const TOKEN_KEY = "gvg.tokens";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

// --- Token storage ----------------------------------------------------------------
// localStorage is enough for a research demo; a production deployment would move the
// refresh token to an httpOnly cookie (which needs a backend change, not just this file).

function readTokens(): TokenPair | null {
  try {
    const raw = localStorage.getItem(TOKEN_KEY);
    return raw ? (JSON.parse(raw) as TokenPair) : null;
  } catch {
    return null;
  }
}

export function saveTokens(tokens: TokenPair | null): void {
  try {
    if (tokens) localStorage.setItem(TOKEN_KEY, JSON.stringify(tokens));
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage blocked (private window) — the session simply won't survive a reload.
  }
}

export function hasTokens(): boolean {
  return readTokens() !== null;
}

// --- Core request -----------------------------------------------------------------

// Refresh tokens are single-use on the server, so two requests failing with 401 at the
// same moment must share ONE refresh call — a second one would replay the burnt token,
// which the backend treats as theft and answers by revoking every session.
let refreshInFlight: Promise<boolean> | null = null;

async function refreshTokens(): Promise<boolean> {
  const current = readTokens();
  if (!current) return false;
  refreshInFlight ??= (async () => {
    try {
      const res = await fetch(`${BASE}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: current.refresh_token }),
      });
      if (!res.ok) return false;
      saveTokens((await res.json()) as TokenPair);
      return true;
    } catch {
      return false;
    } finally {
      refreshInFlight = null;
    }
  })();
  return refreshInFlight;
}

async function send(path: string, init: RequestInit, auth: boolean): Promise<Response> {
  const build = (): RequestInit => {
    const headers = new Headers(init.headers);
    const tokens = readTokens();
    if (auth && tokens) headers.set("Authorization", `Bearer ${tokens.access_token}`);
    return { ...init, headers };
  };

  let res = await fetch(`${BASE}${path}`, build());
  if (res.status === 401 && auth && (await refreshTokens())) {
    res = await fetch(`${BASE}${path}`, build());
  }
  if (res.status === 401 && auth) {
    saveTokens(null);
    window.dispatchEvent(new Event("gvg:signed-out"));
  }
  return res;
}

async function toError(res: Response): Promise<ApiError> {
  try {
    const body = (await res.json()) as { code?: string; message?: string };
    return new ApiError(res.status, body.code ?? "error", body.message ?? res.statusText);
  } catch {
    return new ApiError(res.status, "error", res.statusText || "Request failed");
  }
}

async function request<T>(path: string, init: RequestInit = {}, auth = true): Promise<T> {
  let res: Response;
  try {
    res = await send(path, init, auth);
  } catch {
    throw new ApiError(0, "network", "Can't reach the GeoVisionGPT server. Is the backend running?");
  }
  if (!res.ok) throw await toError(res);
  return (await res.json()) as T;
}

function json(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

// --- Endpoints --------------------------------------------------------------------

export const api = {
  login: (email: string, password: string) =>
    request<AuthResponse>("/auth/login", json("POST", { email, password }), false),

  register: (email: string, full_name: string, password: string) =>
    request<AuthResponse>("/auth/register", json("POST", { email, full_name, password }), false),

  logout: async (): Promise<void> => {
    const tokens = readTokens();
    if (tokens) {
      // Best effort: the local session is dropped regardless of whether this lands.
      await request("/auth/logout", json("POST", { refresh_token: tokens.refresh_token })).catch(
        () => undefined,
      );
    }
  },

  me: () => request<User>("/auth/me"),

  listImages: (limit = 100) => request<Page<SatelliteImage>>(`/images?limit=${limit}`),

  getImage: (id: string) => request<SatelliteImage>(`/images/${id}`),

  uploadImage: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<SatelliteImage>("/images", { method: "POST", body: form });
  },

  deleteImage: (id: string) => request<{ message: string }>(`/images/${id}`, { method: "DELETE" }),

  stats: () => request<VqaStats>("/vqa/stats"),

  fusions: () => request<FusionType[]>("/vqa/fusions"),

  ask: (imageId: string, question: string) =>
    request<VqaQuery>(`/images/${imageId}/ask`, json("POST", { question })),

  history: (imageId: string) => request<Page<VqaQuery>>(`/images/${imageId}/vqa?limit=50`),

  /** The thumbnail endpoint needs the bearer header, which an <img src> can't send. */
  thumbnailBlob: async (id: string): Promise<Blob> => {
    const res = await send(`/images/${id}/thumbnail`, {}, true);
    if (!res.ok) throw await toError(res);
    return res.blob();
  },
};
