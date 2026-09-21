"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const { user, loading, signIn, signUp } = useAuth();
  const router = useRouter();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!loading && user) router.replace("/images");
  }, [loading, user, router]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "signin") await signIn(email, password);
      else await signUp(email, fullName, password);
      router.replace("/images");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const input =
    "w-full rounded-sm border border-line bg-ink px-3 py-2.5 text-sm text-text placeholder:text-faint focus:border-elementwise focus:outline-none";

  return (
    <main className="grid min-h-screen lg:grid-cols-[1.15fr_1fr]">
      <section className="relative hidden flex-col justify-between overflow-hidden border-r border-line p-12 lg:flex">
        <div className="font-display text-lg font-bold tracking-tight">
          GeoVision<span className="text-elementwise">GPT</span>
        </div>

        <div className="max-w-xl">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-elementwise">
            Visual question answering · Earth observation
          </p>
          <h1 className="mt-5 font-display text-5xl font-bold leading-[1.05] tracking-tight text-balance">
            Ask a satellite image a question.
          </h1>
          <p className="mt-5 max-w-md text-base leading-relaxed text-muted">
            Upload a scene, ask in plain English, and watch two lightweight models — one
            published baseline, one new architecture — answer side by side.
          </p>
        </div>

        <div className="flex gap-8 font-mono text-xs text-muted">
          <div>
            <div className="text-2xl font-medium text-text tabular">8.2M</div>
            parameters (baseline)
          </div>
          <div>
            <div className="text-2xl font-medium text-text tabular">1,000</div>
            answer classes
          </div>
          <div>
            <div className="text-2xl font-medium text-text tabular">57k</div>
            training questions
          </div>
        </div>

        {/* A locator crosshair, drawn rather than imaged: no stock photo to license or load. */}
        <svg
          aria-hidden
          viewBox="0 0 400 400"
          className="pointer-events-none absolute -right-24 top-1/2 h-[26rem] w-[26rem] -translate-y-1/2 text-elementwise opacity-[0.16]"
          fill="none"
          stroke="currentColor"
        >
          <circle cx="200" cy="200" r="190" />
          <circle cx="200" cy="200" r="130" strokeDasharray="3 7" />
          <circle cx="200" cy="200" r="60" />
          <path d="M200 0v400M0 200h400" />
          <path d="M60 60l280 280M340 60L60 340" strokeDasharray="2 10" />
        </svg>
      </section>

      <section className="flex items-center justify-center px-6 py-12">
        <form onSubmit={onSubmit} className="rise w-full max-w-sm space-y-5">
          <div>
            <h2 className="font-display text-2xl font-bold tracking-tight">
              {mode === "signin" ? "Sign in" : "Create an account"}
            </h2>
            <p className="mt-1 text-sm text-muted">
              {mode === "signin"
                ? "Pick up where you left off."
                : "Researcher accounts can upload images and ask questions."}
            </p>
          </div>

          {mode === "signup" && (
            <label className="block space-y-1.5">
              <span className="text-xs font-medium text-muted">Full name</span>
              <input
                className={input}
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                autoComplete="name"
                required
              />
            </label>
          )}
          <label className="block space-y-1.5">
            <span className="text-xs font-medium text-muted">Email</span>
            <input
              className={input}
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              required
            />
          </label>
          <label className="block space-y-1.5">
            <span className="text-xs font-medium text-muted">Password</span>
            <input
              className={input}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === "signin" ? "current-password" : "new-password"}
              minLength={mode === "signup" ? 8 : undefined}
              required
            />
            {mode === "signup" && (
              <span className="block text-xs text-faint">
                At least 8 characters, not containing your email name.
              </span>
            )}
          </label>

          {error && (
            <p role="alert" className="rounded-sm border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-sm bg-elementwise px-4 py-2.5 text-sm font-semibold text-ink transition hover:brightness-110 disabled:opacity-60"
          >
            {busy ? "One moment…" : mode === "signin" ? "Sign in" : "Create account"}
          </button>

          <button
            type="button"
            onClick={() => {
              setMode(mode === "signin" ? "signup" : "signin");
              setError(null);
            }}
            className="w-full text-center text-sm text-muted hover:text-text"
          >
            {mode === "signin" ? "New here? Create an account" : "Already registered? Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}
