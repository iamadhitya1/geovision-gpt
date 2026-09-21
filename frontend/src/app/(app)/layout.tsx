"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { FusionType } from "@/lib/types";

const NAV = [{ href: "/images", label: "Library" }];

function ModelStatus() {
  const [fusions, setFusions] = useState<FusionType[] | null>(null);
  useEffect(() => {
    api.fusions().then(setFusions).catch(() => setFusions([]));
  }, []);

  if (fusions === null) return null;
  const online = fusions.length;
  const tone = online === 0 ? "text-danger" : online === 1 ? "text-cross" : "text-ok";
  return (
    <div
      className="flex items-center gap-2 rounded-full border border-line px-3 py-1 font-mono text-xs"
      title={online === 0 ? "No trained checkpoints are loaded on the server." : fusions.join(", ")}
    >
      <span className={`h-1.5 w-1.5 rounded-full bg-current ${tone}`} />
      <span className="text-muted">
        {online === 0 ? "no models loaded" : `${online} of 2 models online`}
      </span>
    </div>
  );
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { user, loading, signOut } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [loading, user, router]);

  if (loading || !user) {
    return (
      <div className="flex flex-1 items-center justify-center font-mono text-sm text-faint">
        loading…
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-ink/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-8 px-5">
          <Link href="/images" className="font-display text-base font-bold tracking-tight">
            GeoVision<span className="text-elementwise">GPT</span>
          </Link>
          <nav className="flex gap-1">
            {NAV.map((item) => {
              const active = pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`rounded-sm px-3 py-1.5 text-sm transition ${
                    active ? "bg-raised text-text" : "text-muted hover:text-text"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto flex items-center gap-4">
            <ModelStatus />
            <div className="hidden text-right leading-tight sm:block">
              <div className="text-sm">{user.full_name}</div>
              <div className="font-mono text-[11px] uppercase tracking-wider text-faint">{user.role}</div>
            </div>
            <button
              onClick={async () => {
                await signOut();
                router.replace("/login");
              }}
              className="rounded-sm border border-line px-3 py-1.5 text-sm text-muted transition hover:border-line-strong hover:text-text"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-5 py-8">{children}</main>
    </div>
  );
}
