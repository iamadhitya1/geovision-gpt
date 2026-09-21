"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, hasTokens, saveTokens } from "./api";
import type { User } from "./types";

interface AuthState {
  user: User | null;
  /** True until the first "am I already signed in?" check finishes. */
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, fullName: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (hasTokens()) {
        try {
          const me = await api.me();
          if (!cancelled) setUser(me);
        } catch {
          // Expired beyond refresh, or the server is down — either way, signed out.
        }
      }
      if (!cancelled) setLoading(false);
    })();
    const onSignedOut = () => setUser(null);
    window.addEventListener("gvg:signed-out", onSignedOut);
    return () => {
      cancelled = true;
      window.removeEventListener("gvg:signed-out", onSignedOut);
    };
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    const res = await api.login(email, password);
    saveTokens(res.tokens);
    setUser(res.user);
  }, []);

  const signUp = useCallback(async (email: string, fullName: string, password: string) => {
    const res = await api.register(email, fullName, password);
    saveTokens(res.tokens);
    setUser(res.user);
  }, []);

  const signOut = useCallback(async () => {
    await api.logout();
    saveTokens(null);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, loading, signIn, signUp, signOut }),
    [user, loading, signIn, signUp, signOut],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
