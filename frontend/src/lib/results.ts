import type { FusionType } from "./types";

/**
 * Held-out test-set accuracy on RSVQA-LR (9,838 questions), from `ml/README.md`.
 * These are measured numbers, not placeholders: update `cross_attention` from the
 * evaluate_cross_attention job's output — it stays `null` (rendered as "pending") until
 * that run has actually produced a result.
 */
export interface FusionResult {
  overall: number;
  average: number;
  byType: { rural_urban: number; presence: number; comp: number; count: number };
}

export const TEST_SET_SIZE = 9838;

export const RESULTS: Record<FusionType, FusionResult | null> = {
  elementwise: {
    overall: 72.74,
    average: 75.16,
    byType: { rural_urban: 92.0, presence: 90.7, comp: 90.4, count: 27.5 },
  },
  cross_attention: null,
};

export const QUESTION_TYPES: { key: keyof FusionResult["byType"]; label: string; hint: string }[] = [
  { key: "rural_urban", label: "Rural / urban", hint: "Is this a rural or an urban area?" },
  { key: "presence", label: "Presence", hint: "Is there a road / water area?" },
  { key: "comp", label: "Comparison", hint: "Are there more roads than buildings?" },
  { key: "count", label: "Counting", hint: "How many buildings are there?" },
];
