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
  // Job 4332 (trained) / 4333 (evaluated), 2026-09-21: the corrected token-sequence version.
  // The first attempt (job 4324) scored 36.06% because the question never reached the
  // classifier — a design bug documented in ml/README.md, deliberately not shown here.
  cross_attention: {
    overall: 72.33,
    average: 72.29,
    byType: { rural_urban: 81.0, presence: 90.15, comp: 89.23, count: 28.77 },
  },
};

// What a predictor that NEVER sees the image scores: for each question, the most common
// training answer to that exact question text (ml/scripts/question_only_baseline.py).
// RSVQA-LR's questions are templated, so this is strong — and it is the fair floor: the gap
// between a model and this row is what the image actually contributes.
export const QUESTION_ONLY: FusionResult = {
  overall: 70.42,
  average: 64.84,
  byType: { rural_urban: 56.0, presence: 88.3, comp: 87.1, count: 28.0 },
};

// `n` is the number of test questions of that type. It matters when reading the gaps:
// rural/urban has only 100, so one model getting 11 more of them right moves that bar by 11
// points — and, because "average accuracy" weights every type equally, moves the average too.
export const QUESTION_TYPES: {
  key: keyof FusionResult["byType"];
  label: string;
  hint: string;
  n: number;
}[] = [
  { key: "rural_urban", label: "Rural / urban", hint: "Is this a rural or an urban area?", n: 100 },
  { key: "presence", label: "Presence", hint: "Is there a road / water area?", n: 2955 },
  { key: "comp", label: "Comparison", hint: "Are there more roads than buildings?", n: 4002 },
  { key: "count", label: "Counting", hint: "How many buildings are there?", n: 2781 },
];
