import type { FusionType } from "./types";

/**
 * Held-out test-set accuracy on RSVQA-LR (9,838 questions), from `ml/README.md`.
 * These are the "widened" (`gen_xcit` / `gen_xcit_xattn`) models the app actually serves —
 * trained on RSVQA-LR + RSVQA-HR with augmentation, not the original single-dataset models.
 * Measured on RSVQA-LR's test split so this stays comparable to the original numbers; the
 * unseen-city (RSVQA-HR Philadelphia) and robustness results live only in `ml/README.md`.
 */
export interface FusionResult {
  overall: number;
  average: number;
  byType: { rural_urban: number; presence: number; comp: number; count: number };
}

export const TEST_SET_SIZE = 9838;

export const RESULTS: Record<FusionType, FusionResult | null> = {
  // TAG=gen_xcit, job 4341 (trained) / job 4349 (evaluated), 2026-09-22/23.
  elementwise: {
    overall: 72.02,
    average: 73.65,
    byType: { rural_urban: 88.0, presence: 89.0, comp: 89.8, count: 27.9 },
  },
  // TAG=gen_xcit_xattn, job 4365 (trained) / job 4366 (evaluated), 2026-09-23/24.
  cross_attention: {
    overall: 71.96,
    average: 73.91,
    byType: { rural_urban: 89.0, presence: 89.3, comp: 89.1, count: 28.3 },
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
