import type { FusionType } from "./types";

/**
 * Held-out test-set accuracy on RSVQA-LR (9,838 questions), from `ml/README.md`.
 * These are the "widened" models — trained on RSVQA-LR + RSVQA-HR with augmentation, not the
 * original single-dataset models — and each figure is the MEAN OF THREE TRAINING RUNS (seeds 0,
 * 1 and 2: `gen_xcit`, `gen_xcit_s1`, `gen_xcit_s2` and the `gen_xcit_xattn…` equivalents),
 * produced by `ml/scripts/aggregate_seeds.py`. A single run turned out to be misleading: the
 * two fusions looked tied when only seed 0 had been trained. The app serves the seed-0 models.
 * Measured on RSVQA-LR's test split so this stays comparable to the original numbers; the
 * unseen-city (RSVQA-HR Philadelphia) and robustness results live only in `ml/README.md`.
 */
export const SEEDS = 3;

export interface FusionResult {
  overall: number;
  /** Standard deviation of `overall` across the training runs, in points. */
  overallSpread: number;
  average: number;
  byType: { rural_urban: number; presence: number; comp: number; count: number };
}

export const TEST_SET_SIZE = 9838;

export const RESULTS: Record<FusionType, FusionResult | null> = {
  // Mean of gen_xcit (seed 0), gen_xcit_s1, gen_xcit_s2; evaluated 2026-09-23, 10-05, 10-07.
  elementwise: {
    overall: 72.51,
    overallSpread: 0.69,
    average: 74.18,
    byType: { rural_urban: 88.67, presence: 89.68, comp: 90.3, count: 28.1 },
  },
  // Mean of gen_xcit_xattn (seed 0), gen_xcit_xattn_s1, gen_xcit_xattn_s2.
  // Its average (69.6) is pulled down and made noisy by rural/urban, which has 100 test
  // questions and swung between 54% and 89% from one run to the next.
  cross_attention: {
    overall: 71.15,
    overallSpread: 0.77,
    average: 69.59,
    byType: { rural_urban: 73.67, presence: 88.38, comp: 88.44, count: 27.88 },
  },
};

// What a predictor that NEVER sees the image scores: for each question, the most common
// training answer to that exact question text (ml/scripts/question_only_baseline.py).
// RSVQA-LR's questions are templated, so this is strong — and it is the fair floor: the gap
// between a model and this row is what the image actually contributes.
export const QUESTION_ONLY: FusionResult = {
  overall: 70.42,
  overallSpread: 0,
  average: 64.84,
  byType: { rural_urban: 56.0, presence: 88.3, comp: 87.1, count: 28.0 },
};

// Counting scored the way the RSVQA paper scores it: the answer is placed in one of five ranges
// (0 / 1-10 / 11-100 / 101-1,000 / >1,000) instead of having to match an exact integer, which at
// 10 m per pixel nobody can read reliably. The app answers count questions this way too. The models'
// figures use the probability-per-range decoding (ml/scripts/count_bucket_eval.py);
// "question only" is the no-image floor in the same ranges. n counts every test count question,
// including ones whose exact answer is outside the 1,000-answer vocabulary (a range can still hold them).
// Elementwise is the mean of 3 runs (68.1, 68.5, 67.6 — the seed-1 run predates the metric);
// cross-attention the mean of 3 (61.7, 56.7, 57.4, spread 2.7). A separate ensemble check
// (ml/scripts/ensemble_eval.py) scored seeds 0-2 on the in-vocabulary count questions:
// 68.0, 67.9, 68.4 — consistent.
export const COUNT_IN_RANGES = {
  n: 2947,
  elementwise: 68.1,
  cross_attention: 58.6,
  question_only: 57.9,
  elementwise_runs: 3,
  cross_attention_runs: 3,
};

/**
 * The Gujarat result: both models fine-tuned on 3,985 labelled Gujarat tiles, scored ONCE on the
 * six test districts the models never saw (497 tiles), from `ml/README.md`. Balanced accuracy is
 * the mean of the rate of correct "yes" answers and of correct "no" answers, so a model that always
 * says the same thing scores 50% however common that answer is. Building count is the share placed
 * in the right one of five ranges, against always naming the commonest range. The labels come from
 * map products made by models (WorldCover, JRC surface water, GHSL, Google Open Buildings), not from
 * a ground survey. The pass rule (80% balanced accuracy for water, forest and rural/urban; counts
 * above the baseline) was fixed before any Gujarat model was trained.
 */
export const GUJARAT = {
  testTiles: 497,
  districts: 6,
  passMark: 80,
  countBaseline: 39.4,
  /** The Gujarat elementwise model on the original RSVQA-LR test set (no measurable loss). */
  elementwiseOnLr: 72.67,
  rows: [
    {
      key: "water",
      label: "Water area",
      hint: "Is there a water area?",
      old: { elementwise: 49.6, cross_attention: 49.3 },
      now: { elementwise: 80.5, cross_attention: 74.9 },
    },
    {
      key: "forest",
      label: "Forest",
      hint: "Is there a forest?",
      old: { elementwise: 53.4, cross_attention: 52.6 },
      now: { elementwise: 73.8, cross_attention: 74.9 },
    },
    {
      key: "rural_urban",
      label: "Rural / urban",
      hint: "Is it a rural or an urban area?",
      old: { elementwise: 73.4, cross_attention: 56.5 },
      now: { elementwise: 82.7, cross_attention: 85.2 },
    },
    {
      key: "count",
      label: "Building count",
      hint: "How many buildings are there? (in ranges)",
      old: { elementwise: 51.3, cross_attention: 19.3 },
      now: { elementwise: 68.0, cross_attention: 69.4 },
    },
  ],
} as const;

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
