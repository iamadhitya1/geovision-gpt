import type { FusionType } from "./types";

// Full class strings (not template-built) so Tailwind's scanner sees every one of them.
export const MODEL_INFO: Record<
  FusionType,
  { label: string; short: string; blurb: string; text: string; bg: string; border: string; hex: string }
> = {
  elementwise: {
    label: "Elementwise fusion",
    short: "Baseline",
    blurb: "Published LiT-4-RSVQA approach: image and question features are multiplied together.",
    text: "text-elementwise",
    bg: "bg-elementwise",
    border: "border-elementwise",
    hex: "#5fd4c4",
  },
  cross_attention: {
    label: "Cross-attention fusion",
    short: "This project",
    blurb: "The question attends over image features, so it can focus on the regions it asks about.",
    text: "text-cross",
    bg: "bg-cross",
    border: "border-cross",
    hex: "#f0b13e",
  },
};

export const FUSION_ORDER: FusionType[] = ["elementwise", "cross_attention"];

// Question styles the model was actually trained on (RSVQA-LR). It answers from a fixed
// vocabulary, so these give the strongest demos.
export const SUGGESTED_QUESTIONS = [
  "Is there a water area?",
  "Is a road present?",
  "How many buildings are there?",
  "Is it a rural or an urban area?",
  "Is there a residential building?",
];

export function formatConfidence(value: number): string {
  const pct = value * 100;
  if (pct >= 10) return `${pct.toFixed(0)}%`;
  if (pct >= 1) return `${pct.toFixed(1)}%`;
  return `${pct.toFixed(2)}%`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
