# Research Contribution — SGV: Spectral-Grounded Verification for Remote-Sensing VQA

> **Status:** Proposed (Phase 0). Implemented across Modules 3, 6, 7, 8.

## 1. The problem, stated precisely

General-purpose VLMs are fluent but **quantitatively unreliable** on overhead imagery. Ask
Qwen2.5-VL "what percentage of this image is vegetation?" and it will confidently answer "roughly
40%" — a number produced by language priors, not by measurement. The failure is systematic and it is
the single biggest obstacle to using VLMs in an operational Earth-observation setting, where a wrong
flood-extent number is worse than no number.

Existing remote-sensing VQA work responds by **fine-tuning** (RSVQA, EarthVQA baselines, GeoChat).
That is expensive, locks the system to one dataset's answer vocabulary, and has to be redone for every
new backbone.

## 2. The proposed contribution

**SGV is a training-free, model-agnostic verification framework that converts physics-based spectral
indices into a verifier for VLM-generated claims.**

Three components:

### 2.1 Spectral Evidence Injection (SEI)

Before generation, deterministically compute from the raster itself:

- NDVI, NDWI, MNDWI, NDBI per-pixel, then thresholded coverage fractions
- Ground sample distance and **true area per pixel** from the affine geotransform
- Cloud fraction, valid-data mask
- Coarse land-cover histogram from a lightweight segmenter

These are serialised into a structured `EVIDENCE` block appended to the prompt, **with real units**
(m², %, not pixels). The VLM is instructed to reason *from* the evidence and to state when it
disagrees with it.

### 2.2 Claim Extraction and Verification Loop (CEV)

The generated answer is parsed for **quantitative claims** — percentages, counts, areas, and
presence/absence assertions. Each claim is checked against the deterministic evidence:

- `|claim − measurement| > τ` → contradiction
- presence claim not supported by any index threshold → unsupported

On contradiction, the system re-generates **once**, with the specific contradiction surfaced in the
prompt. Persisting contradictions are reported to the user as a flagged low-confidence answer rather
than silently returned. This is the concrete mechanism behind the brief's "Hallucination Detection".

### 2.3 Calibrated Confidence

Confidence is a *learned combination*, not a number the model invents:

```
conf = σ( w1·mean_token_logprob + w2·evidence_agreement + w3·cross_view_consistency + b )
```

where `cross_view_consistency` is answer agreement across *k* augmented views (rotation, flip,
mild scale). Weights are fit on a held-out split by minimising negative log-likelihood, so
**Expected Calibration Error (ECE) becomes a reportable metric** — this is the part that makes the
"Confidence Score" feature in the brief scientifically defensible instead of cosmetic.

## 3. Why this is publishable

- **Novel framing.** Spectral indices are normally *outputs* of remote-sensing pipelines. SGV uses
  them as a **verifier** over a language model's claims. That inversion is the idea.
- **Training-free.** No fine-tuning, so it transfers across backbones — which is directly testable.
- **Falsifiable.** It makes sharp predictions that can fail.

## 4. Experimental design

**H1** — SEI improves accuracy on quantitative questions (area/percentage/count) on EarthVQA and
RSVQA-LR relative to an identical prompt without evidence.
**H2** — CEV reduces quantitative hallucination rate (fraction of answers whose numeric claim
deviates > τ from ground truth) without harming free-form quality (BLEU/ROUGE/CIDEr).
**H3** — Calibrated confidence achieves lower ECE than raw sequence likelihood.
**H4** — Gains hold across Qwen2.5-VL, InternVL3, and LLaVA-NeXT (generality).

**Ablation table (paper Table III):** baseline → +SEI → +CEV → +calibration → full SGV.

**Honest risk:** H1 may hold strongly while H2 shows only a modest effect, since re-generation can
fix the number but keep the flawed rationale. That is itself a reportable finding, and the ablation
is designed to separate the two.

## 5. Target venues

IEEE JSTARS, IEEE GRSL (letter format suits a training-free method), or an IGARSS 2027 submission.
Fallback: arXiv preprint plus the university report.

## 6. Mapping to the module plan

| SGV component | Implemented in |
|---|---|
| SEI — index computation | Module 3 (Preprocessing) |
| SEI — evidence serialisation into prompt | Module 6 (Vision-Language Fusion) |
| CEV — claim parsing and re-generation | Module 7 (Answer Generation) |
| Calibration | Module 7 + Module 10 (Analytics) |
| Faithful grounding overlay | Module 8 (Explainable AI) |
