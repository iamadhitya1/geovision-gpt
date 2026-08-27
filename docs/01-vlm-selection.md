# Vision-Language Model Selection Study

> **Status:** Decided (Phase 0). Revisit after Module 6 benchmarking.
> **Owner:** M. Adhitya

## 1. Why this decision matters

Every downstream module inherits the backbone's limits:

- **Module 4 (Feature Extraction)** inherits the vision encoder's resolution handling. Satellite
  objects (vehicles ~4 px, buildings ~15 px at 10 m GSD) vanish under aggressive downsampling.
- **Module 8 (Explainable AI)** inherits the model's *grounding* ability. If the backbone cannot
  emit coordinates, every bounding box must be faked by a separate detector, and the explanation no
  longer explains the answer that was actually given.
- **Module 3 (Preprocessing)** inherits the band constraint. Every general-purpose VLM sees 3-channel
  RGB. Sentinel-2 has 13 bands. That gap must be designed for, not ignored.

## 2. Candidate comparison

| Model | Params | Vision encoder strategy | Native grounding | Multi-band | License | Verdict |
|---|---|---|---|---|---|---|
| **Qwen2.5-VL** | 3B/7B/32B/72B | Native dynamic resolution (NaViT-style, variable patch count) + window attention | **Yes** — boxes *and* points in parseable JSON | RGB only | Apache-2.0 | **Selected (backbone)** |
| InternVL 2.5 / 3 | 1B–78B | Dynamic tiling into fixed 448 px tiles | Partial | RGB only | Apache-2.0 (weights) | Comparison arm |
| LLaVA-NeXT | 7B/13B/34B | AnyRes fixed grid over CLIP ViT | No | RGB only | Apache-2.0 / LLaMA | Baseline arm |
| Molmo (Molmo-7B-D) | 7B | CLIP ViT + PixMo data, **pointing** supervision | Points (not boxes) | RGB only | Apache-2.0 | Counting specialist arm |
| Phi-3-Vision | 4.2B | CLIP-based, small | No | RGB only | MIT | Edge / quantized arm |
| CLIP (& RemoteCLIP) | 0.4B | Contrastive dual encoder | N/A | RGB only | MIT | **Component**, not backbone |
| DOFA | ~0.1–0.3B | Wavelength-conditioned dynamic weight generation | N/A | **Yes — arbitrary bands** | Apache-2.0 | **Component (spectral branch)** |

### Why CLIP and DOFA are components, not candidates

CLIP is **not generative**. It scores image–text similarity; it cannot answer "explain your reasoning."
It earns a place as the **retrieval index** for the RAG path (Module 6) and for zero-shot scene tagging.

DOFA is a **remote-sensing foundation encoder**, not a VLM. Its value is exactly the gap the others
have: it conditions its patch-embedding weights on the physical **wavelength** of each input band, so
it ingests Sentinel-2's 13 bands, SAR, or hyperspectral without retraining. It becomes the spectral
branch feeding evidence into the backbone.

## 3. Decision — Qwen2.5-VL-7B-Instruct as the reasoning backbone

Ranked reasons:

1. **Native dynamic resolution.** Tiling approaches (InternVL, LLaVA-NeXT AnyRes) cut a large scene
   into fixed tiles, which severs objects at tile boundaries and destroys global context — a linear
   feature like a river or road is precisely what tiling breaks. Qwen2.5-VL processes the native
   aspect ratio with a variable number of visual tokens, which matches non-square GeoTIFF crops.
2. **Grounding is first-class.** It emits `<|box_start|>(x1,y1),(x2,y2)<|box_end|>` and point outputs
   that parse deterministically. Module 8's bounding boxes then come from the *same* forward pass that
   produced the answer, making the explanation faithful rather than post-hoc.
3. **Deployment realism.** Apache-2.0, first-class `transformers` and `vLLM` support, and AWQ/GPTQ
   4-bit quantization lands the 7B at roughly 6 GB VRAM — a free Colab T4 runs it. Given the dev
   machine has no GPU, this is the difference between a reproducible experiment and a stalled project.
4. **OCR strength.** Map legends, scale bars, and annotated imagery are read reliably.

**Trade-off accepted:** InternVL 3 edges it on some pure-reasoning benchmarks. We accept a small
reasoning delta to gain grounding fidelity, because grounding *is* Module 8.

## 4. Resulting hybrid architecture

```
RGB composite ──> Qwen2.5-VL-7B (reasoning + grounding + language) ──┐
                                                                     ├─> Fused answer
13-band stack ──> DOFA encoder ──> spectral evidence ────────────────┤
                                                                     │
NDVI/NDWI/NDBI (deterministic, rasterio) ────────────────────────────┘
                          │
RemoteCLIP index <────────┘  (RAG retrieval over prior analysed scenes)
```

## 5. Benchmark plan (feeds the paper's Table II)

All arms evaluated on EarthVQA + RSVQA-LR under identical prompts:
Qwen2.5-VL-7B, Qwen2.5-VL-3B, InternVL3-8B, LLaVA-NeXT-7B, Molmo-7B-D, Phi-3-Vision.
Metrics: accuracy (yes/no, counting, area), BLEU/ROUGE-L/CIDEr (free-form), latency, peak VRAM.

## 6. Inference backend strategy (consequence of no local GPU)

`InferenceBackend` is a Protocol with three implementations:

| Backend | Use |
|---|---|
| `OpenAICompatibleBackend` | Default for development — OpenRouter / Together / any vLLM server |
| `LocalTransformersBackend` | Used when `CUDA` is present; the deployment target |
| `NullBackend` | Deterministic stub for CI and unit tests |

No module above the infrastructure layer ever imports `torch` directly.
