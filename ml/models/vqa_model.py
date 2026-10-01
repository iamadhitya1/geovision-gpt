"""LiT-4-RSVQA-style architecture: BERT-Tiny + a lightweight pretrained image transformer
+ an elementwise-product fusion module + an MLP classifier.

Follows Hackel, Clasen, Ravanbakhsh & Demir, "LiT-4-RSVQA: Lightweight Transformer-based
Visual Question Answering in Remote Sensing" (arXiv:2306.00758, TU Berlin, 2023).

Confirmed available in this project's dependencies (verified against actual installed
package listings, not assumed):
  - text encoder:  "google/bert_uncased_L-2_H-128_A-2"        (HuggingFace transformers) —
                    the original Google-released BERT-Tiny checkpoint (2 layers, 128
                    hidden, 2 heads — matches the paper's spec exactly). The community
                    mirror "prajjwal1/bert-tiny" was tried first and rejected: its
                    config.json is missing a `model_type` key that current `transformers`
                    requires, so AutoConfig can't load it. This is the authoritative
                    source, not a fallback.
  - image encoder: "xcit_nano_12_p16_224.fb_dist_in1k"        (timm) — the paper's best config
                    "deit_tiny_distilled_patch16_224.fb_in1k" (timm) — alternative
                    "mobilevit_s.cvnets_in1k"                  (timm) — alternative
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import timm
import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

ImageEncoderName = Literal["xcit_nano", "deit_tiny", "mobilevit_s"]
FusionName = Literal["elementwise", "cross_attention"]

_TIMM_MODEL_NAMES: dict[ImageEncoderName, str] = {
    "xcit_nano": "xcit_nano_12_p16_224.fb_dist_in1k",
    "deit_tiny": "deit_tiny_distilled_patch16_224.fb_in1k",
    "mobilevit_s": "mobilevit_s.cvnets_in1k",
}

TEXT_ENCODER_NAME = "google/bert_uncased_L-2_H-128_A-2"


@dataclass(frozen=True, slots=True)
class VqaModelConfig:
    image_encoder: ImageEncoderName = "xcit_nano"  # best in the paper's own results
    freeze_image_encoder: bool = False  # fine-tuned, not frozen solid — see the paper's
    # own finding that fine-tuning beats both "frozen" and "trained from scratch"
    freeze_text_encoder: bool = False
    fusion_dim: int = 512
    num_answers: int = 1000
    classifier_hidden_dim: int = 512
    dropout: float = 0.25  # matches the paper's reported dropout


class ImageEncoder(nn.Module):
    """Wraps a timm lightweight vision transformer, pretrained on ImageNet.

    ``num_classes=0`` tells timm to drop its classification head and return the pooled
    feature vector directly — exactly the "image feature vector" Part 1 of the study notes
    describes coming out of a ViT's [CLS] token (or timm's equivalent global pooling for
    encoders like MobileViT that don't use a literal CLS token).
    """

    def __init__(self, name: ImageEncoderName, *, freeze: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(_TIMM_MODEL_NAMES[name], pretrained=True, num_classes=0)
        self.output_dim: int = self.backbone.num_features

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.backbone(pixel_values)  # (batch, output_dim)

    def forward_tokens(self, pixel_values: Tensor) -> Tensor:
        """Every image token before pooling — (batch, num_tokens, output_dim). Cross-attention
        needs these: it lets each question word look at individual image regions, which a
        single pooled vector can't offer."""
        tokens: Tensor = self.backbone.forward_features(pixel_values)
        if tokens.dim() == 4:  # conv-style encoders (MobileViT) return a (B, C, H, W) map
            tokens = tokens.flatten(2).transpose(1, 2)
        return tokens


class TextEncoder(nn.Module):
    """Wraps BERT-Tiny. Uses the [CLS] token's final hidden state as the sentence
    representation — the standard choice for a BERT-family encoder used for classification,
    matching how the [CLS] token is described in Part 1 of the study notes."""

    def __init__(self, *, freeze: bool = False) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(TEXT_ENCODER_NAME)
        self.output_dim: int = self.backbone.config.hidden_size

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # (batch, output_dim) — the [CLS] token

    def forward_tokens(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """One vector per question token — (batch, seq_len, output_dim)."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class ElementwiseProductFusion(nn.Module):
    """The paper's fusion module: project both modalities to a common dimension, then
    combine with an elementwise (Hadamard) product — see "Fusion mechanisms" in the study
    notes for how this compares to bilinear pooling or cross-attention fusion.

    This is deliberately the simplest fusion in the whole architecture and is the intended
    place to swap in something more sophisticated (e.g. CrossAttentionFusion below) as a
    genuine, measurable experiment — not a fixed, unchangeable part of the design.
    """

    def __init__(self, image_dim: int, text_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.output_dim = fusion_dim

    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        return self.image_proj(image_features) * self.text_proj(text_features)


class CrossAttentionFusion(nn.Module):
    """Fusion by cross-attention: every word of the question looks over all the image's
    patch tokens and pulls in the regions relevant to it (see Part 1 of the study notes).
    This is the project's own experiment, compared against ElementwiseProductFusion on
    identical data and encoders.

    Unlike the elementwise fusion it takes *sequences* — image patch tokens and question
    word tokens — not one pooled vector each. That is essential, and is the fix for the
    first version of this class: with each modality squashed to a single token, attention
    ran over one key, softmax over one item is always exactly 1, and the question had zero
    influence on the output (verified: changing the question changed the output by 0.0).
    That version trained to 36% accuracy, the level of a model that never sees the question.
    """

    def __init__(self, image_dim: int, text_dim: int, fusion_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(fusion_dim)
        self.output_dim = fusion_dim

    def forward(
        self,
        image_tokens: Tensor,
        text_tokens: Tensor,
        text_mask: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """``image_tokens`` (B, N, image_dim), ``text_tokens`` (B, L, text_dim),
        ``text_mask`` (B, L) with 1 for real words and 0 for padding.

        ``return_attention=True`` additionally returns the (B, L, N) attention weights —
        how much each question word attended to each image token — for the interpretability
        view (see ``docs/presenters-guide.html`` "attention heatmap"). Off by default so every
        existing call site (training, evaluation) is unaffected."""
        image_kv = self.image_proj(image_tokens)  # (B, N, F): what each image region offers
        text_q = self.text_proj(text_tokens)  # (B, L, F): what each word is looking for
        attended, attn_weights = self.cross_attn(
            query=text_q, key=image_kv, value=image_kv, need_weights=return_attention
        )
        # Residual: each word keeps its own meaning alongside what it found in the image.
        fused_tokens = self.norm(text_q + attended)  # (B, L, F)

        # Average over real words only, so padding never dilutes the question's vector.
        mask = text_mask.unsqueeze(-1).to(fused_tokens.dtype)
        pooled = (fused_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if return_attention:
            return pooled, attn_weights  # attn_weights: (B, L, N), averaged over heads
        return pooled


class Classifier(nn.Module):
    """MLP head over the fused representation, matching the paper's "classification
    module" — VQA framed as picking one of ``num_answers`` classes (see "Classify vs.
    generate" in the study notes)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_answers: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(self, fused: Tensor) -> Tensor:
        return self.net(fused)  # (batch, num_answers) — raw logits


class LitRSVQAModel(nn.Module):
    """The full pipeline: image + question in, answer-class logits out."""

    def __init__(
        self, config: VqaModelConfig | None = None, fusion: FusionName = "elementwise"
    ) -> None:
        super().__init__()
        self.config = config or VqaModelConfig()

        self.image_encoder = ImageEncoder(
            self.config.image_encoder, freeze=self.config.freeze_image_encoder
        )
        self.text_encoder = TextEncoder(freeze=self.config.freeze_text_encoder)
        fusion_cls = (
            CrossAttentionFusion if fusion == "cross_attention" else ElementwiseProductFusion
        )
        self.fusion: ElementwiseProductFusion | CrossAttentionFusion = fusion_cls(
            self.image_encoder.output_dim, self.text_encoder.output_dim, self.config.fusion_dim
        )
        self.classifier = Classifier(
            self.fusion.output_dim,
            self.config.classifier_hidden_dim,
            self.config.num_answers,
            self.config.dropout,
        )

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """``return_attention=True`` is only meaningful with cross-attention fusion loaded —
        it additionally returns the fusion's (B, L, N) attention weights alongside the
        logits, for the attention-heatmap interpretability view. Raises if the loaded
        fusion is elementwise, which has no per-patch attention to return; callers should
        check ``isinstance(model.fusion, CrossAttentionFusion)`` first."""
        if isinstance(self.fusion, CrossAttentionFusion):
            if return_attention:
                fused, attn_weights = self.fusion(
                    self.image_encoder.forward_tokens(pixel_values),
                    self.text_encoder.forward_tokens(input_ids, attention_mask),
                    attention_mask,
                    return_attention=True,
                )
                return self.classifier(fused), attn_weights
            fused = self.fusion(
                self.image_encoder.forward_tokens(pixel_values),
                self.text_encoder.forward_tokens(input_ids, attention_mask),
                attention_mask,
            )
        else:
            if return_attention:
                raise ValueError(
                    "return_attention=True needs cross-attention fusion — elementwise fusion "
                    "has no per-patch attention weights to return."
                )
            fused = self.fusion(
                self.image_encoder(pixel_values), self.text_encoder(input_ids, attention_mask)
            )
        return self.classifier(fused)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


def build_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)


def build_image_transform(image_encoder: ImageEncoderName):
    """Returns the exact preprocessing (resize, normalisation) the chosen pretrained
    checkpoint expects — using timm's own resolved config rather than hand-picking
    normalisation constants, since a mismatch here silently degrades a pretrained model's
    accuracy without raising any error."""
    model = timm.create_model(_TIMM_MODEL_NAMES[image_encoder], pretrained=False)
    data_config = timm.data.resolve_data_config({}, model=model)
    return timm.data.create_transform(**data_config, is_training=False)


if __name__ == "__main__":
    # A CPU smoke test — confirms the whole pipeline is wired correctly (shapes match, a
    # forward and backward pass both run without error) before any real training, and
    # before any GPU time is spent on it.
    torch.manual_seed(0)
    config = VqaModelConfig(num_answers=1000)
    model = LitRSVQAModel(config)
    model.train()

    tokenizer = build_tokenizer()
    batch = tokenizer(
        ["Is it a rural or an urban area", "How many buildings are in the image"],
        padding=True,
        return_tensors="pt",
    )
    dummy_images = torch.randn(2, 3, 224, 224)

    logits = model(dummy_images, batch["input_ids"], batch["attention_mask"])
    print("logits shape:", logits.shape)  # expect (2, 1000)
    assert logits.shape == (2, config.num_answers)

    loss = logits.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(
        "params with gradients:", len(grad_norms), "/ total params:", len(list(model.parameters()))
    )

    print("total parameters:", f"{model.num_parameters():,}")
    print("trainable parameters:", f"{model.num_parameters(trainable_only=True):,}")
    print("SMOKE TEST PASSED")
