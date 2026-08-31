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
    """An alternative fusion module: the question attends directly to the image features
    (cross-attention, see Part 1 of the study notes) before combining, instead of squashing
    each modality into one vector first. This is the natural candidate for the project's
    own experimental contribution — swap this in for ElementwiseProductFusion and compare
    accuracy on identical data/encoders to measure whether richer fusion actually helps.
    """

    def __init__(self, image_dim: int, text_dim: int, fusion_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True)
        self.output_dim = fusion_dim

    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        # Treated as single-token sequences per modality: cross-attention here reduces to a
        # learned, query-conditioned reweighting of the image vector by the question vector.
        # A richer version would attend over per-patch image tokens rather than one pooled
        # vector — a natural next step, not implemented here to keep the baseline honest
        # about what it actually does.
        image_token = self.image_proj(image_features).unsqueeze(1)  # (batch, 1, fusion_dim)
        text_token = self.text_proj(text_features).unsqueeze(1)  # (batch, 1, fusion_dim)
        attended, _ = self.cross_attn(query=text_token, key=image_token, value=image_token)
        return attended.squeeze(1)


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

    def __init__(self, config: VqaModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or VqaModelConfig()

        self.image_encoder = ImageEncoder(
            self.config.image_encoder, freeze=self.config.freeze_image_encoder
        )
        self.text_encoder = TextEncoder(freeze=self.config.freeze_text_encoder)
        self.fusion = ElementwiseProductFusion(
            self.image_encoder.output_dim, self.text_encoder.output_dim, self.config.fusion_dim
        )
        self.classifier = Classifier(
            self.fusion.output_dim,
            self.config.classifier_hidden_dim,
            self.config.num_answers,
            self.config.dropout,
        )

    def forward(self, pixel_values: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        image_features = self.image_encoder(pixel_values)
        text_features = self.text_encoder(input_ids, attention_mask)
        fused = self.fusion(image_features, text_features)
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
