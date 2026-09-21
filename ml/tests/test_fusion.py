"""Fusion modules must actually use BOTH inputs.

The first CrossAttentionFusion collapsed each modality to a single token, so attention ran
over one key, softmax over one item is always 1, and the question had exactly zero effect
on the output. It trained to ~36% accuracy — a model that never sees the question — and
nothing in the training loop flagged it. These tests check the property directly, on random
tensors (no pretrained weights or dataset needed).
"""

from __future__ import annotations

import torch

from models.vqa_model import CrossAttentionFusion, ElementwiseProductFusion

IMAGE_DIM, TEXT_DIM, FUSION_DIM = 24, 16, 32
NUM_IMAGE_TOKENS, SEQ_LEN = 9, 6
TOLERANCE = 1e-4


def _cross() -> CrossAttentionFusion:
    torch.manual_seed(0)
    return CrossAttentionFusion(IMAGE_DIM, TEXT_DIM, FUSION_DIM, num_heads=4).eval()


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(2, NUM_IMAGE_TOKENS, IMAGE_DIM, generator=generator)
    text = torch.randn(2, SEQ_LEN, TEXT_DIM, generator=generator)
    mask = torch.ones(2, SEQ_LEN)
    return image, text, mask


def test_cross_attention_output_depends_on_the_question() -> None:
    fusion = _cross()
    image, question_a, mask = _inputs(1)
    _, question_b, _ = _inputs(2)

    with torch.no_grad():
        out_a = fusion(image, question_a, mask)
        out_b = fusion(image, question_b, mask)

    assert (out_a - out_b).abs().max() > TOLERANCE


def test_cross_attention_output_depends_on_the_image() -> None:
    fusion = _cross()
    image_a, question, mask = _inputs(1)
    image_b, _, _ = _inputs(3)

    with torch.no_grad():
        out_a = fusion(image_a, question, mask)
        out_b = fusion(image_b, question, mask)

    assert (out_a - out_b).abs().max() > TOLERANCE


def test_cross_attention_ignores_padding_positions() -> None:
    fusion = _cross()
    image, question, _ = _inputs(1)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.float)
    garbled = question.clone()
    garbled[0, 3:] = 99.0  # whatever sits in padding must not leak into the answer
    garbled[1, 5:] = -99.0

    with torch.no_grad():
        clean = fusion(image, question, mask)
        noisy = fusion(image, garbled, mask)

    assert torch.allclose(clean, noisy, atol=TOLERANCE)


def test_cross_attention_output_shape_is_batch_by_fusion_dim() -> None:
    fusion = _cross()
    image, question, mask = _inputs(1)
    with torch.no_grad():
        assert fusion(image, question, mask).shape == (2, FUSION_DIM)


def test_elementwise_fusion_also_uses_both_inputs() -> None:
    torch.manual_seed(0)
    fusion = ElementwiseProductFusion(IMAGE_DIM, TEXT_DIM, FUSION_DIM).eval()
    image_a, image_b = torch.randn(2, IMAGE_DIM), torch.randn(2, IMAGE_DIM)
    text_a, text_b = torch.randn(2, TEXT_DIM), torch.randn(2, TEXT_DIM)

    with torch.no_grad():
        base = fusion(image_a, text_a)
        assert (base - fusion(image_a, text_b)).abs().max() > TOLERANCE
        assert (base - fusion(image_b, text_a)).abs().max() > TOLERANCE
