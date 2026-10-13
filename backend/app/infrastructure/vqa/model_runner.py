"""PyTorch-backed implementation of :class:`~app.domain.interfaces.VqaModelPort`.

The only place ``torch``, ``transformers``, and ``timm`` are allowed to be imported
outside ``ml/`` itself, mirroring how ``RasterImageAnalyzer`` owns ``rasterio``. Model
classes come from ``ml/models/vqa_model.py`` (installed as a local path dependency — see
``pyproject.toml``) rather than being duplicated here, so training and serving can never
silently drift onto two different architectures.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from dataset.count_ranges import RANGE_LABELS, is_count_question
from dataset.rsvqa_lr_dataset import AnswerVocabulary
from models.vqa_model import (
    CrossAttentionFusion,
    ImageEncoderName,
    LitRSVQAModel,
    build_image_transform,
    build_tokenizer,
)
from PIL import Image, UnidentifiedImageError

from app.core.logging import get_logger
from app.domain.entities import AnswerAlternative, AttentionHeatmap, VqaAnswer
from app.domain.enums import FusionType
from app.domain.errors import CorruptImageError, ModelUnavailableError
from app.infrastructure.vqa.count_decoding import CountDecoder

# XCiT-Nano prepends one summary ("class") token at index 0 before its 196 real patch
# tokens (confirmed against timm's xcit.py source, not assumed) — the heatmap drops that
# one and reshapes the remaining 196 into the image's 14x14 patch grid.
_ATTENTION_GRID_SIZE = 14

_COUNT_NOTE = (
    "Counts are given as a range, the way the original RSVQA benchmark scores them — an exact "
    "number can't be read reliably from imagery at this resolution."
)

logger = get_logger(__name__)


def resolve_device(preference: str) -> torch.device:
    """``preference`` is ``"auto"``, ``"cpu"``, or ``"cuda"`` — see ``GVG_VQA_DEVICE`` in
    ``core/config.py``. ``"auto"`` mirrors ``ml/scripts/train.py``'s own device selection,
    so the same trained weights behave identically whichever way this process picks."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TorchVqaModelRunner:
    """Loads whichever trained checkpoints exist on disk and serves predictions from them.

    A missing checkpoint file is logged and skipped, not a startup failure — the model
    track and the web-app track of this project were built on different schedules (see
    ``ml/README.md``), and the app must still start and serve the image-management
    features even before a checkpoint has been copied down from the HPC cluster.
    """

    def __init__(
        self,
        *,
        checkpoint_paths: dict[FusionType, Path],
        vocab_path: Path,
        device: str = "auto",
        default_image_encoder: ImageEncoderName = "xcit_nano",
    ) -> None:
        self._device = resolve_device(device)
        self._tokenizer: Any | None = None  # built lazily — see _tokenizer_instance()
        self._image_transforms: dict[ImageEncoderName, Callable[[Image.Image], torch.Tensor]] = {}
        self._models: dict[FusionType, LitRSVQAModel] = {}
        self._model_encoders: dict[FusionType, ImageEncoderName] = {}

        self._vocab: AnswerVocabulary | None = None
        self._count_decoder: CountDecoder | None = None
        if vocab_path.exists():
            self._vocab = AnswerVocabulary.from_json(vocab_path)
            self._count_decoder = CountDecoder(self._vocab)
        else:
            logger.warning("vqa_vocab_missing", path=str(vocab_path))

        for fusion, path in checkpoint_paths.items():
            if not path.exists():
                logger.warning("vqa_checkpoint_missing", fusion=fusion.value, path=str(path))
                continue
            self._models[fusion] = self._load_checkpoint(path, fusion)

        logger.info(
            "vqa_model_runner_ready",
            device=str(self._device),
            loaded_fusions=[f.value for f in self._models],
        )

    def _load_checkpoint(self, path: Path, fusion: FusionType) -> LitRSVQAModel:
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        config = checkpoint["config"]
        model = LitRSVQAModel(config, fusion=fusion.value)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()
        self._model_encoders[fusion] = config.image_encoder
        return model

    def _transform_for(self, encoder: ImageEncoderName) -> Callable[[Image.Image], torch.Tensor]:
        if encoder not in self._image_transforms:
            self._image_transforms[encoder] = build_image_transform(encoder)
        return self._image_transforms[encoder]

    def _tokenizer_instance(self) -> Any:
        """Loading BERT-Tiny's tokenizer costs real startup time (and, without a local HF
        cache, a network round-trip) — not worth paying when no checkpoint ever actually
        loaded, which is the common case on a machine the trained weights haven't been
        copied to yet (see the class docstring)."""
        if self._tokenizer is None:
            self._tokenizer = build_tokenizer()
        return self._tokenizer

    def available_fusions(self) -> frozenset[FusionType]:
        return frozenset(self._models.keys())

    async def predict(
        self, image_data: bytes, question: str, *, fusion: FusionType, top_k: int = 3
    ) -> VqaAnswer:
        if fusion not in self._models or self._vocab is None:
            raise ModelUnavailableError(fusion=fusion.value)

        return await asyncio.to_thread(self._predict_sync, image_data, question, fusion, top_k)

    def _predict_sync(
        self, image_data: bytes, question: str, fusion: FusionType, top_k: int
    ) -> VqaAnswer:
        assert self._vocab is not None  # checked in predict()
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise CorruptImageError(
                "Couldn't decode this image for the model — it may not actually be a "
                "valid raster despite passing upload validation."
            ) from exc

        model = self._models[fusion]
        transform = self._transform_for(self._model_encoders[fusion])
        pixel_values = transform(image).unsqueeze(0).to(self._device)

        tokenizer = self._tokenizer_instance()
        encoded = tokenizer([question], padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = model(pixel_values, input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)[0]

        if self._count_decoder is not None and is_count_question(question):
            return self._count_answer(fusion, logits[0], top_k)

        k = min(top_k, probabilities.shape[0])
        top_probs, top_indices = torch.topk(probabilities, k=k)

        alternatives = tuple(
            AnswerAlternative(answer=self._vocab.decode(int(index)), confidence=float(prob))
            for prob, index in zip(top_probs.tolist(), top_indices.tolist(), strict=True)
        )
        return VqaAnswer(
            fusion=fusion,
            answer=alternatives[0].answer,
            confidence=alternatives[0].confidence,
            alternatives=alternatives,
        )

    def _count_answer(self, fusion: FusionType, logits: torch.Tensor, top_k: int) -> VqaAnswer:
        """A counting question, answered as the most probable *range* — see
        :mod:`app.infrastructure.vqa.count_decoding` for why this beats the exact-number
        argmax. ``confidence`` is that range's total probability."""
        assert self._count_decoder is not None  # checked by the caller
        range_probs = self._count_decoder.range_probabilities(logits)
        ranked = sorted(enumerate(range_probs), key=lambda item: item[1], reverse=True)
        alternatives = tuple(
            AnswerAlternative(answer=RANGE_LABELS[index], confidence=prob)
            for index, prob in ranked[: max(top_k, 1)]
        )
        return VqaAnswer(
            fusion=fusion,
            answer=alternatives[0].answer,
            confidence=alternatives[0].confidence,
            alternatives=alternatives,
            note=_COUNT_NOTE,
        )

    async def explain(self, image_data: bytes, question: str) -> AttentionHeatmap:
        if FusionType.CROSS_ATTENTION not in self._models or self._vocab is None:
            raise ModelUnavailableError(fusion=FusionType.CROSS_ATTENTION.value)

        return await asyncio.to_thread(self._explain_sync, image_data, question)

    def _explain_sync(self, image_data: bytes, question: str) -> AttentionHeatmap:
        assert self._vocab is not None  # checked in explain()
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise CorruptImageError(
                "Couldn't decode this image for the model — it may not actually be a "
                "valid raster despite passing upload validation."
            ) from exc

        model = self._models[FusionType.CROSS_ATTENTION]
        assert isinstance(model.fusion, CrossAttentionFusion)  # guaranteed by how it was loaded
        transform = self._transform_for(self._model_encoders[FusionType.CROSS_ATTENTION])
        pixel_values = transform(image).unsqueeze(0).to(self._device)

        tokenizer = self._tokenizer_instance()
        encoded = tokenizer([question], padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.no_grad():
            logits, attn = model(pixel_values, input_ids, attention_mask, return_attention=True)
            probabilities = torch.softmax(logits, dim=-1)[0]

            # attn: (1, L, N) — average over real question words only, same masking logic
            # as the fusion's own pooling step, so padding never dilutes the heatmap.
            mask = attention_mask[0].to(attn.dtype)  # (L,)
            per_patch = (attn[0] * mask.unsqueeze(-1)).sum(dim=0) / mask.sum().clamp(min=1.0)
            patch_weights = per_patch[1:]  # drop index 0, the summary token — see module docstring
            # Renormalise over the 196 real patches alone, since the summary token also
            # absorbed some attention mass that a patch-only heatmap shouldn't silently drop.
            patch_weights = patch_weights / patch_weights.sum().clamp(min=1e-8)

        top_prob, top_index = torch.topk(probabilities, k=1)
        return AttentionHeatmap(
            fusion=FusionType.CROSS_ATTENTION,
            answer=self._vocab.decode(int(top_index.item())),
            confidence=float(top_prob.item()),
            grid_size=_ATTENTION_GRID_SIZE,
            weights=tuple(patch_weights.tolist()),
        )
