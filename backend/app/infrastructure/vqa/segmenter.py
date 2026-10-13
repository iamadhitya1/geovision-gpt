"""Where are the things the question named? Highlighting for the web app.

The trained VQA model answers with a label and cannot locate anything — it was never shown a
segmentation mask, and none of the datasets used in this project contain one. So pointing at
"the roads" is done by a separate step that needs no training data of our own, and the right
method depends on what is being pointed at, because at ~10 m per pixel the image alone only
shows some things:

* vegetation, water - from the image: a colour index (excess green) for vegetation, and
                      CLIPSeg (Lueddecke & Ecker, 2022) restricted to dark pixels for water.
* roads, buildings, forest, grass, farmland, cemetery - OpenStreetMap map data, drawn at the
                      scene's coordinates, when the image is georeferenced and the map data is
                      reachable. A road is about one pixel wide here and a forest looks like any
                      other green field, so no image-only method can honestly find them.
                      This is **map data, not something detected in the image**, and is
                      labelled that way. Where it isn't available, buildings fall back to
                      CLIPSeg (built-up areas) and forest/grass/farmland to the colour index
                      (all green vegetation, types not told apart); roads and cemeteries have
                      no image-only fallback and say so.

Every layer reports which method produced it and a caveat wherever the method is weak.

CLIPSeg's weights (~577 MB) are loaded on first use and cached by Hugging Face — see the
"Everything the app needs that isn't in git" section of the root README.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError

from app.core.logging import get_logger
from app.domain.concepts import HighlightConcept
from app.domain.entities import ConceptHighlight, HighlightLayer, MapFeature
from app.domain.errors import CorruptImageError, ImageryUnavailableError, ModelUnavailableError
from app.domain.interfaces import MapDataPort

logger = get_logger(__name__)

Bounds = tuple[float, float, float, float]  # west, south, east, north

# Long edge the analysis runs at. Satellite chips here are 256 px; this only matters for
# larger uploads, where it keeps CPU time bounded (the overlay scales back up in the page).
_MAX_SIDE = 512

# Excess-green threshold, tuned by eye on the demo images: fields and woodland clear it,
# bare soil, water, roofs and roads do not.
_VEGETATION_EXG_THRESHOLD = 0.05

# CLIPSeg's scores are relative, so they're rescaled to 0-1 per image and cut at this level.
_CLIPSEG_RELATIVE_THRESHOLD = 0.6

# Water is dark in RGB imagery; bright sand and surf that CLIPSeg also calls "water" are not.
_WATER_MAX_BRIGHTNESS = 110

_FROM_MAP_DATA = {
    HighlightConcept.ROADS,
    HighlightConcept.BUILDINGS,
    HighlightConcept.FOREST,
    HighlightConcept.GRASS,
    HighlightConcept.FARMLAND,
    HighlightConcept.CEMETERY,
}
_VEGETATION_LIKE = {
    HighlightConcept.VEGETATION,
    HighlightConcept.FOREST,
    HighlightConcept.GRASS,
    HighlightConcept.FARMLAND,
}

_LABELS = {
    HighlightConcept.VEGETATION: "vegetation",
    HighlightConcept.FOREST: "forest",
    HighlightConcept.GRASS: "grass",
    HighlightConcept.FARMLAND: "farmland",
    HighlightConcept.WATER: "water",
    HighlightConcept.BUILDINGS: "buildings",
    HighlightConcept.ROADS: "roads",
    HighlightConcept.CEMETERY: "cemetery",
}

_COLORS = {
    HighlightConcept.VEGETATION: (70, 220, 110),
    HighlightConcept.FOREST: (30, 170, 70),
    HighlightConcept.GRASS: (150, 230, 80),
    HighlightConcept.FARMLAND: (220, 200, 70),
    HighlightConcept.WATER: (80, 170, 255),
    HighlightConcept.BUILDINGS: (255, 170, 60),
    HighlightConcept.ROADS: (255, 90, 90),
    HighlightConcept.CEMETERY: (190, 120, 255),
}

_METHOD_COLOUR_INDEX = "colour index (excess green) — not a trained model"
_METHOD_CLIPSEG = "CLIPSeg zero-shot, text-prompted — not trained on this data"
_METHOD_MAP_DATA = "OpenStreetMap map data for this location — not detected from the image"

_CAVEAT_MAP_DATA = (
    "Drawn from map data at this scene's coordinates, so it can be newer or older than the "
    "imagery, and anything nobody has mapped is missing. It is not something the model found."
)
_CAVEAT_ROADS_DRAWN_WIDE = " Roads are drawn wider than they really are so they can be seen."
_CAVEAT_GREEN_TYPES = (
    "Marks all green vegetation: from the image alone it can't tell forest from grass from "
    "farmland."
)
_CAVEAT_BUILT_UP = (
    "Shows built-up areas. At this resolution a single building is only a few pixels wide, so "
    "individual buildings can't be picked out from the image."
)
# Why map data wasn't used. Phrased as a reason so it reads correctly both ways: as a note
# that an image-based fallback was used, and as the explanation of why a thing can't be shown.
_REASON_NO_LOCATION = "this image has no location"
_REASON_MAP_DATA_OFF = "map data is switched off on this server"
_REASON_MAP_DATA_UNREACHABLE = "map data couldn't be reached"


def _hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


# --- geometry -------------------------------------------------------------------------------


def _mercator_y(latitude: float) -> float:
    return math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2))


def rasterise(
    features: Sequence[MapFeature], bounds: Bounds, size: tuple[int, int], *, line_width: int
) -> np.ndarray:
    """Draw map features onto a boolean mask the shape of the image.

    ``bounds`` is the image's (west, south, east, north); the image is in Web Mercator, so
    latitude is projected the same way rather than interpolated linearly.
    """
    west, south, east, north = bounds
    width, height = size
    y_north, y_south = _mercator_y(north), _mercator_y(south)

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = (lon - west) / (east - west) * (width - 1)
        y = (y_north - _mercator_y(lat)) / (y_north - y_south) * (height - 1)
        return x, y

    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for feature in features:
        points = [project(lon, lat) for lon, lat in feature.points]
        if feature.kind == "polygon" and len(points) >= 3:
            draw.polygon(points, fill=255)
            draw.line([*points, points[0]], fill=255, width=1)  # keeps tiny polygons visible
        elif len(points) >= 2:
            draw.line(points, fill=255, width=line_width)
    return np.asarray(np.asarray(canvas) > 127, dtype=bool)


# --- the segmenter ------------------------------------------------------------------------


class ZeroShotSegmenter:
    """Implements :class:`~app.domain.interfaces.SegmentationPort`."""

    def __init__(
        self,
        *,
        clipseg_model_id: str,
        device: str = "cpu",
        map_data: MapDataPort | None = None,
    ) -> None:
        self._model_id = clipseg_model_id
        self._device = torch.device("cuda" if device == "cuda" else "cpu")
        self._map_data = map_data
        self._lock = threading.Lock()
        self._processor: Any | None = None
        self._model: Any | None = None

    # --- loading ------------------------------------------------------------

    def _ensure_clipseg(self) -> tuple[Any, Any]:
        """Loaded on first use: ~577 MB and a few seconds, wasted on a server that is only
        ever asked about things that come from map data."""
        with self._lock:
            if self._model is None:
                try:
                    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

                    processor: Any = CLIPSegProcessor.from_pretrained(self._model_id)
                    model: Any = CLIPSegForImageSegmentation.from_pretrained(self._model_id)
                except Exception as exc:
                    logger.warning("clipseg_unavailable", model_id=self._model_id, error=str(exc))
                    raise ModelUnavailableError(
                        f"The segmentation model '{self._model_id}' isn't available on this "
                        "server. Download it once with the command in the README's "
                        "'Everything the app needs that isn't in git' section.",
                        model=self._model_id,
                    ) from exc
                self._model = model.to(self._device).eval()
                self._processor = processor
                logger.info("clipseg_loaded", model_id=self._model_id, device=str(self._device))
        assert self._processor is not None and self._model is not None
        return self._processor, self._model

    def warm_up(self) -> None:
        """Load the weights now rather than on the first request. A missing download is
        logged by ``_ensure_clipseg`` and ignored here: the first real request will report it."""
        with contextlib.suppress(ModelUnavailableError):
            self._ensure_clipseg()

    # --- port ---------------------------------------------------------------

    async def highlight(
        self,
        image_data: bytes,
        concepts: Sequence[HighlightConcept],
        *,
        bounds_wgs84: Bounds | None = None,
    ) -> ConceptHighlight:
        image = await asyncio.to_thread(self._decode, image_data)
        built = await asyncio.gather(
            *(self._layer(concept, image, bounds_wgs84) for concept in concepts)
        )
        overlay = await asyncio.to_thread(
            _render_overlay,
            [(mask, _COLORS[HighlightConcept(layer.concept)]) for layer, mask in built],
        )
        return ConceptHighlight(layers=tuple(layer for layer, _ in built), overlay_png=overlay)

    # --- one thing ----------------------------------------------------------

    async def _layer(
        self, concept: HighlightConcept, image: Image.Image, bounds: Bounds | None
    ) -> tuple[HighlightLayer, np.ndarray]:
        reason: str | None = None  # why map data wasn't used, if it wasn't
        caveat: str | None

        if concept in _FROM_MAP_DATA:
            if bounds is None:
                reason = _REASON_NO_LOCATION
            elif self._map_data is None:
                reason = _REASON_MAP_DATA_OFF
            else:
                try:
                    features = await self._map_data.fetch(concept, bounds)
                except ImageryUnavailableError:
                    reason = _REASON_MAP_DATA_UNREACHABLE
                else:
                    width = max(2, round(image.width / 128))
                    mask = rasterise(features, bounds, image.size, line_width=width)
                    caveat = _CAVEAT_MAP_DATA + (
                        _CAVEAT_ROADS_DRAWN_WIDE if concept is HighlightConcept.ROADS else ""
                    )
                    return self._make_layer(concept, mask, _METHOD_MAP_DATA, caveat), mask

        # Image-based methods: used outright for vegetation and water, and as the fallback for
        # the things that prefer map data.
        if concept in _VEGETATION_LIKE:
            mask = await asyncio.to_thread(self._vegetation_mask, image)
            caveat = None if concept is HighlightConcept.VEGETATION else _CAVEAT_GREEN_TYPES
            method = _METHOD_COLOUR_INDEX
        elif concept is HighlightConcept.WATER:
            mask = await asyncio.to_thread(self._clipseg_mask, image, concept)
            caveat, method = None, _METHOD_CLIPSEG
        elif concept is HighlightConcept.BUILDINGS:
            mask = await asyncio.to_thread(self._clipseg_mask, image, concept)
            caveat, method = _CAVEAT_BUILT_UP, _METHOD_CLIPSEG
        else:
            # Roads and cemeteries: no honest image-only method exists at this resolution.
            raise ImageryUnavailableError(
                f"Showing {_LABELS[concept]} needs map data (it needs the image's location and "
                f"an internet connection), but {reason or 'it is not available'}.",
                concept=concept.value,
            )

        if reason:
            note = f"{reason[0].upper()}{reason[1:]}, so this was found from the image instead."
            caveat = f"{note} {caveat}" if caveat else note
        mask = _smooth(mask)
        return self._make_layer(concept, mask, method, caveat), mask

    @staticmethod
    def _make_layer(
        concept: HighlightConcept, mask: np.ndarray, method: str, caveat: str | None
    ) -> HighlightLayer:
        return HighlightLayer(
            concept=concept.value,
            label=_LABELS[concept],
            color=_hex(_COLORS[concept]),
            coverage=float(mask.mean()),
            method=method,
            caveat=caveat,
        )

    # --- image-based masks (each returns a boolean H x W array) ---------------

    @staticmethod
    def _decode(image_data: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise CorruptImageError(
                "Couldn't decode this image for highlighting — it may not actually be a "
                "valid raster despite passing upload validation."
            ) from exc
        if max(image.size) > _MAX_SIDE:
            scale = _MAX_SIDE / max(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    @staticmethod
    def _vegetation_mask(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image).astype(np.float32)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        excess_green = (2 * g - r - b) / (r + g + b + 1e-6)
        return np.asarray(excess_green > _VEGETATION_EXG_THRESHOLD, dtype=bool)

    def _clipseg_mask(self, image: Image.Image, concept: HighlightConcept) -> np.ndarray:
        processor, model = self._ensure_clipseg()
        prompt = "water" if concept is HighlightConcept.WATER else "buildings"
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        prob = torch.sigmoid(logits).reshape(logits.shape[-2], logits.shape[-1]).cpu().numpy()

        # Resize the model's fixed-size output back to the image, then rescale to 0-1.
        prob_image = Image.fromarray((prob * 255).astype(np.uint8)).resize(
            image.size, Image.Resampling.BILINEAR
        )
        scores = np.asarray(prob_image).astype(np.float32) / 255.0
        spread = float(scores.max() - scores.min())
        mask = np.asarray(
            ((scores - scores.min()) / (spread + 1e-6)) > _CLIPSEG_RELATIVE_THRESHOLD, dtype=bool
        )
        if spread < 1e-3:  # a flat response means "nothing here", not "everything here"
            mask = np.zeros_like(mask)

        if concept is HighlightConcept.WATER:
            brightness = np.asarray(image.convert("L"))
            mask &= brightness < _WATER_MAX_BRIGHTNESS
        return mask


# --- rendering ------------------------------------------------------------------------------


def _smooth(mask: np.ndarray) -> np.ndarray:
    """Tidy a raw image-based mask so the highlight reads as regions, not noise: a median
    filter drops single-pixel speckle, then a close (grow, shrink) fills hairline gaps such as
    a ditch between two fields, and an open (shrink, grow) removes the specks that are left.
    Not applied to map data — it would erase the thin lines roads are drawn as."""
    image = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    image = image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))  # close
    image = image.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))  # open
    return np.asarray(np.asarray(image) > 127, dtype=bool)


def _render_overlay(layers: Sequence[tuple[np.ndarray, tuple[int, int, int]]]) -> bytes:
    """Transparent PNG: everything outside every layer darkened, each layer tinted and edged
    in its own colour."""
    height, width = layers[0][0].shape
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    union = np.zeros((height, width), dtype=bool)
    for mask, _ in layers:
        union |= mask
    overlay[~union] = (0, 0, 0, 140)  # dim everything that is NOT one of the things asked about

    for mask, color in layers:
        overlay[mask] = (*color, 60)  # light tint, so the area itself reads
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        grown = np.asarray(mask_image.filter(ImageFilter.MaxFilter(3))) > 127
        shrunk = np.asarray(mask_image.filter(ImageFilter.MinFilter(3))) > 127
        overlay[grown & ~shrunk] = (*color, 255)  # a thin edge

    buffer = io.BytesIO()
    Image.fromarray(overlay, mode="RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
