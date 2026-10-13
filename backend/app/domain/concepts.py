"""What a question is about, so the app can point at it on the image.

The VQA model answers a question with a label; it has no notion of *where* anything is.
Pointing at the answer ("yes, there are roads — here") is done by a separate component, and
this module decides what to point at: every thing the question names (RSVQA's comparison
questions name two), and what *kind* of question it is. Pure Python, no frameworks.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal


class HighlightConcept(StrEnum):
    VEGETATION = "vegetation"
    FOREST = "forest"
    GRASS = "grass"
    FARMLAND = "farmland"
    WATER = "water"
    BUILDINGS = "buildings"
    ROADS = "roads"
    CEMETERY = "cemetery"


# Word stems per concept, matched at a word start. These cover the object names the RSVQA
# question templates use ("residential buildings", "water areas", "grass areas", "forests",
# "farmlands", "roads", "cemetery") plus a few everyday synonyms.
_PATTERNS: dict[HighlightConcept, re.Pattern[str]] = {
    HighlightConcept.VEGETATION: re.compile(r"\b(vegetation|tree|crop|green)", re.I),
    HighlightConcept.FOREST: re.compile(r"\b(forest|wood)", re.I),
    HighlightConcept.GRASS: re.compile(r"\b(grass|meadow|lawn)", re.I),
    HighlightConcept.FARMLAND: re.compile(r"\b(farm|agricultur|field)", re.I),
    HighlightConcept.WATER: re.compile(r"\b(water|river|lake|sea\b|pond|stream|canal|ocean)", re.I),
    HighlightConcept.BUILDINGS: re.compile(
        r"\b(building|house|housing|residential|commercial)", re.I
    ),
    HighlightConcept.ROADS: re.compile(r"\b(road|street|highway|motorway)", re.I),
    HighlightConcept.CEMETERY: re.compile(r"\b(cemeter|graveyard)", re.I),
}

# Words describing a thing's size or shape ("a large road", "a circular water area"). The
# highlight shows the thing without regard to them, and says so.
_MODIFIERS = re.compile(r"\b(large|big|medium|small|tiny|circular|rectangular|square)\b", re.I)

_COUNT = re.compile(r"^\s*(how many|what(?:'s| is) the (?:number|amount) of|number of)\b", re.I)
_COMPARISON = re.compile(r"\bthan\b|\bequal to the number of\b", re.I)

# At most this many things are drawn at once — a comparison names two.
MAX_TARGETS = 2

QuestionKind = Literal["presence", "count", "comparison"]


def resolve_targets(question: str) -> tuple[HighlightConcept, ...]:
    """Every supported thing ``question`` names, in the order it names them (at most two)."""
    found: list[tuple[int, HighlightConcept]] = []
    for concept, pattern in _PATTERNS.items():
        match = pattern.search(question)
        if match:
            found.append((match.start(), concept))
    found.sort(key=lambda item: item[0])
    return tuple(concept for _, concept in found[:MAX_TARGETS])


def question_kind(question: str) -> QuestionKind:
    """Whether the question asks if something is there, how many, or compares two things.

    It matters for what a "no" means: for a presence question it says the thing is absent
    (so there's nothing to point at); for a comparison it only says one count isn't larger.
    """
    if _COMPARISON.search(question):
        return "comparison"
    if _COUNT.match(question):
        return "count"
    return "presence"


def ignored_modifiers(question: str) -> list[str]:
    """Size/shape words in the question that the highlight can't take into account."""
    seen: list[str] = []
    for match in _MODIFIERS.finditer(question):
        word = match.group(1).lower()
        if word not in seen:
            seen.append(word)
    return seen
