"""Is this a question the model was trained to answer — and if not, what is the closest one?

The VQA models pick an answer from a fixed list and were trained only on the fixed question
templates of RSVQA: whether something is present, how many there are, a comparison of two
things, and whether the area is rural or urban. They have no notion of other wording, so a
differently-worded question ("Is there a river?", "How many houses?") is read as the nearest
template, and a question of a different *kind* ("What colour is the water?", "Describe this
image") is turned away with an explanation instead of being answered with a confident guess.
Pure Python, no frameworks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HELP_TEXT = (
    "This model answers four kinds of questions about a satellite image: whether something is "
    "present (“Is there a road?”), how many there are (“How many buildings are there?”), "
    "comparisons (“Are there more buildings than roads?”), and “Is it a rural or an urban "
    "area?”. It was trained on buildings (residential and commercial), roads, water areas, grass "
    "areas, forests, farmland, wetlands, parking, orchards, meadows, islands and a few other "
    "map features. It can't answer open-ended questions, colours, sizes in metres, or why."
)

RURAL_URBAN = "Is it a rural or an urban area?"


@dataclass(frozen=True)
class QuestionCheck:
    """``text`` is what to give the model. ``note`` is shown under the answer when the
    question was reworded or needs a caution. ``supported`` is False when the model should
    not be asked at all, and ``reason`` then says why."""

    supported: bool
    text: str
    note: str | None = None
    reason: str | None = None


# Specific names before general ones, so "residential buildings" is one thing, not two.
# (singular, plural as the dataset spells it, pattern)
_OBJECTS: tuple[tuple[str, str, str], ...] = (
    (
        "residential building",
        "residential buildings",
        r"\bresidential(?:\s+buildings?)?\b|\bhous(?:e|es|ing)\b|\bhomes?\b|\bapartments?\b|\bflats?\b",
    ),
    (
        "commercial building",
        "commercial buildings",
        r"\bcommercial(?:\s+buildings?)?\b|\bshops?\b|\bmalls?\b|\boffices?\b|\bmarkets?\b",
    ),
    ("building", "buildings", r"\bbuildings?\b|\bstructures?\b"),
    ("road", "roads", r"\broads?\b|\bstreets?\b|\bhighways?\b|\bmotorways?\b"),
    (
        "water area",
        "water areas",
        r"\bwater(?:\s+areas?)?\b|\brivers?\b|\blakes?\b|\bsea\b|\bponds?\b|\bstreams?\b"
        r"|\bcanals?\b|\bocean\b|\breservoirs?\b",
    ),
    ("grass area", "grass areas", r"\bgrass(?:\s+areas?|land|lands)?\b|\blawns?\b"),
    ("forest", "forests", r"\bforests?\b|\bwoodlands?\b|\btrees?\b"),
    ("farmland", "farmlands", r"\bfarm(?:land|lands|s)?\b|\bfields?\b|\bcrops?\b|\bagricultur\w*"),
    ("wetland", "wetlands", r"\bwetlands?\b|\bmarsh\w*|\bswamps?\b"),
    ("parking", "parkings", r"\bparking\b|\bcar parks?\b"),
    ("orchard", "orchards", r"\borchards?\b"),
    ("meadow", "meadows", r"\bmeadows?\b"),
    ("island", "islands", r"\bislands?\b"),
    ("scrub", "scrubs", r"\bscrubs?\b"),
    ("heath", "heaths", r"\bheaths?\b"),
    ("cemetery", "cemeterys", r"\bcemeter\w*|\bgraveyards?\b"),
    ("playground", "playgrounds", r"\bplaygrounds?\b"),
    ("farmyard", "farmyards", r"\bfarmyards?\b"),
)
_COMPILED = tuple((s, p, re.compile(pat, re.I)) for s, p, pat in _OBJECTS)

# Every object word the dataset's own questions use — a question that already starts like a
# template and names one of these is passed to the model exactly as written.
_DATASET_OBJECT = re.compile(
    r"\b(building|road|water|grass|forest|farmland|wetland|parking|orchard|scrub|pitch|meadow|"
    r"island|heath|farmyard|industrial|cemeter\w*|playground|worship|reserve|wood|garden|"
    r"village|park|school|sport\w*|golf)",
    re.I,
)
_TEMPLATE_START = re.compile(
    r"^\s*(?:is (?:a|an|there a|there an)\b"
    r"|how many\b"
    r"|what is the (?:number|amount) of\b"
    r"|are there (?:more|less) .+ than\b"
    r"|is the number of .+ equal to the number of\b)",
    re.I,
)
_DATASET_RURAL_URBAN = re.compile(r"^\s*is it a rural or an urban area\??\s*$", re.I)

_RURAL = re.compile(r"\brural\b", re.I)
_URBAN = re.compile(r"\burban\b", re.I)
_MORE = re.compile(r"\b(?:more|greater|larger|higher)\b.*\bthan\b", re.I)
_LESS = re.compile(r"\b(?:less|fewer|lower)\b.*\bthan\b", re.I)
_EQUAL = re.compile(r"\bequal\b|\bsame number\b", re.I)
_COUNT = re.compile(
    r"^\s*(?:how many|how much|what(?:'s| is) the (?:number|amount|count) of|number of|count)\b",
    re.I,
)
_PRESENCE = re.compile(
    r"^\s*(?:(?:is|are) (?:there|any|a|an)\b|(?:does|do) .*\b(?:have|contain|include|show|see)\b"
    r"|can you see\b|any\b)|\bpresent\b|\bvisible\b",
    re.I,
)
_VEGETATION = re.compile(r"\b(?:vegetation|greenery|green|plants?)\b", re.I)


def _objects_in(question: str) -> list[tuple[str, str]]:
    """(singular, plural) of every supported thing the question names, in order, each once."""
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str, str]] = []
    for singular, plural, pattern in _COMPILED:
        for match in pattern.finditer(question):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            taken.append(span)
            found.append((span[0], singular, plural))
    found.sort()
    unique: list[tuple[str, str]] = []
    for _, singular, plural in found:
        if (singular, plural) not in unique:
            unique.append((singular, plural))
    return unique


def _article(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


def _reworded(original: str, text: str) -> QuestionCheck:
    note = None
    if text.strip().lower().rstrip("?") != original.strip().lower().rstrip("?"):
        note = f"Read as “{text}” — the closest question this model was trained on."
    return QuestionCheck(supported=True, text=text, note=note)


def _unsupported(why: str) -> QuestionCheck:
    return QuestionCheck(supported=False, text="", reason=f"{why} {HELP_TEXT}")


def check_question(question: str) -> QuestionCheck:
    """Decide whether the model can be asked ``question``, and in what words."""
    text = question.strip()
    if not text:
        return _unsupported("The question is empty.")

    if _DATASET_RURAL_URBAN.match(text):
        return QuestionCheck(supported=True, text=text)
    if _TEMPLATE_START.match(text) and _DATASET_OBJECT.search(text):
        return QuestionCheck(supported=True, text=text)

    objects = _objects_in(text)

    wants_rural_urban = (_RURAL.search(text) and _URBAN.search(text)) or (
        (_RURAL.search(text) or _URBAN.search(text)) and not objects
    )
    if wants_rural_urban:
        return _reworded(text, RURAL_URBAN)

    if _MORE.search(text) or _LESS.search(text) or _EQUAL.search(text):
        if len(objects) < 2:
            return _unsupported(
                "A comparison needs two things the model knows (for example roads and buildings)."
            )
        (_, first), (_, second) = objects[0], objects[1]
        if _EQUAL.search(text):
            return _reworded(text, f"Is the number of {first} equal to the number of {second}?")
        word = "more" if _MORE.search(text) else "less"
        return _reworded(text, f"Are there {word} {first} than {second}?")

    if _COUNT.match(text):
        if objects:
            return _reworded(text, f"How many {objects[0][1]} are there?")
        if _VEGETATION.search(text):
            return _loose_vegetation(text)
        return _unsupported("I couldn't find a thing the model was trained to count.")

    if _PRESENCE.search(text):
        if len(objects) > 1:
            return _unsupported(
                "That names several things at once; ask about one at a time, or compare two "
                "(“Are there more roads than buildings?”)."
            )
        if objects:
            singular = objects[0][0]
            return _reworded(text, f"Is there {_article(singular)} {singular}?")
        if _VEGETATION.search(text):
            return _loose_vegetation(text)
        return _unsupported("I couldn't find a thing the model was trained to recognise.")

    return _unsupported("That isn't a question of the kind the model was trained on.")


def _loose_vegetation(text: str) -> QuestionCheck:
    return QuestionCheck(
        supported=True,
        text=text,
        note=(
            "“Vegetation” isn't one of the things the model was trained on (it knows grass areas, "
            "forests and farmland), so treat its answer as a rough guide. The highlight is "
            "computed from the image's colours instead."
        ),
    )
