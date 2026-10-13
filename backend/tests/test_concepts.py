"""What a question is about, so the app can point at it — see ``app/domain/concepts.py``."""

from __future__ import annotations

import pytest

from app.domain.concepts import (
    HighlightConcept,
    ignored_modifiers,
    question_kind,
    resolve_targets,
)

C = HighlightConcept


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is there any vegetation?", (C.VEGETATION,)),
        ("How many grass areas are there?", (C.GRASS,)),
        ("Is there a forest?", (C.FOREST,)),
        ("Is a farmland present?", (C.FARMLAND,)),
        ("Is there a water area?", (C.WATER,)),
        ("What is the number of rivers?", (C.WATER,)),
        ("How many residential buildings are there?", (C.BUILDINGS,)),
        ("Is there a commercial building?", (C.BUILDINGS,)),
        ("Is there a road?", (C.ROADS,)),
        ("Is a large road present?", (C.ROADS,)),
        ("What is the number of roads?", (C.ROADS,)),
        ("Is a cemetery present?", (C.CEMETERY,)),
    ],
)
def test_a_question_naming_one_thing_resolves_to_it(
    question: str, expected: tuple[HighlightConcept, ...]
) -> None:
    assert resolve_targets(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Are there more roads than buildings?", (C.ROADS, C.BUILDINGS)),
        ("Are there less residential buildings than water areas?", (C.BUILDINGS, C.WATER)),
        (
            "Is the number of water areas equal to the number of roads?",
            (C.WATER, C.ROADS),
        ),
        ("Are there less roads than residential buildings in the image?", (C.ROADS, C.BUILDINGS)),
    ],
)
def test_a_comparison_names_two_things_in_the_order_it_names_them(
    question: str, expected: tuple[HighlightConcept, ...]
) -> None:
    assert resolve_targets(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "Is it a rural or an urban area?",  # about the whole scene, not one thing
        "Is there a bridge?",  # a real thing, but nothing here can show it
        "Is this a research site?",  # contains "sea" only inside another word
    ],
)
def test_a_question_naming_nothing_supported_resolves_to_nothing(question: str) -> None:
    assert resolve_targets(question) == ()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is there a road?", "presence"),
        ("Is a water area present?", "presence"),
        ("Is it a rural or an urban area?", "presence"),
        ("How many roads are there?", "count"),
        ("What is the number of buildings?", "count"),
        ("What is the amount of forests?", "count"),
        ("Are there more roads than buildings?", "comparison"),
        ("Are there less buildings than water areas?", "comparison"),
        ("Is the number of roads equal to the number of water areas?", "comparison"),
    ],
)
def test_question_kind(question: str, expected: str) -> None:
    assert question_kind(question) == expected


def test_size_and_shape_words_are_reported_once_each() -> None:
    assert ignored_modifiers("Is there a large circular water area?") == ["large", "circular"]
    assert ignored_modifiers("Is a small road or a small building present?") == ["small"]


def test_a_question_with_no_size_or_shape_words_reports_none() -> None:
    assert ignored_modifiers("Is there a road?") == []
