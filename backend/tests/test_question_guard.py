"""The question guard: reword loose wording to a trained template, turn away the rest."""

from __future__ import annotations

import pytest

from app.domain.question_guard import RURAL_URBAN, check_question

# Questions copied from RSVQA's own templates must reach the model exactly as written.
TEMPLATE_QUESTIONS = [
    "Is a road present?",
    "Is there an island?",
    "Is there a water area?",
    "Is there a residential building?",
    "Is a medium commercial building present in the image?",
    "Is there a large commercial building next to a  water area?",
    "How many buildings are there?",
    "What is the number of roads?",
    "What is the amount of grass areas in the image?",
    "How many roads on the right of a  residential building are there?",
    "Are there less residential buildings than water areas?",
    "Are there more buildings than roads?",
    "Is the number of buildings equal to the number of roads?",
    "Is it a rural or an urban area",
    "Is it a rural or an urban area?",
]


@pytest.mark.parametrize("question", TEMPLATE_QUESTIONS)
def test_a_template_question_is_passed_through_unchanged(question: str) -> None:
    check = check_question(question)

    assert check.supported
    assert check.text == question.strip()
    assert check.note is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Is there a river?", "Is there a water area?"),
        ("is there a lake", "Is there a water area?"),
        ("Is a house present?", "Is there a residential building?"),
        ("Does the image contain a highway?", "Is there a road?"),
        ("Are there any trees?", "Is there a forest?"),
        ("How many houses are there?", "How many residential buildings are there?"),
        ("Count the roads", "How many roads are there?"),
        ("How many rivers?", "How many water areas are there?"),
        ("Are there fewer buildings than roads?", "Are there less buildings than roads?"),
        ("Is there more water than farmland?", "Are there more water areas than farmlands?"),
        (
            "Is the number of houses the same number as roads?",
            "Is the number of residential buildings equal to the number of roads?",
        ),
        ("Is this an urban area?", RURAL_URBAN),
        ("Is it urban or rural?", RURAL_URBAN),
    ],
)
def test_loose_wording_is_read_as_the_nearest_trained_question(
    question: str, expected: str
) -> None:
    check = check_question(question)

    assert check.supported
    assert check.text == expected
    assert check.note is not None and expected in check.note


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        "Describe this image.",
        "What is in this picture?",
        "What colour is the water?",
        "Why is the river brown?",
        "Is the road wide?",
        "How many cars are there?",
        "Is there more water than land?",  # "land" isn't something the model knows
        "Is the river bigger than the lake?",  # one thing, twice
        "Does it contain a road and a river?",  # several things at once
    ],
)
def test_a_different_kind_of_question_is_turned_away_with_what_can_be_asked(
    question: str,
) -> None:
    check = check_question(question)

    assert not check.supported
    assert check.reason is not None
    assert "Is there a road?" in check.reason


def test_vegetation_is_let_through_with_a_caution_not_rewritten() -> None:
    check = check_question("Is there any vegetation?")

    assert check.supported
    assert check.text == "Is there any vegetation?"
    assert check.note is not None and "rough guide" in check.note


def test_the_note_is_absent_when_rewording_changes_nothing_but_case_or_punctuation() -> None:
    check = check_question("is there a river")
    assert check.text == "Is there a water area?"

    same = check_question("how many roads are there")
    assert same.supported
    assert same.note is None
