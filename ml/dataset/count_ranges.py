"""The five count ranges RSVQA-LR is scored in, and how to spot a counting question.

RSVQA-LR's released labels are exact integers, but the original paper (Lobry et al., 2020,
Section II-B) quantises low-resolution counts into 0 / 1-10 / 11-100 / 101-1000 / >1000
before scoring, and reports 67.01% count accuracy that way. At 10 m per pixel a model cannot
honestly resolve "158 buildings" from "159", so this project reports counts in the same
ranges: the evaluation scripts score them this way (``ml/scripts/count_bucket_eval.py``) and
the web app answers count questions this way (``backend``'s model runner). Both import this
one module so the two can never disagree about where a range starts or ends.
"""

from __future__ import annotations

import re

# Stable identifiers — used as JSON keys in evaluation output.
RANGE_NAMES = ["0", "1-10", "11-100", "101-1000", ">1000"]

# What a person reads in the app.
RANGE_LABELS = ["none (0)", "1 to 10", "11 to 100", "101 to 1,000", "more than 1,000"]

# RSVQA's count questions all open one of these ways ("How many residential buildings are
# there?", "What is the number of roads?").
_COUNT_QUESTION = re.compile(r"^\s*(how many|what(?:'s| is) the number of|number of)\b", re.I)


def range_index(count: int) -> int:
    """Which of the five ranges an exact count falls in (0-4)."""
    if count == 0:
        return 0
    if count <= 10:
        return 1
    if count <= 100:
        return 2
    if count <= 1000:
        return 3
    return 4


def is_count_question(question: str) -> bool:
    return _COUNT_QUESTION.match(question) is not None
