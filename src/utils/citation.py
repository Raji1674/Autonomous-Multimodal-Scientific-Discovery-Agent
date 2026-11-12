"""Deterministic citation keys: surname + year + first meaningful title word."""
from __future__ import annotations

import re
import string
from collections.abc import Iterable

from src.schemas.paper import Paper

_STOP = {"a", "an", "the", "on", "of", "for", "in", "and", "to", "with", "is", "are", "by", "at"}


def _surname(authors: list[str]) -> str:
    if not authors:
        return "anon"
    name = authors[0].strip()
    surname = name.split(",")[0] if "," in name else (name.split()[-1] if name.split() else "")
    surname = re.sub(r"[^a-z0-9]", "", surname.lower())
    return surname or "anon"


def _year(date: str) -> str:
    return date[:4] if len(date) >= 4 and date[:4].isdigit() else "nd"


def _title_word(title: str) -> str:
    for tok in re.findall(r"[A-Za-z0-9]+", title.lower()):
        if tok not in _STOP:
            return tok
    return "paper"


def make_citation_key(paper: Paper, existing: Iterable[str] = ()) -> str:
    """Return a citation key; append b, c, ... if the base key already exists."""
    base = f"{_surname(paper.authors)}{_year(paper.date)}{_title_word(paper.title)}"
    taken = set(existing)
    if base not in taken:
        return base
    for letter in string.ascii_lowercase[1:]:
        if base + letter not in taken:
            return base + letter
    raise ValueError(f"Too many citation key collisions for {base}")
