"""Hypothesis schema and a pure-code validator (no LLM involved)."""
from __future__ import annotations

from pydantic import BaseModel, Field

REQUIRED_TEXT = ["hypothesis", "motivation", "novelty_reasoning", "experiment",
                 "success_metric", "falsification_condition"]


class Hypothesis(BaseModel):
    hypothesis: str
    motivation: str
    supporting_evidence: list[str] = Field(default_factory=list)  # citation_keys
    contradicting_evidence: list[str] = Field(default_factory=list)
    novelty_reasoning: str
    experiment: str
    success_metric: str
    falsification_condition: str


def validate_hypothesis(h: dict, allowed_keys: set[str]) -> list[str]:
    """Return a list of problems (empty list means valid)."""
    errors = []
    for name in REQUIRED_TEXT:
        if not str(h.get(name, "")).strip():
            errors.append(f"missing or empty field: {name}")
    support = h.get("supporting_evidence")
    if not isinstance(support, list) or not support:
        errors.append("supporting_evidence must be a non-empty list of citation keys")
        support = []
    contra = h.get("contradicting_evidence", [])
    if not isinstance(contra, list):
        errors.append("contradicting_evidence must be a list")
        contra = []
    for key in list(support) + list(contra):
        if key not in allowed_keys:
            errors.append(f"citation key not in retrieved sources: {key}")
    return errors
