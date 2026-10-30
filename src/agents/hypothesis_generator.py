"""Hypothesis Generator: RAG-grounded, with a code-based citation validator."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient, complete_json
from src.schemas.hypothesis import validate_hypothesis
from src.schemas.state import FailureState, ResearchState

SYSTEM = (
    "You propose testable research hypotheses using ONLY the literature summary and the allowed citation keys. "
    'Reply with JSON: {"hypotheses": [{"hypothesis","motivation","supporting_evidence","contradicting_evidence",'
    '"novelty_reasoning","experiment","success_metric","falsification_condition"}]}. '
    "supporting_evidence and contradicting_evidence are lists of citation keys copied EXACTLY from the allowed list. "
    "Novelty is your estimate only, not a guarantee."
)

NOVELTY_NOTE = "Novelty is an LLM estimate, not a formal literature novelty check."


def _score(h: dict) -> int:
    """Simple ranking: more supporting evidence and a stated metric rank higher."""
    return len(h.get("supporting_evidence", [])) + (1 if h.get("success_metric") else 0)


class HypothesisGenerator(BaseAgent):
    name = "hypothesis_generator"
    failure_on_error = FailureState.HYPOTHESIS_FAILED

    def __init__(self, llm: LLMClient, max_retries: int = 2):
        super().__init__()
        self.llm, self.max_retries = llm, max_retries

    def run(self, state: ResearchState) -> dict:
        allowed = set(state.get("citations", {}))
        base = (f"Research question: {state['research_question']}\n\nLiterature summary:\n"
                f"{state['literature_summary']}\n\nAllowed citation keys: {sorted(allowed)}\n"
                "Propose 3 candidate hypotheses.")
        prompt, problems = base, []
        for _ in range(self.max_retries + 1):
            data = complete_json(self.llm, SYSTEM, prompt)
            candidates = data.get("hypotheses", []) if isinstance(data, dict) else []
            valid, problems = [], []
            for i, h in enumerate(candidates):
                errs = validate_hypothesis(h, allowed)
                if errs:
                    problems.append(f"candidate {i + 1}: " + "; ".join(errs))
                else:
                    valid.append({**h, "novelty_note": NOVELTY_NOTE})
            if valid:
                valid.sort(key=_score, reverse=True)
                return {"hypotheses": valid, "selected_hypothesis": valid[0]}
            if not candidates:
                problems.append("no hypotheses returned")
            prompt = base + "\n\nYour previous answer was rejected for these reasons:\n- " + "\n- ".join(problems)
        return {"failure_state": FailureState.HYPOTHESIS_FAILED.value,
                "error": "no valid grounded hypothesis: " + " | ".join(problems)}
