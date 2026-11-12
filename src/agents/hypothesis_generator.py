
"""Hypothesis Generator: RAG-grounded, memory-aware, with citation validation."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient, complete_json
from src.schemas.hypothesis import validate_hypothesis
from src.schemas.state import FailureState, ResearchState

SYSTEM = (
    "You propose testable research hypotheses using the supplied literature summary "
    "and previous research memory as background context. "
    "Previous research memory may help identify research gaps, avoid repeated experiments, "
    "and refine hypotheses, but it is NOT a citable source. "
    "Supporting and contradicting evidence must come ONLY from the literature summary "
    "and the allowed citation keys. "
    'Reply with JSON: {"hypotheses": [{"hypothesis","motivation","supporting_evidence",'
    '"contradicting_evidence","novelty_reasoning","experiment","success_metric",'
    '"falsification_condition"}]}. '
    "supporting_evidence and contradicting_evidence must be lists of citation keys "
    "copied EXACTLY from the allowed list. Never invent citation keys. "
    "Novelty is your estimate only, not a guarantee."
)

NOVELTY_NOTE = "Novelty is an LLM estimate, not a formal literature novelty check."


def _score(h: dict) -> int:
    """Simple ranking: more supporting evidence and a stated metric rank higher."""
    return len(h.get("supporting_evidence", [])) + (
        1 if h.get("success_metric") else 0
    )


class HypothesisGenerator(BaseAgent):
    name = "hypothesis_generator"
    failure_on_error = FailureState.HYPOTHESIS_FAILED

    def __init__(self, llm: LLMClient, max_retries: int = 2):
        super().__init__()
        self.llm = llm
        self.max_retries = max_retries

    def run(self, state: ResearchState) -> dict:
        allowed = set(state.get("citations", {}))

        # Retrieve previously completed research from long-term memory.
        memory_context = state.get("memory_context") or []

        memory_items = []
        for item in memory_context:
            if not isinstance(item, dict):
                continue

            memory_items.append(
                "\n".join(
                    [
                        f"Previous research question: {item.get('research_question', 'N/A')}",
                        f"Previous hypothesis: {item.get('hypothesis', 'N/A')}",
                        f"Previous findings: {item.get('findings', 'N/A')}",
                        f"Previous verdict: {item.get('verdict', 'N/A')}",
                        f"Previous dataset: {item.get('dataset', 'N/A')}",
                    ]
                )
            )

        memory_text = (
            "\n\n".join(memory_items)
            if memory_items
            else "No relevant previous research found."
        )

        base = (
            f"Current research question:\n{state['research_question']}\n\n"
            f"Current literature summary:\n{state['literature_summary']}\n\n"
            f"Previous research memory (background only; not citable):\n"
            f"{memory_text}\n\n"
            f"Allowed citation keys from current literature: {sorted(allowed)}\n\n"
            "Instructions:\n"
            "1. Propose 3 testable candidate hypotheses.\n"
            "2. Use previous research memory only as background to identify gaps "
            "or avoid repeating earlier experiments.\n"
            "3. Cite supporting and contradicting evidence only with allowed citation keys.\n"
            "4. Do not invent citation keys or cite previous memory records.\n"
            "5. Include a measurable success metric and a falsification condition."
        )

        prompt = base
        problems = []

        for _ in range(self.max_retries + 1):
            data = complete_json(self.llm, SYSTEM, prompt)
            candidates = (
                data.get("hypotheses", [])
                if isinstance(data, dict)
                else []
            )

            valid, problems = [], []

            for i, hypothesis in enumerate(candidates):
                errs = validate_hypothesis(hypothesis, allowed)

                if errs:
                    problems.append(
                        f"candidate {i + 1}: " + "; ".join(errs)
                    )
                else:
                    valid.append(
                        {
                            **hypothesis,
                            "novelty_note": NOVELTY_NOTE,
                        }
                    )

            if valid:
                valid.sort(key=_score, reverse=True)

                return {
                    "hypotheses": valid,
                    "selected_hypothesis": valid[0],
                }

            if not candidates:
                problems.append("no hypotheses returned")

            prompt = (
                base
                + "\n\nYour previous answer was rejected for these reasons:\n- "
                + "\n- ".join(problems)
                + "\nReturn corrected JSON. Follow the citation restrictions."
            )

        return {
            "failure_state": FailureState.HYPOTHESIS_FAILED.value,
            "error": "no valid grounded hypothesis: " + " | ".join(problems),
        }
