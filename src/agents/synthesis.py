
"""Scientific Synthesis Agent: generate a research report from workflow evidence."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.schemas.state import FailureState, ResearchState


SYNTHESIS_SYSTEM = """
You are the Scientific Synthesis Agent.

Write a structured scientific research report using ONLY the evidence
provided in the workflow state.

Include:
1. Abstract.
2. Research question and motivation.
3. Literature review.
4. Hypothesis.
5. Methods and experiment plan.
6. Results, if actual results are available.
7. Critical evaluation.
8. Limitations.
9. Conclusion and future work.

Do not invent citations, experimental results, or statistical values.
Clearly distinguish proposed work, missing evidence, and unapproved findings.
"""


class SynthesisAgent(BaseAgent):
    name = "synthesis"
    failure_on_error = FailureState.SYNTHESIS_FAILED

    def __init__(self, llm: LLMClient):
        super().__init__()
        self.llm = llm

    def run(self, state: ResearchState) -> dict:
        evidence = {
            "research_question": state.get("research_question"),
            "literature_summary": state.get("literature_summary"),
            "citations": state.get("citations", {}),
            "hypotheses": state.get("hypotheses", []),
            "selected_hypothesis": state.get("selected_hypothesis"),
            "experiment_plan": state.get("experiment_plan"),
            "experiment_results": state.get("experiment_results"),
            "statistical_results": state.get("statistical_results"),
            "critic_feedback": state.get("critic_feedback"),
            "verdict": state.get("verdict"),
        }

        report = self.llm.complete(
            SYNTHESIS_SYSTEM,
            "Prepare the research report from this workflow evidence:\n"
            + str(evidence),
        )

        return {"final_report": report}
