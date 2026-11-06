
import json

from src.agents.hypothesis_generator import HypothesisGenerator
from src.schemas.state import FailureState


VALID_HYPOTHESIS = {
    "hypothesis": "More study time improves student performance.",
    "motivation": "Study time may influence learning outcomes.",
    "supporting_evidence": ["smith2024"],
    "contradicting_evidence": [],
    "novelty_reasoning": "Tests the relationship on a new dataset.",
    "experiment": "Train a regression model using study time.",
    "success_metric": "R-squared greater than 0.5.",
    "falsification_condition": "No significant relationship is observed.",
}


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user, *, json_mode=False):
        self.calls.append({
            "system": system,
            "user": user,
            "json_mode": json_mode,
        })

        if not self.responses:
            raise RuntimeError("No scripted responses left")

        return json.dumps(self.responses.pop(0))


def sample_state():
    return {
        "research_question": "How does study time affect student performance?",
        "literature_summary": "Study time may influence student performance.",
        "citations": {
            "smith2024": {
                "title": "Student Performance Research"
            }
        },
    }


def test_hypothesis_generator_accepts_and_ranks_valid_hypotheses():
    stronger = {
        **VALID_HYPOTHESIS,
        "supporting_evidence": ["smith2024", "smith2024"],
    }

    weaker = {
        **VALID_HYPOTHESIS,
        "hypothesis": "Study time has an effect on performance.",
        "supporting_evidence": ["smith2024"],
    }

    llm = FakeLLM([
        {"hypotheses": [weaker, stronger]}
    ])

    generator = HypothesisGenerator(llm=llm)
    result = generator(sample_state())

    assert len(result["hypotheses"]) == 2
    assert result["selected_hypothesis"]["supporting_evidence"] == [
        "smith2024", "smith2024"
    ]
    assert result["hypotheses"][0]["novelty_note"]
    assert llm.calls[0]["json_mode"] is True


def test_hypothesis_generator_rejects_unknown_citation_and_retries():
    invalid = {
        **VALID_HYPOTHESIS,
        "supporting_evidence": ["unknown2024"],
    }

    llm = FakeLLM([
        {"hypotheses": [invalid]},
        {"hypotheses": [VALID_HYPOTHESIS]},
    ])

    generator = HypothesisGenerator(llm=llm, max_retries=1)
    result = generator(sample_state())

    assert len(llm.calls) == 2
    assert "previous answer was rejected" in llm.calls[1]["user"]
    assert result["selected_hypothesis"]["supporting_evidence"] == [
        "smith2024"
    ]


def test_hypothesis_generator_stops_after_max_retries():
    invalid = {
        **VALID_HYPOTHESIS,
        "supporting_evidence": ["unknown2024"],
    }

    llm = FakeLLM([
        {"hypotheses": [invalid]},
        {"hypotheses": [invalid]},
        {"hypotheses": [invalid]},
    ])

    generator = HypothesisGenerator(llm=llm, max_retries=2)
    result = generator(sample_state())

    assert len(llm.calls) == 3
    assert result["failure_state"] == FailureState.HYPOTHESIS_FAILED.value
    assert "no valid grounded hypothesis" in result["error"]


def test_hypothesis_generator_handles_llm_failure():
    class FailingLLM:
        def complete(self, system, user, *, json_mode=False):
            raise RuntimeError("LLM connection failed")

    generator = HypothesisGenerator(llm=FailingLLM())
    result = generator(sample_state())

    assert result["failure_state"] == FailureState.HYPOTHESIS_FAILED.value
    assert "LLM connection failed" in result["error"]
