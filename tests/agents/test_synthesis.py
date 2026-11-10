
from src.agents.synthesis import SynthesisAgent
from src.schemas.state import FailureState


class FakeLLM:
    def __init__(self, response="Research report generated"):
        self.response = response
        self.last_prompt = None

    def complete(self, system, user, json_mode=False):
        self.last_prompt = user
        return self.response


def test_synthesis_generates_final_report():
    llm = FakeLLM("Final scientific research report")

    agent = SynthesisAgent(llm)

    result = agent.run({
        "research_question": "Cardiovascular disease prediction",
        "literature_summary": "Previous research evidence",
        "hypotheses": [{"statement": "Test hypothesis"}],
        "experiment_results": {
            "result_label": "executed"
        },
        "verdict": "accept",
    })

    assert result["final_report"] == "Final scientific research report"
    assert "Cardiovascular disease prediction" in llm.last_prompt


def test_synthesis_includes_csv_critic_evidence():
    llm = FakeLLM()
    agent = SynthesisAgent(llm)

    agent.run({
        "research_question": "CSV experiment",
        "experiment_results": {
            "success": True,
            "critic": {
                "verdict": "accept",
                "p_value": 0.01,
            },
        },
    })

    assert "csv_critic_results" in llm.last_prompt
    assert "p_value" in llm.last_prompt
    assert "0.01" in llm.last_prompt


def test_synthesis_rejects_empty_llm_response():
    llm = FakeLLM("")

    agent = SynthesisAgent(llm)

    result = agent({
        "research_question": "Test question"
    })

    assert result["failure_state"] == FailureState.SYNTHESIS_FAILED.value
