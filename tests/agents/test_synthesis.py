
from src.agents.synthesis import SynthesisAgent
from src.schemas.state import FailureState


class FailingLLM:
    def complete(self, system_prompt, user_prompt):
        raise RuntimeError("LLM connection failed")


def test_synthesis_agent_maps_llm_error_to_failure_state():
    agent = SynthesisAgent(llm=FailingLLM())

    result = agent({
        "research_question": "Predict student performance"
    })

    assert result["failure_state"] == FailureState.SYNTHESIS_FAILED.value
    assert "LLM connection failed" in result["error"]
