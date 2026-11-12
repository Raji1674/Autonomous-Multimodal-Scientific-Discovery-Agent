
import json

from src.agents.hypothesis_generator import HypothesisGenerator


class CaptureLLM:
    def __init__(self):
        self.system = None
        self.prompt = None
        self.json_mode = None

    def complete(self, system, prompt, json_mode=False):
        self.system = system
        self.prompt = prompt
        self.json_mode = json_mode

        response = {
            "hypotheses": [
                {
                    "hypothesis": "Regularization improves classification.",
                    "motivation": "Test the effect of regularization.",
                    "supporting_evidence": ["paper_1"],
                    "contradicting_evidence": [],
                    "novelty_reasoning": "Builds on previous findings.",
                    "experiment": "Compare regularized and unregularized models.",
                    "success_metric": "Accuracy",
                    "falsification_condition": "No improvement in accuracy.",
                }
            ]
        }

        return json.dumps(response)


def make_state(memory_context):
    return {
        "research_question": "Does regularization improve classification?",
        "literature_summary": "Current literature discusses classification.",
        "citations": {
            "paper_1": {
                "title": "Classification research",
            }
        },
        "memory_context": memory_context,
    }


def test_previous_research_memory_is_included_in_llm_prompt():
    llm = CaptureLLM()
    agent = HypothesisGenerator(llm)

    state = make_state([
        {
            "research_question": "Previous classification study",
            "hypothesis": "Regularization may improve accuracy.",
            "findings": {"accuracy": 0.91},
            "verdict": "inconclusive",
            "dataset": "Previous dataset",
        }
    ])

    result = agent.run(state)

    assert result.get("selected_hypothesis") is not None
    assert "Previous classification study" in llm.prompt
    assert "Regularization may improve accuracy." in llm.prompt
    assert "inconclusive" in llm.prompt
    assert "Previous research memory" in llm.prompt
    assert "not citable" in llm.prompt.lower()
    assert llm.json_mode is True


def test_empty_memory_is_handled():
    llm = CaptureLLM()
    agent = HypothesisGenerator(llm)

    state = make_state([])

    result = agent.run(state)

    assert result.get("selected_hypothesis") is not None
    assert "No relevant previous research found." in llm.prompt
