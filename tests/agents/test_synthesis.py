
from src.agents.synthesis import (
    SynthesisAgent,
    validate_latex,
    validate_report,
)
from src.schemas.state import FailureState


VALID_REPORT = """
# Abstract
Research summary.

# Research Question and Motivation
Research question.

# Literature Review
Literature evidence.

# Selected Hypothesis
Selected hypothesis.

# Actual Methodology
Actual methodology.

# Dataset and Validation
Dataset validation.

# Experimental Results
Experiment results.

# Statistical Analysis
Statistical results.

# Human-Paper Benchmark
Contextual comparison and limitations.

# Limitations
Study limitations.

# Conclusion and Future Work
Conclusion.
"""

VALID_LATEX = r"""
\documentclass{article}
\begin{document}
\begin{abstract}
Scientific abstract.
\end{abstract}
\end{document}
"""


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, system, user, json_mode=False):
        self.prompts.append({
            "system": system,
            "user": user,
        })

        if not self.responses:
            return ""

        return self.responses.pop(0)


def sample_state():
    return {
        "research_question": "Does stronger regularization improve accuracy?",
        "literature_summary": "Previous research findings.",
        "citations": {
            "paper_1": "Example literature reference"
        },
        "hypotheses": [
            {"statement": "Stronger regularization improves accuracy"}
        ],
        "selected_hypothesis": (
            "Stronger regularization improves accuracy"
        ),
        "experiment_plan": {
            "model": "logreg",
            "dataset": "breast_cancer",
        },
        "experiment_results": {
            "result_label": "executed",
            "baseline": [0.95, 0.96],
            "treatment": [0.94, 0.97],
        },
        "statistical_results": {
            "p_adjusted": 0.42,
            "mean_diff": 0.0,
        },
        "critic_feedback": {
            "verdict": "inconclusive",
        },
        "verdict": "inconclusive",
    }


def test_synthesis_generates_report_and_latex(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    llm = FakeLLM([VALID_REPORT, VALID_LATEX])
    agent = SynthesisAgent(llm)

    result = agent(sample_state())

    assert result["final_report"] == VALID_REPORT.strip()
    assert result["latex_abstract"] == VALID_LATEX.strip()
    assert result["human_paper_benchmark"]["status"] == "not_comparable"
    assert len(llm.prompts) == 2
    assert (tmp_path / "outputs" / "research_abstract.tex").exists()


def test_synthesis_passes_workflow_evidence_to_both_calls(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    llm = FakeLLM([VALID_REPORT, VALID_LATEX])
    agent = SynthesisAgent(llm)

    agent(sample_state())

    for call in llm.prompts:
        prompt = call["user"]

        assert "Does stronger regularization improve accuracy?" in prompt
        assert "p_adjusted" in prompt
        assert "0.42" in prompt
        assert "paper_1" in prompt
        assert "inconclusive" in prompt


def test_synthesis_includes_csv_critic_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    llm = FakeLLM([VALID_REPORT, VALID_LATEX])
    agent = SynthesisAgent(llm)

    state = sample_state()
    state["experiment_results"]["critic"] = {
        "verdict": "accept",
        "p_value": 0.01,
    }

    agent(state)

    assert "csv_critic_results" in llm.prompts[0]["user"]
    assert "0.01" in llm.prompts[0]["user"]


def test_synthesis_rejects_empty_report():
    llm = FakeLLM(["", VALID_LATEX])
    result = SynthesisAgent(llm)(sample_state())

    assert result["failure_state"] == FailureState.SYNTHESIS_FAILED.value
    assert "empty report" in result["error"].lower()


def test_synthesis_rejects_missing_report_sections():
    llm = FakeLLM(["# Abstract\nOnly an abstract.", VALID_LATEX])
    result = SynthesisAgent(llm)(sample_state())

    assert result["failure_state"] == FailureState.SYNTHESIS_FAILED.value
    assert "missing sections" in result["error"].lower()


def test_synthesis_rejects_empty_latex(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    llm = FakeLLM([VALID_REPORT, ""])
    result = SynthesisAgent(llm)(sample_state())

    assert result["failure_state"] == FailureState.SYNTHESIS_FAILED.value
    assert "latex abstract generation failed" in result["error"].lower()


def test_synthesis_rejects_malformed_latex(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    llm = FakeLLM([
        VALID_REPORT,
        r"\begin{abstract}Missing document",
    ])

    result = SynthesisAgent(llm)(sample_state())

    assert result["failure_state"] == FailureState.SYNTHESIS_FAILED.value
    assert "invalid structure" in result["error"].lower()


def test_report_validator_detects_missing_sections():
    missing = validate_report("# Abstract\nOnly an abstract.")

    assert "Literature Review" in missing
    assert "Human-Paper Benchmark" in missing


def test_latex_validator():
    assert validate_latex(VALID_LATEX)
    assert not validate_latex(
        r"\begin{abstract}No complete document"
    )
