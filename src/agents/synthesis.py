
"""Scientific Synthesis Agent: generate a research report from workflow evidence."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.schemas.state import FailureState, ResearchState
from src.experiments.human_paper_benchmark import (
    evaluate_human_paper_benchmark,
)


SYNTHESIS_SYSTEM = """
You are a scientific report writer. Generate a report using ONLY
the supplied workflow evidence.

STRICT FACTUAL RULES:

1. Never invent algorithms, models, datasets, features, metrics,
   validation methods, results, citations, or statistical values.

2. For CSV classification experiments, the actual implementation
   compares:
   - Baseline: RandomForestClassifier with n_estimators=100
   - Treatment: RandomForestClassifier with n_estimators=200
   - Evaluation metric: Accuracy
   - Repeated random seeds: Use the exact number in the evidence.
   - Train/test split: Use only the exact values in the evidence.

3. Do NOT claim that SVM, XGBoost, GCN, neural networks, PCA,
   cross-validation, or any other method was executed unless
   explicitly confirmed by the workflow evidence.

4. The selected hypothesis is the hypothesis under investigation.
   Do not replace it with another hypothesis.
   If the experiment did not directly test the selected hypothesis,
   clearly state that it was not directly tested.

5. Use the exact dataset name, target column, row count,
   feature count, and validation findings from the evidence.
   Never describe a sample dataset as a real clinical dataset
   unless the evidence confirms it.

6. Report baseline and treatment results exactly as supplied.
   Do not calculate, estimate, or invent missing values.

7. Statistical verdict:
   - accept: predefined criteria met in the expected direction.
   - reject: significant evidence in the opposite direction.
   - inconclusive: insufficient evidence.
   Never describe inconclusive as reject.

8. Clearly separate literature findings, proposed hypotheses,
   completed experiments, and future work.

9. If information is missing, write "Not available in the
   workflow evidence." Do not fill gaps with assumptions.

Write these sections:
1. Abstract
2. Research Question and Motivation
3. Literature Review
4. Selected Hypothesis
5. Actual Methodology
6. Dataset and Validation
7. Experimental Results
8. Statistical Analysis
9. Limitations
10. Conclusion and Future Work

Accuracy is more important than making the report sound impressive.
"""


class SynthesisAgent(BaseAgent):
    name = "synthesis"
    failure_on_error = FailureState.SYNTHESIS_FAILED

    def __init__(self, llm: LLMClient):
        super().__init__()
        self.llm = llm

    def run(self, state: ResearchState) -> dict:
        """Generate a report from literature, experiment and critic evidence."""

        experiment_results = state.get("experiment_results") or {}

        # CSV classification experiments store their critic output
        # inside experiment_results["critic"].
        csv_critic_results = (
            experiment_results.get("critic")
            if isinstance(experiment_results, dict)
            else None
        )

        human_paper_benchmark = evaluate_human_paper_benchmark(state)

        evidence = {
            "research_question": state.get("research_question"),
            "literature": [],
            "literature_summary": state.get("literature_summary"),
            "citations": state.get("citations", {}),
            "hypotheses": [],
            "selected_hypothesis": state.get("selected_hypothesis"),
            "experiment_plan": state.get("experiment_plan"),
            "detected_task": state.get("detected_task"),
            "task_detection": state.get("task_detection"),
            "selected_dataset": state.get("selected_dataset"),
            "dataset_validation": state.get("dataset_validation"),
            "dataset_metadata": state.get("dataset_metadata"),
            "experiment_results": experiment_results,
            "experiment_protocol": experiment_results.get(
                "experiment_protocol"
            ),
            "statistical_results": state.get("statistical_results"),
            "critic_feedback": state.get("critic_feedback"),
            "csv_critic_results": csv_critic_results,
            "human_paper_benchmark": human_paper_benchmark,
            "verdict": state.get("verdict"),
            "failure_state": state.get("failure_state"),
            "error": state.get("error"),
        }

        # Preserve the complete evidence for both LLM calls.
        evidence_text = json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        report_prompt = (
            "Prepare a structured scientific research report "
            "using the following workflow evidence.\n\n"
            f"{evidence_text}"
        )

        report = self.llm.complete(
            SYNTHESIS_SYSTEM,
            report_prompt,
        )

        if not isinstance(report, str) or not report.strip():
            return {
                "failure_state": FailureState.SYNTHESIS_FAILED.value,
                "error": "The synthesis model returned an empty report.",
            }

        # Include the original research question and critic evidence
        # in the second call as well. This preserves traceability.
        abstract_prompt = f"""
        Create a standalone LaTeX research abstract using ONLY
        the supplied scientific report and workflow evidence.

        Requirements:
        - Use \\documentclass{{article}}
        - Include necessary packages.
        - Include \\begin{{abstract}} and \\end{{abstract}}.
        - Include the research objective, actual methodology,
          dataset, verified results and statistical conclusion.
        - Do not invent any values or claims.
        - Return only valid LaTeX source.
        - Do not use Markdown code fences.

        RESEARCH QUESTION:
        {evidence.get("research_question")}

        CSV_CRITIC_RESULTS:
        {json.dumps(csv_critic_results, ensure_ascii=False, default=str)}

        WORKFLOW EVIDENCE:
        {evidence_text}

        SCIENTIFIC REPORT:
        {report}
        """

        latex_abstract = self.llm.complete(
            "You are a scientific LaTeX document writer.",
            abstract_prompt,
        )

        if (
            not isinstance(latex_abstract, str)
            or not latex_abstract.strip()
        ):
            return {
                "failure_state": FailureState.SYNTHESIS_FAILED.value,
                "error": "LaTeX abstract generation failed.",
            }

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        latex_path = output_dir / "research_abstract.tex"

        latex_path.write_text(
            latex_abstract.strip() + "\n",
            encoding="utf-8",
        )

        return {
            "final_report": report.strip(),
            "human_paper_benchmark": human_paper_benchmark,
            "latex_abstract": latex_abstract.strip(),
            "latex_abstract_path": str(latex_path),
        }
