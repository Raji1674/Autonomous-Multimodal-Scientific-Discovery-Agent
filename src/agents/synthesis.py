
"""Evidence-grounded research synthesis with strict output validation."""

from __future__ import annotations

import json
import logging
import re
import statistics

from pathlib import Path
from typing import Any

from src.agents.base import BaseAgent
from src.llm.client import LLMError
from src.schemas.state import FailureState


logger = logging.getLogger(__name__)


REQUIRED_REPORT_SECTIONS = [
    "Abstract",
    "Research Question and Motivation",
    "Literature Review",
    "Selected Hypothesis",
    "Actual Methodology",
    "Dataset and Validation",
    "Experimental Results",
    "Statistical Analysis",
    "Human-Paper Benchmark",
    "Limitations",
    "Conclusion and Future Work",
]


def validate_latex(latex: str) -> bool:
    """Validate the basic LaTeX document structure."""

    if not isinstance(latex, str) or not latex.strip():
        return False

    required = [
        r"\documentclass",
        r"\begin{document}",
        r"\end{document}",
        r"\begin{abstract}",
        r"\end{abstract}",
    ]

    return all(token in latex for token in required)


def validate_report(report: str) -> list[str]:
    """Return required Markdown headings missing from a report."""

    if not isinstance(report, str) or not report.strip():
        return REQUIRED_REPORT_SECTIONS.copy()

    missing = []

    for section in REQUIRED_REPORT_SECTIONS:
        pattern = (
            rf"(?im)^\s*#{{1,6}}\s+"
            rf"{re.escape(section)}\s*$"
        )

        if not re.search(pattern, report):
            missing.append(section)

    return missing


class SynthesisAgent(BaseAgent):
    """Creates an evidence-grounded scientific research report."""

    REQUIRED_REPORT_SECTIONS = REQUIRED_REPORT_SECTIONS

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def validate_report(self, report: str) -> tuple[bool, list[str]]:
        missing = validate_report(report)
        return not missing, missing

    def validate_latex(self, latex: str) -> bool:
        return validate_latex(latex)

    def _hypothesis_text(self, hypothesis: Any) -> str:
        if isinstance(hypothesis, dict):
            return str(
                hypothesis.get("hypothesis")
                or hypothesis.get("statement")
                or json.dumps(
                    hypothesis,
                    ensure_ascii=False,
                    default=str,
                )
            )

        if isinstance(hypothesis, str):
            return hypothesis.strip()

        return "No selected hypothesis was supplied."

    def _compact_scores(self, values: Any) -> dict:
        if not isinstance(values, (list, tuple)) or not values:
            return {}

        numeric = []

        for value in values:
            try:
                number = float(value)

                if number == number and abs(number) != float("inf"):
                    numeric.append(number)

            except (TypeError, ValueError):
                continue

        if not numeric:
            return {}

        return {
            "count": len(numeric),
            "mean": round(statistics.mean(numeric), 6),
            "sd": (
                round(statistics.stdev(numeric), 6)
                if len(numeric) > 1
                else 0.0
            ),
            "minimum": round(min(numeric), 6),
            "maximum": round(max(numeric), 6),
        }

    def _compact_results(self, results: Any) -> dict:
        if isinstance(results, (list, tuple)):
            return {
                "items": [
                    self._compact_results(item)
                    if isinstance(item, dict)
                    else item
                    for item in results[:20]
                    if isinstance(
                        item,
                        (dict, list, tuple, str, int, float, bool),
                    )
                ]
            }

        if not isinstance(results, dict):
            return {"results": str(results)[:1000]}

        compact = {}

        for key, value in results.items():

            if key in {
                "baseline",
                "treatment",
                "baseline_train",
                "treatment_train",
                "scores",
            }:
                compact[key] = self._compact_scores(value)

            elif key in {
                "predictions",
                "y_true",
                "y_pred",
                "raw_data",
                "logs",
                "generated_code",
                "source_code",
            }:
                continue

            elif isinstance(value, dict):
                compact[key] = self._compact_results(value)

            elif isinstance(value, (list, tuple)):
                compact[key] = [
                    self._compact_results(item)
                    if isinstance(item, dict)
                    else item
                    for item in value[:20]
                    if isinstance(
                        item,
                        (dict, str, int, float, bool),
                    )
                ]

            elif (
                isinstance(value, (str, int, float, bool))
                or value is None
            ):
                compact[key] = value

        return compact

    def _build_evidence(self, state: dict) -> dict:
        results = state.get("experiment_results") or {}

        if not isinstance(results, dict):
            results = {"results": results}

        hypothesis = (
            state.get("selected_hypothesis")
            or state.get("hypothesis")
        )

        plan = state.get("experiment_plan") or {}

        statistical = (
            state.get("statistical_results")
            or results.get("statistical_results")
            or {}
        )

        critic = (
            state.get("critic_feedback")
            or results.get("critic")
            or {}
        )

        csv_critic = (
            state.get("csv_critic_results")
            or results.get("csv_critic_results")
            or results.get("critic")
            or critic
            or {}
        )

        benchmark = (
            state.get("human_paper_benchmark")
            or results.get("human_paper_benchmark")
            or results.get("benchmark")
            or {
                "status": "not_comparable",
                "reason": (
                    "No comparable human-paper benchmark was provided."
                ),
            }
        )

        literature = (
            state.get("literature_summary")
            or state.get("literature_review")
            or ""
        )

        if not isinstance(literature, str):
            literature = str(literature)

        validation = state.get("dataset_validation") or {}
        metadata = state.get("dataset_metadata") or {}

        return {
            "research_question": str(
                state.get("research_question")
                or state.get("question")
                or ""
            )[:500],

            "selected_hypothesis": self._hypothesis_text(hypothesis),

            "hypothesis_details": (
                {
                    "motivation": hypothesis.get("motivation"),
                    "experiment": hypothesis.get("experiment"),
                    "success_metric": hypothesis.get("success_metric"),
                    "falsification_condition": hypothesis.get(
                        "falsification_condition"
                    ),
                }
                if isinstance(hypothesis, dict)
                else {}
            ),

            "hypotheses": state.get("hypotheses") or [],
            "citations": state.get("citations") or {},
            "verdict": state.get("verdict"),
            "literature_summary": literature[:2500],
            "experiment_plan": self._compact_results(plan),
            "dataset_validation": self._compact_results(validation),
            "dataset_metadata": self._compact_results(metadata),
            "experiment_results": self._compact_results(results),
            "statistical_results": self._compact_results(statistical),
            "critic_feedback": self._compact_results(critic),
            "csv_critic_results": self._compact_results(csv_critic),
            "human_paper_benchmark": self._compact_results(benchmark),
        }

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.llm.complete(
            system=system_prompt,
            user=user_prompt,
        )

        if response is None:
            return ""

        return str(response).strip()

    def build_fallback_report(
        self,
        research_question: str,
        hypothesis: Any,
        literature_summary: Any,
        experiment_results: dict,
        statistical_results: Any = None,
        critic_feedback: Any = None,
        experiment_plan: Any = None,
        dataset_validation: Any = None,
        human_paper_benchmark: Any = None,
    ) -> str:
        """Build a deterministic report using recorded evidence."""

        evidence = {
            "experiment_plan": self._compact_results(
                experiment_plan or {}
            ),
            "dataset_validation": self._compact_results(
                dataset_validation or {}
            ),
            "experiment_results": self._compact_results(
                experiment_results or {}
            ),
            "statistical_results": self._compact_results(
                statistical_results or {}
            ),
            "critic_feedback": self._compact_results(
                critic_feedback or {}
            ),
            "human_paper_benchmark": self._compact_results(
                human_paper_benchmark or {
                    "status": "not_comparable",
                    "reason": (
                        "No comparable human-paper benchmark was provided."
                    ),
                }
            ),
        }

        literature_text = (
            str(literature_summary)[:2500]
            if literature_summary
            else "No literature summary was supplied."
        )

        return f"""
# Abstract

This report presents the research question, selected hypothesis,
methodology, dataset evidence, experimental results, and statistical
evaluation available from the research workflow. The report uses
recorded evidence and does not invent missing findings.

# Research Question and Motivation

{research_question or 'No research question was supplied.'}

# Literature Review

{literature_text}

# Selected Hypothesis

{self._hypothesis_text(hypothesis)}

# Actual Methodology

The experiment configuration and methodology recorded by the workflow
are presented below.

{json.dumps(evidence['experiment_plan'], indent=2, default=str)}

# Dataset and Validation

{json.dumps(evidence['dataset_validation'], indent=2, default=str)}

# Experimental Results

{json.dumps(evidence['experiment_results'], indent=2, default=str)}

# Statistical Analysis

{json.dumps(evidence['statistical_results'], indent=2, default=str)}

# Human-Paper Benchmark

{json.dumps(evidence['human_paper_benchmark'], indent=2, default=str)}

# Limitations

The findings are limited to the available dataset, experiment
configuration, and recorded results. A comparable human-paper
benchmark may not be available.

# Conclusion and Future Work

The conclusion is based on the recorded experimental and statistical
evidence. Further validation using additional datasets is recommended.
"""

    def _make_latex(self, report: str) -> str:
        """Create a minimal valid LaTeX document from the report abstract."""

        abstract_match = re.search(
            r"(?ims)^#\s+Abstract\s+(.*?)(?=^\s*#\s+|\Z)",
            report,
        )

        abstract = (
            abstract_match.group(1).strip()
            if abstract_match
            else report[:2000]
        )

        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }

        for old, new in replacements.items():
            abstract = abstract.replace(old, new)

        return (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\begin{abstract}\n"
            + abstract
            + "\n\\end{abstract}\n"
            "\\end{document}\n"
        )

    def failure_result(
        self,
        message: str,
        missing_sections: list[str] | None = None,
    ) -> dict:

        result = {
            "status": "failed",
            "failure_state": FailureState.SYNTHESIS_FAILED.value,
            "error": message,
            "final_report": "",
            "latex_abstract": "",
            "latex_abstract_path": None,
        }

        if missing_sections is not None:
            result["missing_sections"] = missing_sections

        return result

    def _fallback_from_evidence(self, evidence: dict, hypothesis: Any) -> str:
        """Build a complete report without another LLM request."""

        return self.build_fallback_report(
            research_question=evidence["research_question"],
            hypothesis=hypothesis,
            literature_summary=evidence["literature_summary"],
            experiment_results=evidence["experiment_results"],
            statistical_results=evidence["statistical_results"],
            critic_feedback=evidence["critic_feedback"],
            experiment_plan=evidence["experiment_plan"],
            dataset_validation=evidence["dataset_validation"],
            human_paper_benchmark=evidence["human_paper_benchmark"],
        ).strip()

    def run(self, state: dict) -> dict:
        """Generate and validate the Markdown report and LaTeX output."""

        if not isinstance(state, dict):
            return self.failure_result(
                "Synthesis received an invalid workflow state."
            )

        evidence = self._build_evidence(state)

        research_question = evidence["research_question"]

        hypothesis = (
            state.get("selected_hypothesis")
            or state.get("hypothesis")
        )

        system_prompt = """
You are an evidence-grounded scientific research synthesis agent.

Create a complete research report in Markdown.

Rules:
1. Use only the evidence provided.
2. Never invent experimental results, metrics, citations, or findings.
3. Clearly distinguish measured results from interpretation.
4. Mention limitations and unavailable benchmarks honestly.
5. Use all required headings exactly as supplied.
6. Do not claim grid search or hyperparameter tuning unless
   the supplied experiment evidence confirms it.
7. Return only the Markdown report.

Required headings:

# Abstract
# Research Question and Motivation
# Literature Review
# Selected Hypothesis
# Actual Methodology
# Dataset and Validation
# Experimental Results
# Statistical Analysis
# Human-Paper Benchmark
# Limitations
# Conclusion and Future Work
"""

        evidence_json = json.dumps(
            evidence,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        user_prompt = (
            "Prepare the research report using this workflow evidence:\n\n"
            + evidence_json
        )

        # --------------------------------------------------
        # 1. GENERATE REPORT
        # --------------------------------------------------

        try:
            report = self._complete(
                system_prompt,
                user_prompt,
            )

        except LLMError as exc:
            logger.warning(
                "LLM report generation failed; using evidence fallback: %s",
                exc,
            )

            report = self._fallback_from_evidence(
                evidence,
                hypothesis,
            )

        if not report.strip():
            return self.failure_result(
                "Empty report: LLM returned an empty research report."
            )

        # --------------------------------------------------
        # 2. VALIDATE AND REPAIR REPORT
        # --------------------------------------------------

        missing = validate_report(report)

        if missing:

            # Keep repair request small to avoid Groq TPM errors.
            repair_prompt = f"""
The previous report is incomplete.

Missing sections:
{json.dumps(missing)}

Previous report:
{report[:10000]}

Return the complete corrected Markdown report.
Include all required headings exactly.
Use only evidence already present in the previous report.
Do not invent results, metrics, or citations.
"""

            try:
                repaired = self._complete(
                    system_prompt,
                    repair_prompt,
                )

                if repaired.strip():
                    report = repaired

            except LLMError as exc:
                logger.warning(
                    "Report repair failed; using deterministic evidence report: %s",
                    exc,
                )

                report = self._fallback_from_evidence(
                    evidence,
                    hypothesis,
                )

            missing = validate_report(report)

        if missing:
            return self.failure_result(
                "Report validation failed: missing sections in generated report.",
                missing_sections=missing,
            )

        # --------------------------------------------------
        # 3. GENERATE LATEX USING COMPACT CONTEXT
        # --------------------------------------------------

        latex_system_prompt = r"""
You are a scientific LaTeX document generator.

Convert the supplied research report into a valid LaTeX document.

Requirements:
1. Include \documentclass{article}.
2. Include \begin{document}.
3. Include \begin{abstract} and \end{abstract}.
4. Include \end{document}.
5. Escape LaTeX special characters correctly.
6. Return only LaTeX source code.
7. Do not add Markdown code fences.
"""

        # Do not send the complete evidence JSON again.
        # This avoids unnecessary token usage and Groq TPM errors.
        latex_context = {
            "research_question": evidence["research_question"],
            "citations": evidence["citations"],
            "verdict": evidence["verdict"],
            "statistical_results": evidence["statistical_results"],
        }

        latex_user_prompt = (
            "Convert this research report into a LaTeX document:\n\n"
            + report[:12000]
            + "\n\nCompact workflow context:\n"
            + json.dumps(
                latex_context,
                ensure_ascii=False,
                default=str,
            )
        )

        try:
            latex = self._complete(
                latex_system_prompt,
                latex_user_prompt,
            )

        except LLMError as exc:
            logger.warning(
                "LLM LaTeX generation failed; creating LaTeX locally: %s",
                exc,
            )

            latex = self._make_latex(report)

        if not latex.strip():
            return self.failure_result(
                "LaTeX abstract generation failed: LLM returned empty output."
            )

        if not validate_latex(latex):
            return self.failure_result(
                "LaTeX abstract generation failed: invalid structure."
            )

        # --------------------------------------------------
        # 4. SAVE VALIDATED LATEX
        # --------------------------------------------------

        try:
            output_dir = Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)

            latex_path = output_dir / "research_abstract.tex"

            latex_path.write_text(
                latex,
                encoding="utf-8",
            )

        except OSError as exc:
            logger.exception("Could not save LaTeX output.")

            return self.failure_result(
                f"Could not save LaTeX output: {exc}"
            )

        return {
            "status": "success",
            "final_report": report.strip(),
            "latex_abstract": latex,
            "latex_abstract_path": str(latex_path),
            "human_paper_benchmark": evidence[
                "human_paper_benchmark"
            ],
        }
