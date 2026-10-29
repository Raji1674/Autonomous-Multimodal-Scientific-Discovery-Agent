
import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from agents.prompts import (
    CODE_PROMPT,
    CRITIC_PROMPT,
    HYPOTHESIS_PROMPT,
    LITERATURE_PROMPT,
    SYNTHESIS_PROMPT,
)
from agents.state import ResearchState


load_dotenv(override=True)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def create_model() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to your local .env file."
        )

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0,
    )


def format_value(value: Any) -> str:
    """Format state values for inclusion in a prompt."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)

def response_text(response: Any) -> str:
    """Extract readable text from a Gemini response."""
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()

    return str(content)


def build_graph():
    """Build the initial five-agent research workflow."""
    model = create_model()

    
    def literature_agent(state: ResearchState) -> dict:
        """Summarize supplied literature findings using Gemini."""
        findings = state.get("literature_findings", [])

        prompt = LITERATURE_PROMPT.format(
            research_question=state["research_question"]
        )

        response = model.invoke(
            prompt
            + "\n\nSupplied literature findings:\n"
            + format_value(findings)
        )

        return {
            "literature_summary": response_text(response),
        }
   
    def hypothesis_agent(state: ResearchState) -> dict:
        """Generate a testable hypothesis from the literature summary."""
        prompt = HYPOTHESIS_PROMPT.format(
            research_question=state["research_question"],
            literature_findings=state.get(
                "literature_summary",
                "No literature summary is available yet.",
            ),
        )

        response = model.invoke(prompt)
        return {"hypothesis": response_text(response)}


    def code_agent(state: ResearchState) -> dict:
        prompt = CODE_PROMPT.format(
            research_question=state["research_question"],
            hypothesis=state.get("hypothesis", ""),
        )
        response = model.invoke(prompt)
        return {
            "experiment_plan": {"plan": response_text(response)},
            "experiment_results": {
                "status": "not_run",
                "message": (
                    "This workflow creates an experiment plan only. "
                    "Execution requires Member 4's secure sandbox and "
                    "Member 3's experiment implementation."
                ),
            },
        }

    def critic_agent(state: ResearchState) -> dict:
        prompt = CRITIC_PROMPT.format(
            research_question=state["research_question"],
            hypothesis=state.get("hypothesis", ""),
            experiment_plan=format_value(
                state.get("experiment_plan", {})
            ),
            experiment_results=format_value(
                state.get("experiment_results", {})
            ),
        )
        response = model.invoke(prompt)
        return {
            "critique": response_text(response),
            "approved": False,
        }

    def synthesis_agent(state: ResearchState) -> dict:
        prompt = SYNTHESIS_PROMPT.format(
            research_question=state["research_question"],
            literature_findings=format_value(
                state.get("literature_findings", [])
            ),
            hypothesis=state.get("hypothesis", ""),
            experiment_plan=format_value(
                state.get("experiment_plan", {})
            ),
            experiment_results=format_value(
                state.get("experiment_results", {})
            ),
            critique=state.get("critique", ""),
            approved=state.get("approved", False),
        )
        response = model.invoke(prompt)
        return {"final_report": response_text(response)}
    workflow = StateGraph(ResearchState)

    workflow.add_node("literature", literature_agent)
    workflow.add_node("hypothesis", hypothesis_agent)
    workflow.add_node("experiment", code_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("synthesis", synthesis_agent)

    workflow.add_edge(START, "literature")
    workflow.add_edge("literature", "hypothesis")
    workflow.add_edge("hypothesis", "experiment")
    workflow.add_edge("experiment", "critic")
    workflow.add_edge("critic", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()
