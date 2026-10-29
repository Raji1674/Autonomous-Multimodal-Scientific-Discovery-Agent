
from agents.graph import build_graph


def main():
    research_question = input(
        "Enter your scientific research question: "
    ).strip()

    if not research_question:
        print("Please enter a research question.")
        return

    graph = build_graph()

    initial_state = {
        "research_question": research_question,
        "literature_findings": [],
        "experiment_results": {
            "status": "not_run",
            "message": "No experiment has been executed yet.",
        },
        "approved": False,
        "revision_count": 0,
        "errors": [],
    }

    print("\nStarting the research workflow...\n")

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("SCIENTIFIC RESEARCH REPORT")
    print("=" * 60)
    print(result.get("final_report", "No report was generated."))


if __name__ == "__main__":
    main()
