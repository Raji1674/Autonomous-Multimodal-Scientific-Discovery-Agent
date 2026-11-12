
from src.agents.literature_reviewer import LiteratureReviewer
from src.ingestion.pdf_indexer import index_pdf_content
from src.schemas.state import FailureState
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import FakeEmbedder


class FakeLLM:
    def __init__(self):
        self.user_prompt = None

    def complete(self, system, user):
        self.user_prompt = user
        return "Literature summary."


def test_uploaded_pdf_flows_through_rag_and_is_rejected_as_unverified(
    tmp_path,
):
    # 1. Create a real temporary ChromaDB store.
    store = ChromaStore(
        path=tmp_path / "pdf_rag_chroma",
        embedder=FakeEmbedder(dim=64),
    )

    # 2. Index extracted PDF content.
    pdf_result = {
        "pdf_name": "research_paper.pdf",
        "pages": [
            {
                "page_number": 3,
                "text": (
                    "Cardiovascular disease prediction using "
                    "machine learning and statistical analysis."
                ),
            }
        ],
    }

    index_report = index_pdf_content(pdf_result, store)

    assert index_report["success"] is True
    assert index_report["chunks_indexed"] > 0

    # 3. Confirm PDF chunks are actually stored.
    assert store.count() > 0

    # 4. Run the actual Literature Reviewer.
    llm = FakeLLM()
    reviewer = LiteratureReviewer(store, llm, top_k=2)

    result = reviewer.run({
        "research_question": (
            "Cardiovascular disease prediction using machine learning"
        )
    })

    # 5. Uploaded PDF has no verified publication status.
    # It must not become peer-reviewed evidence automatically.
    assert result["failure_state"] == (
        FailureState.RETRIEVAL_FAILED.value
    )
    assert result["literature"] == []
    assert result["citations"] == {}
    assert result["literature_filter"]["eligible_evidence"] == 0
    assert result["literature_filter"]["excluded_unverified"] > 0
    assert llm.user_prompt is None
