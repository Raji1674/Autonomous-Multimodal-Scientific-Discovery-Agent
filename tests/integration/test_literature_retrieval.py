
from src.agents.literature_reviewer import LiteratureReviewer
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import FakeEmbedder
from src.schemas.paper import Chunk


class FakeLLM:
    def __init__(self):
        self.last_user_prompt = None

    def complete(self, system_prompt, user_prompt):
        self.last_user_prompt = user_prompt
        return "Summary based on retrieved ChromaDB evidence."


def test_literature_reviewer_retrieves_real_chroma_chunks(tmp_path):
    # Create a real, temporary ChromaDB vector store.
    store = ChromaStore(
        path=tmp_path / "chroma",
        embedder=FakeEmbedder(),
    )

    # Insert a research paper chunk into ChromaDB.
    chunk = Chunk(
        chunk_id="paper-001#0",
        paper_id="paper-001",
        text=(
            "Research found that regular study time "
            "is associated with improved student performance."
        ),
        metadata={
            "citation_key": "smith2024",
            "title": "Student Performance Research",
            "authors": "Smith et al.",
            "year": 2024,
            "source": "arxiv",
            "publication_status": "preprint",
        },
    )

    store.upsert([chunk])

    # Connect the real Literature Reviewer to the real ChromaDB store.
    llm = FakeLLM()
    reviewer = LiteratureReviewer(
        store=store,
        llm=llm,
        top_k=5,
    )

    result = reviewer({
        "research_question": "How does regular study time affect student performance?"
    })

    # Verify that evidence was retrieved from ChromaDB.
    assert len(result["literature"]) > 0
    assert result["literature_summary"] == (
        "Summary based on retrieved ChromaDB evidence."
    )

    # Verify citation information.
    assert "smith2024" in result["citations"]
    assert result["citations"]["smith2024"]["paper_id"] == "paper-001"

    # Verify retrieved evidence reached the LLM prompt.
    assert "regular study time" in llm.last_user_prompt
    assert "[smith2024]" in llm.last_user_prompt
