
from src.agents.literature_reviewer import LiteratureReviewer
from src.schemas.state import FailureState


class FakeChunk:
    def __init__(self, paper_id, text, metadata):
        self.paper_id = paper_id
        self.text = text
        self.metadata = metadata


class FakeHit:
    def __init__(self, chunk):
        self.chunk = chunk

    def model_dump(self):
        return {
            "paper_id": self.chunk.paper_id,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
        }


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.last_query = None
        self.last_k = None
        self.last_where = None

    def query(self, query, k, where=None):
        self.last_query = query
        self.last_k = k
        self.last_where = where

        if where:
            return [
                hit for hit in self.hits
                if all(
                    hit.chunk.metadata.get(key) == value
                    for key, value in where.items()
                )
            ]

        return self.hits


class FakeLLM:
    def __init__(self, response="Evidence-based literature summary"):
        self.response = response
        self.last_system = None
        self.last_user = None

    def complete(self, system_prompt, user_prompt):
        self.last_system = system_prompt
        self.last_user = user_prompt
        return self.response


def sample_hit():
    return FakeHit(
        FakeChunk(
            paper_id="paper-001",
            text="Study found a relationship between study time and performance.",
            metadata={
                "citation_key": "smith2024",
                "title": "Student Performance Research",
                "authors": "Smith et al.",
                "year": 2024,
                "source": "arXiv",
                "publication_status": "preprint",
            },
        )
    )


def test_literature_reviewer_retrieves_and_summarizes_evidence():
    store = FakeStore([sample_hit()])
    llm = FakeLLM()
    reviewer = LiteratureReviewer(store=store, llm=llm, top_k=5)

    result = reviewer({
        "research_question": "How does study time affect student performance?"
    })

    assert store.last_query == "How does study time affect student performance?"
    assert store.last_k == 5
    assert result["literature_summary"] == "Evidence-based literature summary"
    assert "smith2024" in result["citations"]
    assert result["citations"]["smith2024"]["paper_id"] == "paper-001"
    assert len(result["literature"]) == 1
    assert "Use ONLY the provided sources" in llm.last_system
    assert "Study found a relationship" in llm.last_user


def test_literature_reviewer_fails_when_no_evidence_is_retrieved():
    reviewer = LiteratureReviewer(
        store=FakeStore([]),
        llm=FakeLLM(),
    )

    result = reviewer({
        "research_question": "Predict student performance"
    })

    assert result["failure_state"] == FailureState.RETRIEVAL_FAILED.value
    assert "no evidence retrieved" in result["error"]


def test_literature_reviewer_handles_llm_failure():
    class FailingLLM:
        def complete(self, system_prompt, user_prompt):
            raise RuntimeError("LLM connection failed")

    reviewer = LiteratureReviewer(
        store=FakeStore([sample_hit()]),
        llm=FailingLLM(),
    )

    result = reviewer({
        "research_question": "Predict student performance"
    })

    assert result["failure_state"] == FailureState.RETRIEVAL_FAILED.value
    assert "LLM connection failed" in result["error"]

def test_literature_reviewer_handles_duplicate_citation_keys():
    first = sample_hit()

    second = FakeHit(
        FakeChunk(
            paper_id="paper-002",
            text="Another study found a different relationship.",
            metadata={
                "citation_key": "smith2024",
                "title": "Another Student Performance Study",
                "authors": "Jones et al.",
                "year": 2024,
                "source": "arXiv",
                "publication_status": "preprint",
            },
        )
    )

    class FakeStore:
        def __init__(self, hits):
            self.hits = hits
            self.last_query = None
            self.last_k = None
            self.last_where = None

        def query(self, query, k, where=None):
            self.last_query = query
            self.last_k = k
            self.last_where = where

            if where:
                return [
                    hit for hit in self.hits
                    if all(
                        hit.chunk.metadata.get(key) == value
                        for key, value in where.items()
                    )
                ]

            return self.hits

def test_peer_reviewed_mode_excludes_preprints():
    preprint = sample_hit()

    peer_reviewed = FakeHit(
        FakeChunk(
            paper_id="paper-003",
            text="Peer-reviewed study found a significant relationship.",
            metadata={
                "citation_key": "jones2025",
                "title": "Published Student Performance Study",
                "authors": "Jones et al.",
                "year": 2025,
                "source": "PubMed",
                "publication_status": "peer_reviewed",
            },
        )
    )

    store = FakeStore([preprint, peer_reviewed])
    llm = FakeLLM()

    reviewer = LiteratureReviewer(
        store=store,
        llm=llm,
        top_k=5,
        peer_reviewed_only=True,
    )

    result = reviewer({
        "research_question": "How does study time affect student performance?"
    })

    assert store.last_where == {
        "publication_status": "peer_reviewed"
    }
    assert len(result["literature"]) == 1
    assert result["literature"][0]["paper_id"] == "paper-003"
    assert "jones2025" in result["citations"]
    assert "smith2024" not in result["citations"]
    assert "peer-reviewed sources" in llm.last_system


def test_peer_reviewed_mode_fails_when_only_preprints_exist():
    store = FakeStore([sample_hit()])
    llm = FakeLLM()

    reviewer = LiteratureReviewer(
        store=store,
        llm=llm,
        peer_reviewed_only=True,
    )

    result = reviewer({
        "research_question": "Predict student performance"
    })

    assert store.last_where == {
        "publication_status": "peer_reviewed"
    }
    assert result["failure_state"] == FailureState.RETRIEVAL_FAILED.value
    assert "no evidence retrieved" in result["error"]
    assert llm.last_user is None
