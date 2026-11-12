
from src.agents.literature_reviewer import LiteratureReviewer
from src.schemas.paper import Chunk, ScoredChunk
from src.schemas.state import FailureState


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.requested_k = None

    def query(self, query, k=8, where=None):
        self.requested_k = k
        return self.hits


class FakeLLM:
    def __init__(self):
        self.system_prompt = None
        self.user_prompt = None

    def complete(self, system, user):
        self.system_prompt = system
        self.user_prompt = user
        return "Literature summary with citations."


def make_hit(status, paper_id):
    chunk = Chunk(
        chunk_id=f"{paper_id}#0",
        paper_id=paper_id,
        text=f"Research evidence from {paper_id}.",
        metadata={
            "citation_key": paper_id,
            "title": f"Paper {paper_id}",
            "source": "pubmed",
            "publication_status": status,
        },
    )

    return ScoredChunk(chunk=chunk, score=0.9)


def test_only_peer_reviewed_evidence_is_used():
    hits = [
        make_hit("peer_reviewed", "paper1"),
        make_hit("preprint", "paper2"),
        make_hit("published_unverified", "paper3"),
        make_hit("unknown", "paper4"),
    ]

    store = FakeStore(hits)
    llm = FakeLLM()

    reviewer = LiteratureReviewer(store, llm, top_k=2)

    result = reviewer.run({
        "research_question": "Does model A improve prediction?"
    })

    assert len(result["literature"]) == 1
    assert result["literature"][0]["chunk"]["paper_id"] == "paper1"

    assert list(result["citations"].keys()) == ["paper1"]

    assert "paper1" in llm.user_prompt
    assert "paper2" not in llm.user_prompt
    assert "paper3" not in llm.user_prompt
    assert "paper4" not in llm.user_prompt

    assert result["literature_filter"]["eligible_evidence"] == 1
    assert result["literature_filter"]["excluded_unverified"] == 3


def test_no_peer_reviewed_evidence_returns_failure():
    hits = [
        make_hit("preprint", "paper1"),
        make_hit("unknown", "paper2"),
    ]

    store = FakeStore(hits)
    llm = FakeLLM()

    reviewer = LiteratureReviewer(store, llm)

    result = reviewer.run({
        "research_question": "Does model A improve prediction?"
    })

    assert result["failure_state"] == FailureState.RETRIEVAL_FAILED.value
    assert result["literature"] == []
    assert result["citations"] == {}
    assert result["literature_summary"] == ""
    assert llm.user_prompt is None


def test_retrieves_extra_candidates_before_filtering():
    hits = [
        make_hit("preprint", "paper1"),
        make_hit("peer_reviewed", "paper2"),
    ]

    store = FakeStore(hits)
    reviewer = LiteratureReviewer(
        store,
        FakeLLM(),
        top_k=2,
    )

    reviewer.run({
        "research_question": "Test research question"
    })

    assert store.requested_k == 8


def test_empty_retrieval_returns_failure():
    store = FakeStore([])
    llm = FakeLLM()

    reviewer = LiteratureReviewer(store, llm)

    result = reviewer.run({
        "research_question": "Test research question"
    })

    assert result["failure_state"] == FailureState.RETRIEVAL_FAILED.value
    assert llm.user_prompt is None

def test_missing_publication_status_is_excluded():
    chunk = Chunk(
        chunk_id="pdf:sample#page2#0",
        paper_id="pdf:sample",
        text="PDF research evidence.",
        metadata={
            "citation_key": "pdf:sample",
            "title": "Sample PDF",
            "source": "uploaded_pdf",
            "page_number": 2,
        },
    )

    store = FakeStore([
        ScoredChunk(chunk=chunk, score=0.9)
    ])
    llm = FakeLLM()

    reviewer = LiteratureReviewer(store, llm)

    result = reviewer.run({
        "research_question": "Test question"
    })

    assert result["failure_state"] == FailureState.RETRIEVAL_FAILED.value
    assert result["literature_filter"]["excluded_unverified"] == 1
    assert llm.user_prompt is None


def test_verified_pdf_preserves_page_citation():
    chunk = Chunk(
        chunk_id="pdf:sample#page3#0",
        paper_id="pdf:sample",
        text="Verified PDF research evidence.",
        metadata={
            "citation_key": "pdf:sample",
            "title": "Sample PDF",
            "source": "uploaded_pdf",
            "publication_status": "peer_reviewed",
            "page_number": 3,
        },
    )

    store = FakeStore([
        ScoredChunk(chunk=chunk, score=0.9)
    ])
    llm = FakeLLM()

    reviewer = LiteratureReviewer(store, llm)

    result = reviewer.run({
        "research_question": "Test question"
    })

    assert "page 3" in llm.user_prompt
    assert result["citations"]["pdf:sample"]["pages"] == [3]

def test_uploaded_pdf_visual_without_verified_status_is_excluded():
    chunk = Chunk(
        chunk_id="pdf:sample#visual#page3#0",
        paper_id="pdf:sample",
        text="Figure shows an increasing trend.",
        metadata={
            "citation_key": "pdf:sample",
            "title": "Sample PDF",
            "source": "uploaded_pdf_visual",
            "page_number": 3,
        },
    )

    store = FakeStore([
        ScoredChunk(chunk=chunk, score=0.9)
    ])
    llm = FakeLLM()

    reviewer = LiteratureReviewer(store, llm)

    result = reviewer.run({
        "research_question": "Test question"
    })

    assert result["failure_state"] == FailureState.RETRIEVAL_FAILED.value
    assert result["literature_filter"]["excluded_unverified"] == 1
    assert llm.user_prompt is None