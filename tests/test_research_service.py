
from types import SimpleNamespace

from src.workflow.service import ResearchService


class FakeMemory:
    def __init__(self):
        self.records = [
            {
                "research_question": "Previous research on classification",
                "hypothesis": "Regularization may improve accuracy",
                "verdict": "inconclusive",
                "findings": {},
            }
        ]
        self.retrieved_question = None
        self.saved_state = None
        self.saved_thread_id = None

    def retrieve(self, question, limit=3, exclude_thread_id=None):
        self.retrieved_question = question
        return self.records

    def save(self, state, thread_id):
        self.saved_state = state
        self.saved_thread_id = thread_id
        return "test-memory-id"


class FakeApp:
    def __init__(self):
        self.initial_state = None

    def invoke(self, state, config=None):
        self.initial_state = state
        if state is None:
            return {"research_question": "Resumed research"}
        return {
            **state,
            "failure_state": None,
            "verdict": "inconclusive",
        }


def make_service(memory, app):
    service = ResearchService.__new__(ResearchService)
    service.memory = memory
    service.enable_ingestion = False
    service.checkpoint_path = ":memory:"
    service._build_app = lambda experiment_mode: app
    return service


def test_service_retrieves_and_saves_research_memory():
    memory = FakeMemory()
    app = FakeApp()
    service = make_service(memory, app)

    result = service.run(
        "Does regularization improve classification accuracy?",
        experiment_mode="protocol",
        ingest=False,
        thread_id="service-memory-test",
    )

    # Previous research is retrieved.
    assert memory.retrieved_question == (
        "Does regularization improve classification accuracy?"
    )

    # Retrieved memory is passed into the workflow.
    assert app.initial_state["memory_context"] == memory.records
    assert app.initial_state["memory_retrieval"]["records_found"] == 1

    # Final research state is saved.
    assert memory.saved_state is not None
    assert memory.saved_thread_id == "service-memory-test"

    # Returned result contains memory save information.
    assert result["state"]["memory_saved"] is True
    assert result["state"]["memory_record_id"] == "test-memory-id"


def test_service_resume_does_not_retrieve_new_memory():
    memory = FakeMemory()

    class ResumeApp:
        def get_state(self, config):
            return SimpleNamespace(
                values={"research_question": "Existing research"}
            )

        def invoke(self, state, config=None):
            assert state is None
            return {
                "research_question": "Existing research",
                "failure_state": None,
            }

    app = ResumeApp()
    service = make_service(memory, app)

    result = service.run(
        "Existing research",
        experiment_mode="protocol",
        ingest=False,
        thread_id="resume-memory-test",
        resume=True,
    )

    # Resume should use checkpoint state, not retrieve fresh memory.
    assert memory.retrieved_question is None
    assert result["state"]["research_question"] == "Existing research"
