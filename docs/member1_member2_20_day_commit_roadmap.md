# Member 1 + Member 2: 20-day commit roadmap

Use this as a forward-looking work plan. Make each commit when that day's work is
actually complete; do not backdate commits to imply work happened on earlier days.
The repository already contains several of these foundations, so inspect each item
and adapt it instead of duplicating existing code.

| Day | Suggested commit | Scope |
| --- | --- | --- |
| 1 | `docs: map member 1 and 2 architecture` | Record module ownership, API contracts, and current gaps. |
| 2 | `refactor: define shared paper and workflow state` | Align `Paper`, `ResearchState`, and agent input/output keys. |
| 3 | `feat: build literature search clients` | Normalize arXiv and PubMed search results and API errors. |
| 4 | `feat: deduplicate scholarly records` | Match DOI/title aliases and retain the strongest publication metadata. |
| 5 | `feat: add bounded scholarly PDF downloads` | Restrict PDF downloads to trusted HTTPS hosts and size limits. |
| 6 | `feat: extract text from scholarly PDFs` | Parse text with PyMuPDF and keep abstract fallback behavior. |
| 7 | `feat: chunk papers with citation metadata` | Preserve paper identity, section text, and citation fields on chunks. |
| 8 | `feat: index papers in local vector store` | Embed and upsert chunks idempotently into ChromaDB. |
| 9 | `feat: retrieve cited literature evidence` | Return ranked chunks with stable citation metadata. |
| 10 | `feat: ground literature summaries in retrieved sources` | Keep the Literature Reviewer constrained to returned evidence. |
| 11 | `feat: orchestrate research agents with LangGraph` | Wire review, hypothesis, design, execution, critique, and synthesis. |
| 12 | `feat: validate structured hypothesis and plan output` | Parse model output and report schema failures as workflow state. |
| 13 | `feat: add bounded critic revision routing` | Revise only on explicit critique and enforce the configured limit. |
| 14 | `feat: checkpoint research runs` | Persist state by run ID and expose that ID to the caller. |
| 15 | `feat: add ingestion controls and source reporting` | Configure source limits, PDF limits, and partial-source failures. |
| 16 | `docs: document member 1 and 2 configuration` | Explain API environment variables, model settings, and local storage. |
| 17 | `feat: add offline retrieval demo` | Demonstrate indexing and retrieval without live API or LLM calls. |
| 18 | `test: cover retrieval and workflow failure paths` | Add focused tests for source failures, parsing errors, and routing. |
| 19 | `docs: record evaluation protocol` | Describe retrieval relevance, citation coverage, and workflow criteria. |
| 20 | `docs: publish member 1 and 2 handoff` | Summarize completed pieces, limitations, configuration, and next work. |

The current branch already includes arXiv/PubMed clients, deduplication, chunking,
embeddings, ChromaDB retrieval, LangGraph agents, and critic-bounded revision. Recent
changes add optional bounded PDF extraction and run-keyed checkpoints. Commit those
actual changes with their real commit date when you are ready.
