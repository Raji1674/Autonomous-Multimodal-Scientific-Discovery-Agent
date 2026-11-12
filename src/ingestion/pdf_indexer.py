from pathlib import Path
from src.schemas.paper import Chunk
from src.vectorstore.base import VectorStore


def index_pdf_content(pdf_result: dict, store: VectorStore) -> dict:
    """Index extracted PDF pages into the existing vector store."""
    pdf_name = pdf_result["pdf_name"]
    pdf_id = f"pdf:{Path(pdf_name).stem}"

    chunks = []
    chunk_size = 1000
    overlap = 150

    for page in pdf_result["pages"]:
        text = page["text"].strip()
        start = 0
        index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end].strip()

            if content:
                chunks.append(
                    Chunk(
                        chunk_id=f"{pdf_id}#page{page['page_number']}#{index}",
                        paper_id=pdf_id,
                        text=content,
                        metadata={
                            "title": pdf_name,
                            "source": "uploaded_pdf",
                            "page_number": page["page_number"],
                            "chunk_index": index,
                            "citation_key": pdf_id
                        }
                    )
                )

            if end >= len(text):
                break

            start = end - overlap
            index += 1

    written = store.upsert(chunks)

    return {
        "pdf_name": pdf_name,
        "chunks_created": len(chunks),
        "chunks_indexed": written,
        "success": written == len(chunks)
    }
