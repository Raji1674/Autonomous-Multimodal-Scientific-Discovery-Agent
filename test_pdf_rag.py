from src.utils.config import load_config
from src.vectorstore.chroma_store import ChromaStore
from src.vectorstore.embeddings import SentenceTransformerEmbedder
from src.ingestion.pdf_processor import extract_pdf_content
from src.ingestion.pdf_indexer import index_pdf_content

config = load_config()

store = ChromaStore(
    config.retrieval.chroma_path,
    SentenceTransformerEmbedder(config.retrieval.embedding_model)
)

pdf_path = input("Enter research PDF path: ").strip().strip('"')

result = extract_pdf_content(pdf_path)
indexed = index_pdf_content(result, store)

print("\nPDF INDEXING RESULT")
print(indexed)
print("TOTAL CHROMA DOCUMENTS:", store.count())

question = input("\nEnter a research question: ").strip()
results = store.query(question, k=5)

print("\nRETRIEVED PDF / PAPER CHUNKS")
for item in results:
    print("\nScore:", round(item.score, 4))
    print("Source:", item.chunk.metadata.get("source"))
    print("Page:", item.chunk.metadata.get("page_number", "N/A"))
    print("Text:", item.chunk.text[:500])
