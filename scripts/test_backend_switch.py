from app.core.config import CONFIG
from app.ingestion.loader import load_documents
from app.vectorstore.factory import get_vector_store

docs = load_documents("data/documents")
store = get_vector_store(CONFIG)

store.add_documents(docs)

query = "What is RAG?"
results = store.search(query, top_k=2)

print(f"\nBackend: {CONFIG['vector_store']}")
print(f"Query: {query}")
print("Results:")
for r in results:
    print("-", r.strip())