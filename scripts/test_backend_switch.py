from app.core.config import CONFIG
from app.vectorstore.factory import get_vector_store
from app.vectorstore.utils import initialize_store

query = "What is RAG?"

store = get_vector_store(CONFIG)
store = initialize_store(store)

results = store.search(query, top_k=CONFIG["top_k"])

print(f"\nBackend: {CONFIG['vector_store']}")
print(f"Query: {query}")
print("Results:")

for r in results:
    if isinstance(r, dict):
        print("-", r.get("text", "").strip())
    else:
        print("-", str(r).strip())