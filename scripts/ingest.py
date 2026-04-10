import warnings
warnings.filterwarnings("ignore")
from app.core.config import CONFIG
from app.ingestion.loader import load_documents
from app.vectorstore.factory import get_vector_store

import os
print("DOCUMENT_SOURCE =", os.getenv("DOCUMENT_SOURCE", "local"))

# Load documents
docs = load_documents("data/documents")

# Initialize vector store via factory
store = get_vector_store(CONFIG)

# Add documents
store.add_documents(docs)
store.save()

# Test search
query = "sample query"
results = store.search(query, top_k=CONFIG["top_k"])

print("\n--- SEARCH RESULTS ---")
for r in results:
    if isinstance(r, dict):
        print(f"ID: {r.get('id')}")
        print(f"Category: {r.get('metadata', {}).get('category')}")
        print(r.get("text", "")[:200])
    else:
        print(str(r)[:200])
    print("------")

print("\nIngestion + search complete!")