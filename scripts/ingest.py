import warnings
warnings.filterwarnings("ignore")
from app.core.config import CONFIG
from app.ingestion.loader import load_documents
from app.vectorstore.factory import get_vector_store

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
    print(r[:200])
    print("------")

print("\nIngestion + search complete!")