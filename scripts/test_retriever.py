from app.rag.retriever import DenseRetriever

retriever = DenseRetriever()

query = "What is RAG?"
results = retriever.retrieve(query)

print("\n--- RETRIEVER RESULTS ---")
for r in results:
    print(r.strip())
    print("------")