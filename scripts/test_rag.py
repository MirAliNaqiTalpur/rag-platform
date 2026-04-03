from app.rag.engine import RAGEngine

rag = RAGEngine()

query = "What is RAG?"
result = rag.query(query)

print("\n--- FINAL OUTPUT ---")

print("\nQuery:")
print(result["query"])

print("\nRetrieved Docs:")
for d in result["documents"]:
    print("-", d.strip())

print("\nGenerated Answer:")
print(result["answer"])

print("\nLatency:")
for k, v in result["latency"].items():
    print(f"{k}: {v} sec")