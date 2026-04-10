from app.rag.engine import RAGEngine

rag = RAGEngine()

queries = [
    "What is RAG?",
    "How does a RAG pipeline work?",
    "How is RAG different from fine-tuning?",
    "What are access control policies?",
]

for query in queries:
    result = rag.query(query)

    print("\n" + "=" * 80)
    print("Query:")
    print(result["query"])

    print("\nTop Retrieved Docs:")
    for i, d in enumerate(result["documents"], start=1):
        if isinstance(d, dict):
            text = d.get("text", "").strip()
            source = d.get("id", d.get("source", "unknown"))
        else:
            text = str(d).strip()
            source = "unknown"

        short_text = text[:200].replace("\n", " ")
        print(f"{i}. Source: {source}")
        print(f"   Text: {short_text}")
        print()

    print("Latency:")
    for key, value in result["latency"].items():
        print(f"{key}: {value} sec")