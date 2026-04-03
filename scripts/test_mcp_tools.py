from app.rag.engine import RAGEngine

rag_engine = RAGEngine()

print("\n--- TEST: search_documents ---")
search_result = rag_engine.search_only(query="What is RAG?", top_k=2)
print(search_result)

print("\n--- TEST: answer_query ---")
answer_result = rag_engine.query(query="What is RAG?", top_k=2)
print(answer_result)