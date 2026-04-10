from app.mcp.tools import dispatch_tool

print("\n--- TEST: search_documents ---")
search_result = dispatch_tool(
    "search_documents",
    {"query": "What is RAG?", "top_k": 2}
)
print(search_result)

print("\n--- TEST: answer_query ---")
answer_result = dispatch_tool(
    "answer_query",
    {"query": "What is RAG?", "top_k": 2}
)
print(answer_result)

print("\n--- TEST: search_by_metadata ---")
metadata_result = dispatch_tool(
    "search_by_metadata",
    {"query": "policy", "category": "policy", "top_k": 5}
)
print(metadata_result)