import asyncio
import json
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"

SEARCH_QUERIES = [
    "What is RAG?",
    "difference between RAG and fine-tuning",
    "typical RAG pipeline",
    "access permissions",
    "audit logging",
]


def print_result(label: str, result):
    print(f"\n--- {label} ---")

    if hasattr(result, "data") and result.data is not None:
        data = result.data
    elif hasattr(result, "structured_content") and result.structured_content is not None:
        data = result.structured_content
    else:
        print(result)
        return

    print(json.dumps(data, indent=2, ensure_ascii=False))


async def main():
    client = Client(SERVER_URL)

    async with client:
        print("\n=== MCP SERVER CHECK ===")

        pong = await client.ping()
        print(f"\nPing: {pong}")

        tools = await client.list_tools()
        print("\nAvailable tools:")
        for tool in tools:
            print(f"- {tool.name}")

        print("\n=== SEARCH_DOCUMENTS TESTS ===")
        for query in SEARCH_QUERIES:
            result = await client.call_tool(
                "search_documents",
                {"query": query, "top_k": 2}
            )
            print_result(f"search_documents | query = {query}", result)

        print("\n=== ANSWER_QUERY TESTS ===")
        for query in SEARCH_QUERIES:
            result = await client.call_tool(
                "answer_query",
                {"query": query, "top_k": 2}
            )
            print_result(f"answer_query | query = {query}", result)

        print("\n=== SEARCH_BY_METADATA TESTS ===")
        metadata_tests = [
            {"query": "policy", "category": "policy", "top_k": 5},
            {"query": "logging", "category": "policy", "top_k": 5},
            {"query": "access", "category": "policy", "top_k": 5},
            {"query": "RAG", "category": "rag", "top_k": 5},
        ]

        for params in metadata_tests:
            result = await client.call_tool("search_by_metadata", params)
            print_result(f"search_by_metadata | params = {params}", result)


if __name__ == "__main__":
    asyncio.run(main())