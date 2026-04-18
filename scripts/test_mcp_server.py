import asyncio
from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"
MODEL = "gemini-3-flash-preview"


def extract_data(result):
    """
    Handle common FastMCP result shapes safely.
    """
    if hasattr(result, "data") and result.data is not None:
        return result.data
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    if hasattr(result, "structuredContent") and result.structuredContent is not None:
        return result.structuredContent
    return result


def print_documents(data, limit=2):
    docs = data.get("documents", []) if isinstance(data, dict) else []
    if not docs:
        print("Top Documents: None")
        return

    print("Top Documents:")
    for doc in docs[:limit]:
        if not isinstance(doc, dict):
            print(f"- {doc}")
            continue

        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        filename = meta.get("filename") or doc.get("id") or "unknown"
        category = meta.get("category")

        if category:
            print(f"- {filename} ({category})")
        else:
            print(f"- {filename}")


def print_latency(data):
    latency = data.get("latency", {}) if isinstance(data, dict) else {}
    total_ms = latency.get("total_ms")
    retrieval_ms = latency.get("retrieval_ms")
    generation_ms = latency.get("generation_ms")

    if total_ms is None and retrieval_ms is None and generation_ms is None:
        return

    parts = []
    if retrieval_ms is not None:
        parts.append(f"retrieval={round(retrieval_ms, 2)} ms")
    if generation_ms is not None:
        parts.append(f"generation={round(generation_ms, 2)} ms")
    if total_ms is not None:
        parts.append(f"total={round(total_ms, 2)} ms")

    print("Latency: " + " | ".join(parts))


def print_answer(data, limit=300):
    answer = data.get("answer") if isinstance(data, dict) else None
    if not answer:
        print("Answer: None")
        return

    answer = str(answer).strip().replace("\n", " ")
    if len(answer) > limit:
        answer = answer[:limit].rstrip() + "..."
    print("Answer:")
    print(answer)


async def main():
    client = Client(SERVER_URL)

    async with client:
        print("\n=== MCP SERVER CHECK ===")

        pong = await client.ping()
        print(f"Ping: {pong}")

        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        print("\nAvailable tools:")
        for name in tool_names:
            print(f"- {name}")

        # ---------------------------
        # SEARCH TEST
        # ---------------------------
        print("\n=== SEARCH TEST ===")
        search_query = "What is RAG?"
        result = await client.call_tool(
            "search_documents",
            {"query": search_query, "top_k": 2}
        )
        data = extract_data(result)

        print(f"Query: {search_query}")
        print_documents(data)
        print_latency(data)

        # ---------------------------
        # ANSWER TEST
        # ---------------------------
        print("\n=== ANSWER TEST ===")
        answer_query = "What does the MCP server do in this platform?"
        result = await client.call_tool(
            "answer_query",
            {"query": answer_query, "top_k": 2, "model": MODEL}
        )
        data = extract_data(result)

        print(f"Query: {answer_query}")
        print_answer(data)
        print_documents(data)
        print_latency(data)

        # ---------------------------
        # METADATA TEST
        # ---------------------------
        print("\n=== METADATA TEST ===")
        metadata_params = {"query": "architecture", "category": "architecture", "top_k": 2}
        result = await client.call_tool("search_by_metadata", metadata_params)
        data = extract_data(result)

        print("Query: architecture | category=architecture")
        print_documents(data)
        print_latency(data)

        print("\n=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(main())