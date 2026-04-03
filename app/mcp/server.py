import os
from fastmcp import FastMCP
import urllib.request
import json

mcp = FastMCP("Modular RAG MCP Server")

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://rag-engine:8001")

def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

@mcp.tool
def search_documents(query: str, top_k: int = 3) -> dict:
    return post_json(f"{RAG_BASE_URL}/search", {"query": query, "top_k": top_k})

@mcp.tool
def answer_query(query: str, top_k: int = 3) -> dict:
    return post_json(f"{RAG_BASE_URL}/query", {"query": query, "top_k": top_k})

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)