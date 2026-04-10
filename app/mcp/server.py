from fastmcp import FastMCP
from app.mcp.tools import dispatch_tool

mcp = FastMCP("Modular RAG MCP Server")


@mcp.tool
def search_documents(query: str, top_k: int = 3) -> dict:
    return dispatch_tool(
        "search_documents",
        {"query": query, "top_k": top_k},
    )


@mcp.tool
def answer_query(query: str, top_k: int = 3, model: str = None) -> dict:
    return dispatch_tool(
        "answer_query",
        {"query": query, "top_k": top_k, "model": model},
    )


@mcp.tool
def search_by_metadata(
    query: str,
    category: str = None,
    filetype: str = None,
    top_k: int = 3,
) -> dict:
    return dispatch_tool(
        "search_by_metadata",
        {
            "query": query,
            "category": category,
            "filetype": filetype,
            "top_k": top_k,
        },
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)