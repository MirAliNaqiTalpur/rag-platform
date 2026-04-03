from fastapi import FastAPI
from app.mcp.schemas import (
    QueryRequest,
    QueryResponse,
    JSONRPCRequest,
    JSONRPCResponse,
)
from app.mcp.tools import search_documents, dispatch_tool

app = FastAPI(title="Modular RAG Platform")

@app.get("/")
def root():
    return {"message": "RAG MCP server is running"}

@app.post("/search", response_model=QueryResponse)
def run_search(request: QueryRequest):
    result = search_documents(query=request.query, top_k=request.top_k)
    return QueryResponse(
        query=result["query"],
        documents=result["documents"],
        answer=result["answer"],
        latency=result["latency"]
    )

@app.post("/rpc", response_model=JSONRPCResponse)
def run_rpc(request: JSONRPCRequest):
    try:
        result = dispatch_tool(request.method, request.params)
        return JSONRPCResponse(
            jsonrpc="2.0",
            result=result,
            id=request.id
        )
    except Exception as e:
        return JSONRPCResponse(
            jsonrpc="2.0",
            error={
                "code": -32000,
                "message": str(e)
            },
            id=request.id
        )