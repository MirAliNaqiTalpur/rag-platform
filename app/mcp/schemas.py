from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=3, ge=1, le=20)

class QueryResponse(BaseModel):
    query: str
    documents: List[str]
    answer: str
    latency: Optional[Dict[str, float]] = None

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: Optional[int] = 1

class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[int] = 1