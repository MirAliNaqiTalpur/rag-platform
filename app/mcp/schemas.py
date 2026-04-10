from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DocumentMetadata(BaseModel):
    filename: Optional[str] = None
    source_path: Optional[str] = None
    category: Optional[str] = None
    filetype: Optional[str] = None


class RetrievedDocument(BaseModel):
    id: str
    text: str
    metadata: Optional[DocumentMetadata] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=20)


class MetadataSearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    filetype: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=20)


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=20)
    model: Optional[str] = None


class ToolResult(BaseModel):
    status: str = "success"
    query: Optional[str] = None
    answer: Optional[str] = None
    documents: List[RetrievedDocument] = Field(default_factory=list)
    latency: Optional[Dict[str, float]] = None


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: Optional[int] = 1


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[JSONRPCError] = None
    id: Optional[int] = 1