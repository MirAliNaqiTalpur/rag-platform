Title: MCP Tooling Design
Category: mcp
Tags: mcp, tools, schema, json-rpc
Milestone: month2

# MCP Tooling Design

The MCP server exposes retrieval functions as structured tools. Instead of directly calling Python functions inside an agent loop, the platform defines explicit tool interfaces with predictable inputs and outputs.

Key tools include document search, metadata-aware retrieval, and question answering through the RAG engine. These tools are designed to support deterministic request and response behavior.

The protocol-driven design improves interoperability. It allows the same retrieval capability to be exposed in a standard form and consumed by different agent environments.

This design also improves modularity. The MCP layer does not implement retrieval logic itself. It delegates retrieval to the RAG engine through a defined boundary.