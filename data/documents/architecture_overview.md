Title: Architecture Overview
Category: architecture
Tags: rag, modularity, services, docker
Milestone: month2

# Architecture Overview

This project implements a modular Retrieval-Augmented Generation platform designed as reusable infrastructure rather than a single-purpose script. The platform separates concerns between the RAG engine, MCP server, document storage flow, and deployment configuration.

The RAG engine is responsible for retrieval orchestration, reranking, and answer generation. It should remain independent from cloud-specific details. Retrieval behavior is selected through configuration rather than hard-coded changes.

The MCP server exposes retrieval capabilities through schema-defined tools. This allows tool use through a protocol interface rather than tightly coupling retrieval to one agent framework.

The deployment design emphasizes portability. Local development uses Docker Compose. Cloud deployment uses Terraform and Cloud Run. The same core application logic should work across environments with minimal code changes.

The design goal is to treat RAG as a platform layer that other document-intensive systems can build on.