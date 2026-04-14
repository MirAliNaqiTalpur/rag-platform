Title: Retrieval Strategies
Category: retrieval
Tags: simple, hybrid, metadata, reranking
Milestone: month2

# Retrieval Strategies

The platform supports multiple retrieval strategies through a registry-based design. Strategy selection is configuration-driven so that new retrieval modes can be added with minimal changes to the engine core.

Simple retrieval performs straightforward semantic matching over indexed chunks. Hybrid retrieval is intended to combine signals in a way that can improve results when terminology overlap matters. Metadata-aware retrieval supports filtering or narrowing results using structured document properties.

Reranking is treated as an optional pipeline stage. This allows the platform to retrieve an initial set of documents and then reorder them for better relevance.

The main engineering goal is not just retrieval quality, but modularity and experimental flexibility.