Title: Dataset Reload and GCS Switching
Category: deployment
Tags: reload, gcs, local, runtime-config
Milestone: month2

# Dataset Reload and GCS Switching

The platform supports switching between local documents and cloud-backed document sources through runtime configuration. This is important for deployment portability and for testing the same system across development and cloud environments.

A dataset reload operation updates runtime configuration, rebuilds the active vector store, clears the cached engine state, and reinitializes the engine for subsequent queries.

This makes it possible to test the platform locally using repository documents and later test a cloud-backed flow using GCS without redesigning the application logic.

This behavior supports the project goal of configuration-driven portability rather than environment-specific code forks.