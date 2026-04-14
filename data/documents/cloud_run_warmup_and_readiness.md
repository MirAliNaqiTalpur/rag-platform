Title: Cloud Run Warmup and Readiness
Category: runtime
Tags: cloud-run, warmup, readiness, lazy-init
Milestone: month2

# Cloud Run Warmup and Readiness

The RAG engine uses lazy initialization so that the web service can start quickly without always loading heavy retrieval components immediately. This is useful for containerized deployment, especially in environments where startup time matters.

A health endpoint indicates that the API process is alive. A readiness endpoint reflects whether the retrieval engine has been initialized and is ready for real work. A warmup endpoint can initialize the engine intentionally before user traffic arrives.

This separation makes the system easier to validate and more suitable for cloud deployment than a design where the first end-user query silently performs all initialization work.