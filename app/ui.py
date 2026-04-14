import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from google.cloud import storage

LOCAL_DOCUMENTS_DIR = "data/documents"
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag-engine:8001").rstrip("/")


def fetch_model_options():
    fallback_models = ["gemini-3.1-flash-lite-preview"]
    fallback_default = fallback_models[0]
    fallback_allow_custom = True

    try:
        response = requests.get(f"{RAG_API_URL}/models", timeout=10)
        response.raise_for_status()
        data = response.json()

        models = data.get("available_models", [])
        default_model = data.get("default_model", "")
        allow_custom_models = data.get("allow_custom_models", True)

        models = [m.strip() for m in models if isinstance(m, str) and m.strip()]

        if not models:
            models = fallback_models.copy()

        if not default_model:
            default_model = models[0]

        if default_model not in models:
            models.insert(0, default_model)

        seen = set()
        unique_models = []
        for model in models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)

        return unique_models, default_model, allow_custom_models

    except Exception:
        return fallback_models, fallback_default, fallback_allow_custom


def save_uploaded_files_local(uploaded_files):
    os.makedirs(LOCAL_DOCUMENTS_DIR, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(LOCAL_DOCUMENTS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(uploaded_file.name)

    return saved_files


def save_uploaded_files_gcs(uploaded_files, bucket_name, prefix=""):
    if not bucket_name:
        raise ValueError("GCS bucket name is required.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    saved_files = []
    for uploaded_file in uploaded_files:
        blob_name = f"{prefix.rstrip('/')}/{uploaded_file.name}" if prefix else uploaded_file.name
        blob = bucket.blob(blob_name)
        blob.upload_from_file(uploaded_file, rewind=True)
        saved_files.append(blob_name)

    return saved_files


def list_current_dataset_files(document_source, gcs_bucket_name="", gcs_prefix=""):
    try:
        if document_source == "gcs":
            if not gcs_bucket_name:
                return []

            client = storage.Client()
            bucket = client.bucket(gcs_bucket_name)

            prefix = gcs_prefix.strip()
            blobs = bucket.list_blobs(prefix=prefix if prefix else None)

            files = []
            for blob in blobs:
                name = blob.name
                if name.endswith("/"):
                    continue
                files.append(name)

            return sorted(files)

        if os.path.exists(LOCAL_DOCUMENTS_DIR):
            return sorted(os.listdir(LOCAL_DOCUMENTS_DIR))

        return []
    except Exception as e:
        st.warning(f"Could not list files from selected source: {e}")
        return []


def reload_backend_dataset(
    document_source,
    gcs_bucket_name,
    gcs_prefix,
    vector_store,
    retriever,
    reranker,
    top_k,
    final_model,
    available_models,
):
    payload = {
        "document_source": document_source,
        "gcs_bucket_name": gcs_bucket_name,
        "gcs_prefix": gcs_prefix,
        "vector_store": vector_store,
        "retriever": retriever,
        "reranker": reranker,
        "generator": "gemini",
        "top_k": int(top_k),
        "default_model": final_model,
        "available_models": ",".join(available_models),
    }

    response = requests.post(f"{RAG_API_URL}/reload-dataset", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def render_documents(documents):
    st.subheader("Retrieved Documents")

    for i, doc in enumerate(documents, start=1):
        if isinstance(doc, dict):
            with st.expander(f"{i}. {doc.get('id', f'doc_{i}')}", expanded=False):
                metadata = doc.get("metadata", {}) or {}
                if metadata:
                    filename = metadata.get("filename")
                    source_path = metadata.get("source_path")
                    category = metadata.get("category")
                    filetype = metadata.get("filetype")

                    if filename:
                        st.markdown(f"**Filename:** {filename}")
                    if source_path:
                        st.markdown(f"**Source Path:** {source_path}")
                    if category:
                        st.markdown(f"**Category:** {category}")
                    if filetype:
                        st.markdown(f"**Filetype:** {filetype}")
                    st.markdown("---")

                st.write(doc.get("text", ""))
        else:
            with st.expander(f"{i}. Document {i}"):
                st.write(str(doc))


st.set_page_config(page_title="Modular RAG Platform", layout="wide")

st.title("Modular RAG Platform Demo")
st.caption("MCP-enabled, modular, cloud-agnostic RAG platform")
st.info("Gemini-powered RAG system with configurable retrieval, dataset source, and model selection.")

with st.sidebar:
    st.header("Configuration")

    vector_store_options = ["faiss", "memory"]
    default_vector_store = os.getenv("VECTOR_STORE", "faiss").lower()
    vector_store_index = (
        vector_store_options.index(default_vector_store)
        if default_vector_store in vector_store_options
        else 0
    )
    vector_store = st.selectbox(
        "Vector Store",
        vector_store_options,
        index=vector_store_index,
    )

    retriever_options = ["simple", "hybrid", "metadata"]
    default_retriever = os.getenv("RETRIEVER", "hybrid").lower()
    retriever_index = (
        retriever_options.index(default_retriever)
        if default_retriever in retriever_options
        else 0
    )
    retriever = st.selectbox(
        "Retriever",
        retriever_options,
        index=retriever_index,
    )

    reranker_options = ["none", "simple"]
    default_reranker = os.getenv("RERANKER", "simple").lower()
    reranker_index = (
        reranker_options.index(default_reranker)
        if default_reranker in reranker_options
        else 0
    )
    reranker = st.selectbox(
        "Reranker",
        reranker_options,
        index=reranker_index,
    )

    st.markdown("**Generator:** Gemini")

    top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=10,
        value=int(os.getenv("TOP_K", 3)),
        step=1,
    )

    if os.getenv("GEMINI_API_KEY", "").strip():
        st.success("Gemini API key loaded")
    else:
        st.error("GEMINI_API_KEY missing")

    st.markdown("---")
    st.subheader("Model")

    model_options, default_model, allow_custom_models = fetch_model_options()

    if allow_custom_models:
        use_custom_model = st.checkbox(
            "Use custom model ID",
            value=False,
            help="Enable this to provide a Gemini model ID manually.",
        )
    else:
        use_custom_model = False

    if use_custom_model:
        final_model = st.text_input(
            "Model ID",
            value=default_model,
            help="Example: gemini-2.5-flash",
        ).strip()
    else:
        selected_model_index = model_options.index(default_model) if default_model in model_options else 0
        final_model = st.selectbox(
            "Gemini Model",
            model_options,
            index=selected_model_index,
        )

    st.markdown("---")
    st.subheader("Document Source")

    document_source_options = ["local", "gcs"]
    default_document_source = os.getenv("DOCUMENT_SOURCE", "local").lower()
    document_source_index = (
        document_source_options.index(default_document_source)
        if default_document_source in document_source_options
        else 0
    )
    document_source = st.selectbox(
        "Source",
        document_source_options,
        index=document_source_index,
    )

    gcs_bucket_name = ""
    gcs_prefix = ""

    if document_source == "gcs":
        gcs_bucket_name = st.text_input(
            "GCS Bucket (optional)",
            value="",
            help="Leave empty to use the default bucket configured during deployment.",
        )

        gcs_prefix = st.text_input(
            "Prefix (optional)",
            value="",
            help="Optional folder path inside the bucket. Leave empty to use the default prefix.",
        )

    refresh_clicked = st.button("Load / Refresh Dataset", use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("Upload Files", use_container_width=True):
        if uploaded_files:
            try:
                if document_source == "gcs":
                    save_uploaded_files_gcs(uploaded_files, gcs_bucket_name, gcs_prefix)
                else:
                    save_uploaded_files_local(uploaded_files)

                st.success("Files uploaded successfully.")
                st.info("Click 'Load / Refresh Dataset' to make the backend use the selected source.")
            except Exception as e:
                st.error(f"Upload failed: {e}")
        else:
            st.warning("Please select at least one file.")

    if refresh_clicked:
        try:
            result = reload_backend_dataset(
                document_source=document_source,
                gcs_bucket_name=gcs_bucket_name,
                gcs_prefix=gcs_prefix,
                vector_store=vector_store,
                retriever=retriever,
                reranker=reranker,
                top_k=top_k,
                final_model=final_model,
                available_models=model_options,
            )

            backend_message = result.get("message", "Dataset loaded successfully.")
            st.success(
                f"{backend_message} "
                f"(Source: '{result.get('document_source')}', "
                f"Documents indexed: {result.get('total_documents_loaded', 0)})"
            )

        except Exception as e:
            st.error(f"Dataset refresh failed: {e}")

    current_files = list_current_dataset_files(document_source, gcs_bucket_name, gcs_prefix)
    if current_files:
        st.caption("Available files in selected source")
        st.write(current_files)

query = st.text_input("Enter your query", "What is RAG?")

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    run_query = st.button("Run Query", use_container_width=True)

with col2:
    run_search = st.button("Search Only", use_container_width=True)

if run_query or run_search:
    if not os.getenv("GEMINI_API_KEY", "").strip():
        st.error("GEMINI_API_KEY is not set in the environment.")
    else:
        try:
            if run_query:
                response = requests.post(
                    f"{RAG_API_URL}/query",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "model": final_model,
                    },
                    timeout=300,
                )

                if response.status_code == 503:
                    try:
                        detail = response.json().get("detail", "")
                    except Exception:
                        detail = ""
                    st.warning(
                        detail
                        or "The selected Gemini model is temporarily under high demand. Please try again shortly or choose another model."
                    )
                    st.stop()

                if response.status_code == 400:
                    try:
                        detail = response.json().get("detail", "")
                    except Exception:
                        detail = ""
                    st.error(detail or "Invalid query or model configuration.")
                    st.stop()

                if response.status_code == 502:
                    try:
                        detail = response.json().get("detail", "")
                    except Exception:
                        detail = ""
                    st.error(detail or "Upstream model service error.")
                    st.stop()

                if response.status_code >= 400:
                    try:
                        detail = response.json().get("detail", "")
                    except Exception:
                        detail = response.text
                    st.error(f"Query failed: {detail or f'HTTP {response.status_code}'}")
                    st.stop()

                result = response.json()

                st.subheader("Generated Answer")
                answer = result.get("answer", "")
                if answer:
                    st.write(answer)
                else:
                    st.warning("The backend returned no answer text.")

                if "latency" in result:
                    st.subheader("Latency")
                    st.json(result["latency"])

                render_documents(result.get("documents", []))

            elif run_search:
                response = requests.post(
                    f"{RAG_API_URL}/search",
                    json={"query": query, "top_k": top_k},
                    timeout=300,
                )

                if response.status_code >= 400:
                    try:
                        detail = response.json().get("detail", "")
                    except Exception:
                        detail = response.text
                    st.error(f"Search failed: {detail or f'HTTP {response.status_code}'}")
                    st.stop()

                result = response.json()

                if "latency" in result:
                    st.subheader("Latency")
                    st.json(result["latency"])

                render_documents(result.get("documents", []))

        except requests.exceptions.Timeout:
            st.error("The backend request timed out.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend service.")
        except Exception as e:
            st.error(f"Unexpected UI error: {e}")