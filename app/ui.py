import sys
import os
import shutil
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from google.cloud import storage

from app.rag.engine import RAGEngine
from app.core.config import CONFIG
from app.ingestion.loader import load_documents
from app.vectorstore.factory import get_vector_store
from app.storage.factory import get_storage_backend

DOCUMENTS_DIR = "data/documents"
FAISS_INDEX_DIR = "data/faiss_index"


def fetch_model_options():
    fallback_models = ["gemini-3.1-flash-lite-preview"]
    fallback_default = fallback_models[0]
    fallback_allow_custom = False

    rag_api_url = os.getenv("RAG_API_URL", "http://localhost:8002").rstrip("/")

    try:
        response = requests.get(f"{rag_api_url}/models", timeout=10)
        response.raise_for_status()
        data = response.json()

        models = data.get("available_models", [])
        default_model = data.get("default_model", "")
        allow_custom_models = data.get("allow_custom_models", False)

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


def set_runtime_config(
    vector_store,
    retriever,
    reranker,
    generator,
    top_k,
    document_source,
    gcs_bucket_name="",
    gcs_prefix="",
    default_gemini_model="",
    available_gemini_models="",
):
    os.environ["VECTOR_STORE"] = vector_store
    os.environ["RETRIEVER"] = retriever
    os.environ["RERANKER"] = reranker
    os.environ["GENERATOR"] = generator
    os.environ["TOP_K"] = str(top_k)

    os.environ["DOCUMENT_SOURCE"] = document_source
    os.environ["GCS_BUCKET_NAME"] = gcs_bucket_name
    os.environ["GCS_PREFIX"] = gcs_prefix

    os.environ["DEFAULT_GEMINI_MODEL"] = default_gemini_model
    os.environ["AVAILABLE_GEMINI_MODELS"] = available_gemini_models
    os.environ["DEFAULT_MODEL"] = default_gemini_model
    os.environ["AVAILABLE_MODELS"] = available_gemini_models

    CONFIG["vector_store"] = vector_store
    CONFIG["retriever"] = retriever
    CONFIG["reranker"] = reranker
    CONFIG["generator"] = generator
    CONFIG["top_k"] = int(top_k)

    if generator == "gemini":
        os.environ["LLM_PROVIDER"] = "gemini"
    else:
        os.environ["GEMINI_API_KEY"] = ""


def save_uploaded_files_local(uploaded_files):
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(uploaded_file.name)

    return saved_files


def save_uploaded_files_gcs(uploaded_files, bucket_name, prefix=""):
    if not bucket_name:
        raise ValueError("GCS bucket name is required when document source is gcs.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    saved_files = []
    for uploaded_file in uploaded_files:
        if prefix:
            blob_name = f"{prefix.rstrip('/')}/{uploaded_file.name}"
        else:
            blob_name = uploaded_file.name

        blob = bucket.blob(blob_name)
        blob.upload_from_file(uploaded_file, rewind=True)
        saved_files.append(blob_name)

    return saved_files


def list_current_dataset_files(document_source, gcs_bucket_name="", gcs_prefix=""):
    try:
        if document_source == "gcs":
            if not gcs_bucket_name:
                return []

            backend = get_storage_backend(default_local_path=DOCUMENTS_DIR)
            return sorted(backend.list_files(prefix=gcs_prefix))

        if os.path.exists(DOCUMENTS_DIR):
            return sorted(os.listdir(DOCUMENTS_DIR))

        return []
    except Exception as e:
        st.warning(f"Could not list dataset files: {e}")
        return []


def rebuild_vector_store():
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)

    docs = load_documents(DOCUMENTS_DIR)

    store = get_vector_store(CONFIG)
    store.add_documents(docs)

    try:
        store.save()
    except Exception:
        pass

    return len(docs)


def render_documents(documents, heading="Retrieved Documents"):
    st.subheader(heading)

    for i, doc in enumerate(documents, start=1):
        if isinstance(doc, dict):
            title = f"{i}. {doc.get('id', f'document_{i}')}"
            with st.expander(title):
                metadata = doc.get("metadata", {})
                st.markdown(f"**Category:** {metadata.get('category', 'unknown')}")
                st.markdown(f"**Filetype:** {metadata.get('filetype', 'unknown')}")
                source_path = metadata.get("source_path")
                if source_path:
                    st.markdown(f"**Source Path:** {source_path}")
                st.markdown("---")
                st.write(doc.get("text", ""))
        else:
            with st.expander(f"{i}. Document {i}"):
                st.write(str(doc))


st.set_page_config(page_title="Modular RAG Platform", layout="wide")

st.title("Modular RAG Platform Demo")
st.caption("MCP-enabled, modular, cloud-agnostic RAG platform")
st.info(
    "This demo shows a configurable RAG platform with interchangeable vector stores, "
    "retrieval strategies, reranking, and generator options."
)

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
        index=vector_store_index
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
        index=retriever_index
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
        index=reranker_index
    )

    generator_options = ["mock", "gemini"]
    default_generator = os.getenv("GENERATOR", "mock").lower()
    generator_index = (
        generator_options.index(default_generator)
        if default_generator in generator_options
        else 0
    )
    generator = st.selectbox(
        "Generator",
        generator_options,
        index=generator_index
    )

    top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=10,
        value=int(os.getenv("TOP_K", 3)),
        step=1
    )

    selected_gemini_model = ""
    final_selected_model = ""
    custom_model_name = ""
    available_gemini_models_str = ""

    if generator == "gemini":
        existing_api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if existing_api_key:
            st.success("Gemini API key loaded from environment.")
        else:
            st.error(
                "GEMINI_API_KEY is not set in the environment. "
                "Set it before starting the app."
            )

        model_options, default_model, allow_custom_models = fetch_model_options()

        selected_gemini_model = st.selectbox(
            "Gemini Model",
            model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            help="Choose which Gemini model to use for answer generation."
        )

        if allow_custom_models:
            use_custom_model = st.checkbox(
                "Use custom model name",
                value=False,
                help="Enable this to override the dropdown with a custom Gemini model name."
            )

            if use_custom_model:
                custom_model_name = st.text_input(
                    "Custom model name",
                    value="",
                    help="Example: gemini-3-flash-preview"
                ).strip()

        final_selected_model = custom_model_name if custom_model_name else selected_gemini_model
        available_gemini_models_str = ",".join(model_options)

    st.markdown("---")
    st.subheader("Document Source")

    source_options = ["local", "gcs"]
    default_source = os.getenv("DOCUMENT_SOURCE", "local").lower()
    source_index = (
        source_options.index(default_source)
        if default_source in source_options
        else 0
    )

    document_source = st.selectbox(
        "Source",
        source_options,
        index=source_index
    )

    gcs_bucket_name = ""
    gcs_prefix = ""

    if document_source == "gcs":
        gcs_bucket_name = st.text_input(
            "GCS Bucket Name",
            value=os.getenv("GCS_BUCKET_NAME", "")
        )
        gcs_prefix = st.text_input(
            "GCS Prefix",
            value=os.getenv("GCS_PREFIX", "")
        )

    st.markdown("---")
    st.subheader("Dataset")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )

    if st.button("Upload and Rebuild Index", use_container_width=True):
        set_runtime_config(
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            top_k=top_k,
            document_source=document_source,
            gcs_bucket_name=gcs_bucket_name,
            gcs_prefix=gcs_prefix,
            default_gemini_model=final_selected_model if generator == "gemini" else "",
            available_gemini_models=available_gemini_models_str,
        )

        if uploaded_files:
            try:
                if document_source == "gcs":
                    saved = save_uploaded_files_gcs(
                        uploaded_files=uploaded_files,
                        bucket_name=gcs_bucket_name,
                        prefix=gcs_prefix,
                    )
                else:
                    saved = save_uploaded_files_local(uploaded_files)

                total_docs = rebuild_vector_store()

                st.success(
                    f"Uploaded {len(saved)} file(s) and rebuilt index. "
                    f"Total documents loaded: {total_docs}"
                )
            except Exception as e:
                st.error(f"Upload/rebuild failed: {e}")
        else:
            st.warning("Please select at least one file.")

    current_files = list_current_dataset_files(
        document_source=document_source,
        gcs_bucket_name=gcs_bucket_name,
        gcs_prefix=gcs_prefix,
    )
    if current_files:
        st.caption("Current dataset files")
        st.write(current_files)

query = st.text_input("Enter your query", value="What is RAG?")

col1, col2 = st.columns(2)
with col1:
    run_query = st.button("Run Query", use_container_width=True)
with col2:
    run_search = st.button("Search Only", use_container_width=True)

if run_query or run_search:
    set_runtime_config(
        vector_store=vector_store,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        top_k=top_k,
        document_source=document_source,
        gcs_bucket_name=gcs_bucket_name,
        gcs_prefix=gcs_prefix,
        default_gemini_model=final_selected_model if generator == "gemini" else "",
        available_gemini_models=available_gemini_models_str,
    )

    st.subheader("Current Configuration")
    st.json({
        "vector_store": vector_store,
        "retriever": retriever,
        "reranker": reranker,
        "generator": generator,
        "top_k": top_k,
        "document_source": document_source,
        "gcs_bucket_name": gcs_bucket_name if document_source == "gcs" else "",
        "gcs_prefix": gcs_prefix if document_source == "gcs" else "",
        "gemini_model": final_selected_model if generator == "gemini" else "",
    })

    if generator == "gemini" and not os.getenv("GEMINI_API_KEY", "").strip():
        st.error("GEMINI_API_KEY is not set in the environment.")
    else:
        try:
            rag = RAGEngine()

            if run_query:
                result = rag.query(
                    query=query,
                    top_k=top_k,
                    model_name=final_selected_model if generator == "gemini" else None
                )

                if generator == "mock":
                    st.subheader("Generated Answer (Mock)")
                else:
                    st.subheader("Generated Answer")

                st.write(result["answer"])

                st.subheader("Latency")
                st.json(result["latency"])

                render_documents(result["documents"])

            elif run_search:
                result = rag.search_only(query=query, top_k=top_k)
                render_documents(result["documents"], heading="Search Results")

        except Exception as e:
            st.error(f"Error: {e}")