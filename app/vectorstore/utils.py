from app.ingestion.loader import load_documents


def initialize_store(store, documents_path="data/documents"):
    """
    Initialize a vector store.

    If the store has a usable persisted index, load it.
    Otherwise, load documents and add them fresh.
    """
    try:
        store.load()
        # if load worked and populated documents, use it
        if hasattr(store, "documents") and store.documents:
            return store
    except Exception:
        pass

    docs = load_documents(documents_path)
    store.add_documents(docs)

    try:
        store.save()
    except Exception:
        # some stores (like memory) may not persist
        pass

    return store