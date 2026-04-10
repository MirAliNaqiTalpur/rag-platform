import io
from pathlib import Path
from pypdf import PdfReader

from app.storage.factory import get_storage_backend


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_BINARY_EXTENSIONS = {".pdf"}


def read_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def detect_category(filename: str) -> str:
    return filename.split("_")[0] if "_" in filename else "general"


def build_document(source_path: str, content: str):
    file_name = Path(source_path).name
    ext = Path(file_name).suffix.lower()

    return {
        "id": source_path,
        "text": content,
        "metadata": {
            "filename": file_name,
            "source_path": source_path,
            "category": detect_category(file_name),
            "filetype": ext,
        },
    }


def load_documents(folder_path: str = "data/documents", prefix: str = ""):
    backend = get_storage_backend(default_local_path=folder_path)
    docs = []

    for path in backend.list_files(prefix=prefix):
        file_name = Path(path).name
        ext = Path(file_name).suffix.lower()

        content = None

        try:
            if ext in SUPPORTED_TEXT_EXTENSIONS:
                content = backend.read_text(path)
            elif ext in SUPPORTED_BINARY_EXTENSIONS:
                file_bytes = backend.read_bytes(path)
                content = read_pdf_bytes(file_bytes)
            else:
                print(f"Skipping unsupported file: {path}")
                continue

            if content and content.strip():
                docs.append(build_document(path, content))
            else:
                print(f"Skipping empty content: {path}")

        except Exception as e:
            print(f"Error loading {path}: {e}")

    print(f"Loaded {len(docs)} document(s)")
    return docs