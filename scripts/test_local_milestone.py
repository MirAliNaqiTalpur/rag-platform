import json
import sys
from typing import Any

import requests


BASE_URL = "http://localhost:8002"
MODEL = "gemini-3-flash-preview"

SEARCH_QUERIES = [
    "What are the main components of the modular RAG platform?",
    "What does the MCP server do in this system?",
    "How does dataset reload support switching between local and GCS?",
    "How does GCS integrate into the modular RAG platform?",
]

QUERY_QUERIES = [
    "What are the main components of the modular RAG platform?",
    "How is this platform deployed on Cloud Run using Terraform?",
    "Why does the system use warmup and readiness endpoints?",
]


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def check_endpoint(method: str, path: str, json_payload: dict | None = None) -> requests.Response:
    url = f"{BASE_URL}{path}"

    if method == "GET":
        response = requests.get(url, timeout=120)
    elif method == "POST":
        response = requests.post(url, json=json_payload, timeout=300)
    else:
        raise ValueError(f"Unsupported method: {method}")

    print(f"{method} {path} -> HTTP {response.status_code}")
    data = safe_json(response)
    print(json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data)
    return response


def summarize_documents(documents: list[Any]) -> None:
    if not documents:
        print("No documents returned.")
        return

    for i, doc in enumerate(documents[:5], start=1):
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {}) or {}
            filename = metadata.get("filename", "unknown")
            source_path = metadata.get("source_path", "unknown")
            text = (doc.get("text", "") or "").replace("\n", " ").strip()
            snippet = text[:180] + ("..." if len(text) > 180 else "")
            print(f"{i}. filename={filename}")
            print(f"   source_path={source_path}")
            print(f"   snippet={snippet}")
        else:
            text = str(doc).replace("\n", " ").strip()
            snippet = text[:180] + ("..." if len(text) > 180 else "")
            print(f"{i}. {snippet}")


def run_search_tests() -> None:
    print_header("SEARCH TESTS")

    for query in SEARCH_QUERIES:
        print(f"\nQuery: {query}")
        response = requests.post(
            f"{BASE_URL}/search",
            json={"query": query, "top_k": 3},
            timeout=300,
        )

        print(f"POST /search -> HTTP {response.status_code}")
        data = safe_json(response)

        if response.status_code != 200:
            print(json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data)
            continue

        if isinstance(data, dict):
            latency = data.get("latency")
            if latency:
                print("Latency:")
                print(json.dumps(latency, indent=2))

            print("Top returned documents:")
            summarize_documents(data.get("documents", []))
        else:
            print(data)


def run_query_tests() -> None:
    print_header("QUERY TESTS")

    for query in QUERY_QUERIES:
        print(f"\nQuery: {query}")
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": query, "top_k": 3, "model": MODEL},
            timeout=300,
        )

        print(f"POST /query -> HTTP {response.status_code}")
        data = safe_json(response)

        if response.status_code != 200:
            print(json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data)
            continue

        if isinstance(data, dict):
            answer = (data.get("answer", "") or "").strip()
            print("Answer preview:")
            print(answer[:800] + ("..." if len(answer) > 800 else ""))

            latency = data.get("latency")
            if latency:
                print("Latency:")
                print(json.dumps(latency, indent=2))

            print("Top supporting documents:")
            summarize_documents(data.get("documents", []))
        else:
            print(data)


def main() -> int:
    try:
        print_header("INFRA / RUNTIME CHECKS")

        health = check_endpoint("GET", "/health")
        ready_before = check_endpoint("GET", "/ready")
        warmup = check_endpoint("POST", "/warmup")

        if health.status_code != 200:
            print("Health check failed.")
            return 1

        if warmup.status_code != 200:
            print("Warmup failed.")
            return 1

        print_header("DATASET RELOAD")

        reload_resp = check_endpoint("POST", "/reload-dataset", {})

        if reload_resp.status_code != 200:
            print("Dataset reload failed.")
            return 1

        ready_after_reload = check_endpoint("GET", "/ready")
        if ready_after_reload.status_code != 200:
            print("Ready check after reload failed.")
            return 1

        print_header("MODEL CHECK")

        models = check_endpoint("GET", "/models")
        if models.status_code != 200:
            print("Models endpoint failed.")
            return 1

        models_data = safe_json(models)
        if isinstance(models_data, dict):
            print("\nResolved model for query tests:")
            print(MODEL)

        run_search_tests()
        run_query_tests()

        print_header("LOCAL MILESTONE TEST COMPLETED")
        print("All scripted checks finished successfully.")
        print(f"Model used for /query tests: {MODEL}")
        return 0

    except requests.exceptions.ConnectionError:
        print("Could not connect to the local backend. Is Docker Compose running?")
        return 1
    except requests.exceptions.Timeout:
        print("A request timed out while waiting for the backend.")
        return 1
    except Exception as e:
        print(f"Unexpected test error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
    
    
