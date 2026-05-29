"""RAG pipeline for linear equation pedagogy from Sandoval et al. (2023)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_PATH = ROOT / "data" / "knowledge" / "paper_chunks.json"
BUNDLED_CHROMA_PATH = ROOT / "data" / "chroma_db"
EMBEDDINGS_CACHE_PATH = ROOT / "data" / "knowledge" / "embeddings_cache.json"
COLLECTION_NAME = "linear_equations_pedagogy"

_memory_index: list[dict] | None = None

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _is_serverless() -> bool:
    if os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("VERCEL"):
        return True
    return str(ROOT).startswith("/var/task")


def _chroma_path() -> Path:
    if custom := os.getenv("CHROMA_PATH"):
        return Path(custom)
    if _is_serverless():
        return Path(os.getenv("TMPDIR", "/tmp")) / "chroma_db"
    return BUNDLED_CHROMA_PATH


def _use_memory_backend() -> bool:
    backend = os.getenv("RAG_BACKEND", "auto").lower()
    if backend == "memory":
        return True
    if backend == "chroma":
        return False
    return _is_serverless() and EMBEDDINGS_CACHE_PATH.exists()


def _ensure_chroma_path() -> Path:
    path = _chroma_path()
    if path == BUNDLED_CHROMA_PATH:
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not path.exists() and BUNDLED_CHROMA_PATH.exists():
        shutil.copytree(BUNDLED_CHROMA_PATH, path)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def load_knowledge_chunks() -> list[dict]:
    with KNOWLEDGE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def chunk_document_text(title: str, content: str) -> str:
    return f"{title}\n\n{content}"


def _write_embeddings_cache(
    chunks: list[dict],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    payload = {
        "model": EMBEDDING_MODEL,
        "chunks": [
            {
                "id": chunk["id"],
                "title": meta["title"],
                "section": meta["section"],
                "tags": meta["tags"],
                "content": doc,
                "embedding": emb,
            }
            for chunk, doc, emb, meta in zip(chunks, documents, embeddings, metadatas, strict=True)
        ],
    }
    EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EMBEDDINGS_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_memory_index() -> list[dict]:
    global _memory_index
    if _memory_index is not None:
        return _memory_index
    if not EMBEDDINGS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBEDDINGS_CACHE_PATH.name}. Run: python scripts/ingest.py"
        )
    with EMBEDDINGS_CACHE_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    _memory_index = data["chunks"]
    return _memory_index


def get_collection(client: chromadb.ClientAPI | None = None):
    if client is None:
        chroma_path = _ensure_chroma_path()
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_knowledge(force: bool = False) -> int:
    """Embed paper chunks into ChromaDB. Returns number of chunks indexed."""
    collection = get_collection()
    existing = collection.count()
    if existing > 0 and not force:
        return existing

    if existing > 0 and force:
        chroma_path = _ensure_chroma_path()
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        client.delete_collection(COLLECTION_NAME)
        collection = get_collection(client)

    chunks = load_knowledge_chunks()
    openai_client = get_openai_client()

    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        doc = chunk_document_text(chunk["title"], chunk["content"])
        documents.append(doc)
        ids.append(chunk["id"])
        metadatas.append(
            {
                "title": chunk["title"],
                "section": chunk["section"],
                "tags": ", ".join(chunk.get("tags", [])),
            }
        )

    embeddings = embed_texts(openai_client, documents)
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    _write_embeddings_cache(chunks, documents, embeddings, metadatas)
    return len(documents)


def _retrieve_from_memory(query: str, top_k: int) -> list[dict]:
    indexed = _load_memory_index()
    openai_client = get_openai_client()
    query_embedding = embed_texts(openai_client, [query])[0]

    scored = [
        (_cosine_similarity(query_embedding, item["embedding"]), item) for item in indexed
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    retrieved: list[dict] = []
    for similarity, item in scored[:top_k]:
        retrieved.append(
            {
                "title": item.get("title", ""),
                "section": item.get("section", ""),
                "tags": item.get("tags", ""),
                "content": item.get("content", ""),
                "distance": 1.0 - similarity,
            }
        )
    return retrieved


def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant pedagogical chunks for a student question."""
    if _use_memory_backend():
        return _retrieve_from_memory(query, top_k)

    collection = get_collection()
    if collection.count() == 0:
        ingest_knowledge()

    openai_client = get_openai_client()
    query_embedding = embed_texts(openai_client, [query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict] = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            {
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "tags": meta.get("tags", ""),
                "content": doc,
                "distance": distance,
            }
        )
    return retrieved


def indexed_chunk_count() -> int:
    if _use_memory_backend():
        return len(_load_memory_index())
    return get_collection().count()


def ensure_index_ready() -> None:
    """Load or build the vector index without requiring a writable project data dir."""
    if _use_memory_backend():
        _load_memory_index()
        return
    ingest_knowledge()


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {chunk['title']} ({chunk['section']})]\n{chunk['content']}"
        )
    return "\n\n".join(parts)
