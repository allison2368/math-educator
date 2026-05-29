"""RAG pipeline for linear equation pedagogy from Sandoval et al. (2023)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_PATH = ROOT / "data" / "knowledge" / "paper_chunks.json"
CHROMA_PATH = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "linear_equations_pedagogy"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


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


def get_collection(client: chromadb.ClientAPI | None = None):
    if client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
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
        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
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
    return len(documents)


def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant pedagogical chunks for a student question."""
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


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {chunk['title']} ({chunk['section']})]\n{chunk['content']}"
        )
    return "\n\n".join(parts)
