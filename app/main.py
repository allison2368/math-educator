"""FastAPI application for linear equation RAG tutor."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from app.math_utils import analyze_equation, looks_like_word_problem
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.rag import (
    ensure_index_ready,
    format_context,
    get_openai_client,
    indexed_chunk_count,
    ingest_knowledge,
    retrieve_context,
)

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="Linear Equations RAG Tutor",
    description="Pedagogy-grounded tutor based on Sandoval et al. (2023)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class SolveResponse(BaseModel):
    answer: str
    sources: list[dict]
    is_word_problem: bool
    sympy_hint: str | None = None


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    model: str


@app.on_event("startup")
def startup_ingest():
    try:
        ensure_index_ready()
    except Exception:
        # Allow server to start; /solve will surface missing API key clearly
        pass


@app.get("/api/health", response_model=HealthResponse)
def health():
    try:
        count = indexed_chunk_count()
    except Exception:
        count = 0
    return HealthResponse(
        status="ok",
        indexed_chunks=count,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


@app.post("/api/ingest")
def reingest(force: bool = True):
    try:
        count = ingest_knowledge(force=force)
        return {"status": "ok", "indexed_chunks": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chunks = retrieve_context(question, top_k=req.top_k)
        context = format_context(chunks)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {exc}") from exc

    sympy_hint = None
    if not looks_like_word_problem(question):
        sympy_hint = analyze_equation(question)

    user_prompt = build_user_prompt(question, context, sympy_hint)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        client = get_openai_client()
        completion = client.chat.completions.create(
            model=model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.choices[0].message.content or ""
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {exc}") from exc

    sources = [
        {
            "title": c["title"],
            "section": c["section"],
            "tags": c["tags"],
        }
        for c in chunks
    ]

    return SolveResponse(
        answer=answer,
        sources=sources,
        is_word_problem=looks_like_word_problem(question),
        sympy_hint=sympy_hint,
    )


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
