# Linear Equations RAG Tutor

A retrieval-augmented teaching assistant grounded in:

**Sandoval, I., García-Campos, M. & Sosa, L. (2023).** [*Providing Support and Examples for Teaching Linear Equations in Secondary School*](https://doi.org/10.1007/s10763-022-10283-5). *International Journal of Science and Mathematics Education*, 21, 1265–1287.

The app retrieves pedagogy from the paper (MTSK / KMT framework, episodes E1 & E2, classroom examples) and uses **OpenAI** to explain how to solve linear equations the way the observed teacher does—with guiding questions, equation setup before solving, and verification.

## Features

- **RAG** over structured chunks from your PDF (KMT1–KMT6, teaching episodes, class examples)
- **OpenAI** embeddings (`text-embedding-3-small`) + chat (`gpt-4o-mini` by default)
- **SymPy** check for bare equations (optional hint to the model)
- **Web UI** with math symbol buttons and paper example problems

## Quick start

```bash
cd /Users/allisonpeng/ragmathmodel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

python scripts/ingest.py
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings |

## API

- `GET /api/health` — status and chunk count
- `POST /api/solve` — `{ "question": "...", "top_k": 5 }`
- `POST /api/ingest?force=true` — rebuild vector index

## Project layout

```
app/           FastAPI + RAG + prompts
data/knowledge/paper_chunks.json   Pedagogy chunks from the paper
data/chroma_db/                    Vector store (created on ingest)
static/        Frontend UI
scripts/ingest.py
```

## Paper pedagogy (summary)

1. **E1** — Read & understand → state equation → then solve  
2. **E2** — Compare procedures → verify → return to context  
3. **KMT support** — Questions, implicit data, multiple representations  
4. **Common pitfalls** — *additional*, *double* vs square, parentheses

## License

Educational use. The underlying paper is © Springer Nature; cite the DOI when publishing work based on this tool.
