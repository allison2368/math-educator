#!/usr/bin/env python3
"""Ingest paper knowledge into ChromaDB."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import ingest_knowledge


def main():
    parser = argparse.ArgumentParser(description="Ingest pedagogy chunks into vector DB")
    parser.add_argument("--force", action="store_true", help="Re-index even if data exists")
    args = parser.parse_args()
    count = ingest_knowledge(force=args.force)
    print(f"Indexed {count} chunks into ChromaDB.")


if __name__ == "__main__":
    main()
