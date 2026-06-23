"""Core RAG — Step 1: Indexing.

Load a document, split it into chunks, embed them, and store in Chroma.
Run directly to (re)build the index and print a quick summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from common import build_vectorstore, load_documents, split_documents
from result import TechniqueResult, print_result


def run(question: str | None = None) -> TechniqueResult:  # noqa: ARG001
    result = TechniqueResult(title="Core RAG — Indexing")

    docs = load_documents()
    splits = split_documents(docs)
    vectorstore = build_vectorstore(splits, collection_name="rag_from_scratch")
    count = vectorstore._collection.count()

    result.add_step("Loaded", f"{len(docs)} document(s)")
    result.add_step("Split", f"{len(splits)} chunks")
    result.add_step(
        "Indexed",
        f"{count} chunks into Chroma at '{config.CHROMA_DIR}'",
    )
    return result


if __name__ == "__main__":
    print_result(run())
