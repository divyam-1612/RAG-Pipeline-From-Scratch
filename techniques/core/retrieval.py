"""Core RAG — Step 2: Retrieval.

Embed the question with the same model used for indexing and fetch the
nearest chunks from Chroma.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from common import get_retriever
from result import TechniqueResult, print_result


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Core RAG — Retrieval", question=question)

    retriever = get_retriever()
    docs = retriever.invoke(question)
    for doc in docs:
        result.add_doc(doc.page_content, meta=doc.metadata)
    return result


if __name__ == "__main__":
    print_result(run())
