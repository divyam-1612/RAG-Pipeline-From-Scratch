"""Advanced Indexing — ColBERT (late interaction).

ColBERT embeds at the token level and scores with fine-grained token-to-token
"late interaction" rather than a single vector per document. The easiest way to
try it is the `ragatouille` package.

This module is optional: if `ragatouille` is not installed it prints setup
instructions and exits gracefully instead of failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from common import load_documents, split_documents
from result import TechniqueResult, print_result

SETUP_HINT = """ColBERT requires the optional 'ragatouille' package.

Install it with:
    pip install ragatouille

Then re-run:
    python main.py colbert

Note: ragatouille downloads a ColBERT checkpoint (~galactica/colbertv2.0) on
first use, so the first run needs network access and some disk space.
"""


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(
        title="Advanced Indexing — ColBERT (late interaction)", question=question
    )

    try:
        from ragatouille import RAGPretrainedModel
    except ImportError:
        result.add_step("ragatouille not installed", SETUP_HINT)
        result.answer = "ColBERT is optional — install `ragatouille` to enable it."
        return result

    model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    raw_docs = load_documents()
    chunks = split_documents(raw_docs, chunk_size=512, chunk_overlap=64)
    collection = [c.page_content for c in chunks]

    model.index(
        collection=collection,
        index_name="rag_from_scratch_colbert",
        max_document_length=512,
        split_documents=False,
    )

    hits = model.search(query=question, k=config.DEFAULT_K)
    for r in hits:
        result.add_doc(r["content"], label=f"score={r['score']:.2f}")
    return result


if __name__ == "__main__":
    print_result(run())
