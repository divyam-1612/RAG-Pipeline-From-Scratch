"""Query Translation — RAG-Fusion.

Like Multi-Query, but instead of a plain union we combine the ranked result
lists with Reciprocal Rank Fusion (RRF), rewarding documents that rank highly
across multiple query variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config
from common import format_docs, get_llm, get_retriever
from result import TechniqueResult, print_result

QUERIES_PROMPT = ChatPromptTemplate.from_template(
    """Generate four search queries related to the question below.
Output each query on its own line with no numbering.

Question: {question}"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:"""
)


def reciprocal_rank_fusion(
    results: list[list[Document]], k: int = 60
) -> list[Document]:
    """Combine ranked lists. Score = sum(1 / (k + rank)) across lists."""
    scores: dict[str, float] = {}
    lookup: dict[str, Document] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            lookup[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [lookup[key] for key, _ in ranked]


def generate_queries():
    return (
        QUERIES_PROMPT
        | get_llm()
        | StrOutputParser()
        | (lambda text: [q.strip() for q in text.split("\n") if q.strip()])
    )


def build_retrieval_chain():
    retriever = get_retriever()
    return generate_queries() | retriever.map() | reciprocal_rank_fusion


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Query Translation — RAG-Fusion", question=question)

    docs = build_retrieval_chain().invoke({"question": question})
    result.add_step("Fused documents", f"{len(docs)} (top-ranked first)")

    top_docs = docs[: config.DEFAULT_K]
    for doc in top_docs:
        result.add_doc(doc.page_content, meta=doc.metadata)

    final_chain = (
        {"context": (lambda _: format_docs(top_docs)), "question": RunnablePassthrough()}
        | ANSWER_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    result.answer = final_chain.invoke(question)
    return result


if __name__ == "__main__":
    print_result(run())
