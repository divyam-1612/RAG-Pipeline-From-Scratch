"""Advanced Retrieval — Re-ranking.

Over-retrieve a larger candidate set, then re-rank it with an LLM grader that
scores each document's relevance to the question. Return the best few. This is
a portable, dependency-free alternative to a hosted re-ranker (e.g., Cohere).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from common import format_docs, get_llm, get_retriever
from result import TechniqueResult, print_result
from langchain_core.output_parsers import StrOutputParser

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:"""
)


class Relevance(BaseModel):
    score: float = Field(..., description="Relevance score from 0 (irrelevant) to 10.")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Score how relevant the document is to the question, 0-10."),
        ("human", "Question: {question}\n\nDocument:\n{document}"),
    ]
)


def rerank(question: str, docs: list[Document], top_n: int) -> list[Document]:
    grader = GRADE_PROMPT | get_llm().with_structured_output(Relevance)
    scored: list[tuple[float, Document]] = []
    for doc in docs:
        result = grader.invoke({"question": question, "document": doc.page_content})
        scored.append((result.score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Advanced Retrieval — Re-ranking", question=question)

    # Over-retrieve, then re-rank down to DEFAULT_K.
    candidates = get_retriever(k=config.DEFAULT_K * 3).invoke(question)
    result.add_step("Candidates retrieved", str(len(candidates)))

    top = rerank(question, candidates, top_n=config.DEFAULT_K)
    result.add_step("Kept after re-ranking", str(len(top)))
    for doc in top:
        result.add_doc(doc.page_content, meta=doc.metadata)

    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
    result.answer = chain.invoke({"context": format_docs(top), "question": question})
    return result


if __name__ == "__main__":
    print_result(run())
