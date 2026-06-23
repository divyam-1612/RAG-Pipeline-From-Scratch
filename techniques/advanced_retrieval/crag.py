"""Advanced Retrieval — CRAG (Corrective RAG, simplified).

Grade retrieved documents for relevance. If enough are relevant, answer from
them. If not, fall back to a web search (DuckDuckGo) and answer from that.
This implements the core corrective loop without a full graph framework.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from common import format_docs, get_llm, get_retriever
from result import TechniqueResult, print_result


class GradeDocuments(BaseModel):
    binary_score: str = Field(..., description="'yes' if relevant, else 'no'.")


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a document is relevant to a question. "
            "Answer 'yes' or 'no'.",
        ),
        ("human", "Document:\n{document}\n\nQuestion: {question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:"""
)


def grade_documents(question: str, docs: list[Document]) -> list[Document]:
    grader = GRADE_PROMPT | get_llm().with_structured_output(GradeDocuments)
    relevant: list[Document] = []
    for doc in docs:
        result = grader.invoke({"question": question, "document": doc.page_content})
        if result.binary_score.strip().lower() == "yes":
            relevant.append(doc)
    return relevant


def web_search(question: str) -> list[Document]:
    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        raw = DuckDuckGoSearchResults().invoke(question)
        return [Document(page_content=raw)]
    except Exception as exc:  # network or package issue
        return [Document(page_content=f"(web search unavailable: {exc})")]


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(
        title="Advanced Retrieval — CRAG (Corrective RAG)", question=question
    )

    docs = get_retriever().invoke(question)
    relevant = grade_documents(question, docs)
    result.add_step("Relevance grading", f"{len(relevant)}/{len(docs)} docs relevant")

    if relevant:
        context_docs = relevant
        result.add_step("Decision", "Answer from indexed documents.")
    else:
        result.add_step("Decision", "Documents insufficient -> falling back to web search.")
        context_docs = web_search(question)

    for doc in context_docs:
        result.add_doc(doc.page_content, meta=doc.metadata)

    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
    result.answer = chain.invoke({"context": format_docs(context_docs), "question": question})
    return result


if __name__ == "__main__":
    print_result(run())
