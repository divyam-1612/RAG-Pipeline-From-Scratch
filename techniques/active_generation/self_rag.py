"""Active Generation — Self-RAG (simplified).

Self-RAG adds self-grading loops around generation:
  1. Grade retrieved docs for relevance (re-retrieve if poor).
  2. Generate an answer.
  3. Check the answer is grounded in the docs (no hallucination).
  4. Check the answer actually addresses the question.
If checks fail, retry generation up to a small limit.

Implemented as an explicit state loop for clarity (LangGraph would formalize this).
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


class YesNo(BaseModel):
    binary_score: str = Field(..., description="'yes' or 'no'.")


RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Is the document relevant to the question? Answer 'yes' or 'no'."),
        ("human", "Document:\n{document}\n\nQuestion: {question}"),
    ]
)

GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Is the answer grounded in / supported by the facts? 'yes' or 'no'."),
        ("human", "Facts:\n{documents}\n\nAnswer:\n{generation}"),
    ]
)

USEFUL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Does the answer address the question? 'yes' or 'no'."),
        ("human", "Question: {question}\n\nAnswer:\n{generation}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:"""
)


def _yes(prompt, **kwargs) -> bool:
    grader = prompt | get_llm().with_structured_output(YesNo)
    return grader.invoke(kwargs).binary_score.strip().lower() == "yes"


def filter_relevant(question: str, docs: list[Document]) -> list[Document]:
    return [d for d in docs if _yes(RELEVANCE_PROMPT, document=d.page_content, question=question)]


def run(question: str | None = None, max_retries: int = 2) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Active Generation — Self-RAG", question=question)

    docs = get_retriever().invoke(question)
    relevant = filter_relevant(question, docs) or docs
    result.add_step("Relevant docs after grading", f"{len(relevant)}/{len(docs)}")
    for doc in relevant:
        result.add_doc(doc.page_content, meta=doc.metadata)

    answer_chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
    context = format_docs(relevant)

    answer = ""
    for attempt in range(1, max_retries + 1):
        answer = answer_chain.invoke({"context": context, "question": question})
        grounded = _yes(GROUNDED_PROMPT, documents=context, generation=answer)
        useful = _yes(USEFUL_PROMPT, question=question, generation=answer)
        result.add_step(f"Attempt {attempt}", f"grounded={grounded}, useful={useful}")
        if grounded and useful:
            result.answer = answer
            return result

    result.add_step("Note", "Checks not fully passed; returning best-effort answer.")
    result.answer = answer
    return result


if __name__ == "__main__":
    print_result(run())
