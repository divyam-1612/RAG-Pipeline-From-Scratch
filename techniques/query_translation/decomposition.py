"""Query Translation — Decomposition.

Break a complex question into sub-questions, answer each one using retrieval,
then synthesize the partial answers into a final response (least-to-most style).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from common import format_docs, get_llm, get_retriever
from result import TechniqueResult, print_result

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """Decompose the following question into 3 simpler sub-questions that, when
answered in order, help answer the original. Output each sub-question on its
own line with no numbering.

Question: {question}"""
)

SUBANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Here is the question to answer:
{sub_question}

Use the retrieved context and any prior Q&A pairs.

Context:
{context}

Prior Q&A:
{qa_pairs}

Answer:"""
)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """Using the sub-question answers below, write a complete answer to the
original question.

Original question: {question}

Sub-question answers:
{qa_pairs}

Final answer:"""
)


def decompose(question: str) -> list[str]:
    chain = DECOMPOSE_PROMPT | get_llm() | StrOutputParser()
    text = chain.invoke({"question": question})
    return [q.strip() for q in text.split("\n") if q.strip()]


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Query Translation — Decomposition", question=question)

    retriever = get_retriever()
    llm = get_llm()

    sub_questions = decompose(question)
    result.add_step("Sub-questions", "\n".join(f"- {sq}" for sq in sub_questions))

    qa_pairs = ""
    sub_chain = SUBANSWER_PROMPT | llm | StrOutputParser()
    for sq in sub_questions:
        context = format_docs(retriever.invoke(sq))
        answer = sub_chain.invoke(
            {"sub_question": sq, "context": context, "qa_pairs": qa_pairs or "(none)"}
        )
        qa_pairs += f"\nQ: {sq}\nA: {answer}\n"
        result.add_step(f"Sub-answer: {sq}", answer)

    synthesis = SYNTHESIS_PROMPT | llm | StrOutputParser()
    result.answer = synthesis.invoke({"question": question, "qa_pairs": qa_pairs})
    return result


if __name__ == "__main__":
    print_result(run())
