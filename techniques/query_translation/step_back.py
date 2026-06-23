"""Query Translation — Step-Back Prompting.

Generate a more general "step-back" question, retrieve context for BOTH the
step-back and original questions, then answer with the broader context.
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

STEP_BACK_PROMPT = ChatPromptTemplate.from_template(
    """Your task is to step back and paraphrase the question into a more
generic, higher-level question that is easier to answer.
Output only the step-back question.

Question: {question}"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using the context below, which was gathered from a
broad (step-back) and a specific search.

Broad context:
{step_back_context}

Specific context:
{normal_context}

Question: {question}

Answer:"""
)


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Query Translation — Step-Back", question=question)

    llm = get_llm()
    retriever = get_retriever()

    step_back_question = (STEP_BACK_PROMPT | llm | StrOutputParser()).invoke(
        {"question": question}
    ).strip()
    result.add_step("Step-back question", step_back_question)

    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    result.answer = answer_chain.invoke(
        {
            "question": question,
            "normal_context": format_docs(retriever.invoke(question)),
            "step_back_context": format_docs(retriever.invoke(step_back_question)),
        }
    )
    return result


if __name__ == "__main__":
    print_result(run())
