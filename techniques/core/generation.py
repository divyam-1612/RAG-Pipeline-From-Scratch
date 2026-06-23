"""Core RAG — Step 3: Generation.

The complete chain: retrieve relevant context, stuff it into a prompt with
the question, and let the LLM produce a grounded answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config
from common import format_docs, get_llm, get_retriever
from result import TechniqueResult, print_result

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)


def build_chain():
    retriever = get_retriever()
    llm = get_llm()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Core RAG — Generation", question=question)
    chain = build_chain()
    result.answer = chain.invoke(question)
    return result


if __name__ == "__main__":
    print_result(run())
