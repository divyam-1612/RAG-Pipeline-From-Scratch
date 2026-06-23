"""Query Translation — Multi-Query.

Rewrite the user's question into several alternative phrasings, retrieve for
each, and take the unique union of results. This widens recall by attacking
the index from multiple angles.
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

PERSPECTIVES_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI assistant. Generate five different versions of the user's
question to retrieve relevant documents from a vector database. Provide each
alternative on a new line, with no numbering or extra text.

Original question: {question}"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:"""
)


def generate_queries():
    llm = get_llm(temperature=0.0)
    return (
        PERSPECTIVES_PROMPT
        | llm
        | StrOutputParser()
        | (lambda text: [q.strip() for q in text.split("\n") if q.strip()])
    )


def get_unique_union(doc_lists: list[list[Document]]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for docs in doc_lists:
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
    return unique


def build_retrieval_chain():
    retriever = get_retriever()
    return generate_queries() | retriever.map() | get_unique_union


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Query Translation — Multi-Query", question=question)

    retrieval_chain = build_retrieval_chain()
    docs = retrieval_chain.invoke({"question": question})
    result.add_step("Unique documents retrieved", str(len(docs)))
    for doc in docs:
        result.add_doc(doc.page_content, meta=doc.metadata)

    final_chain = (
        {"context": (lambda _: format_docs(docs)), "question": RunnablePassthrough()}
        | ANSWER_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    result.answer = final_chain.invoke(question)
    return result


if __name__ == "__main__":
    print_result(run())
