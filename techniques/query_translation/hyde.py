"""Query Translation — HyDE (Hypothetical Document Embeddings).

Ask the LLM to write a hypothetical passage that would answer the question,
then retrieve using that passage instead of the raw question. A full passage
often sits closer to real documents in embedding space than a short query.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from common import default_index, format_docs, get_llm
from result import TechniqueResult, print_result

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """Write a short, factual passage that would answer the question below.
Do not say you are unsure; write it as if it came from a reference document.

Question: {question}

Passage:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.

Context:
{context}

Question: {question}

Answer:"""
)


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(title="Query Translation — HyDE", question=question)

    llm = get_llm()
    hyde_doc = (HYDE_PROMPT | llm | StrOutputParser()).invoke({"question": question})
    result.add_step("Hypothetical document", hyde_doc)

    # Retrieve using the hypothetical document as the query.
    docs = default_index().similarity_search(hyde_doc, k=config.DEFAULT_K)
    for doc in docs:
        result.add_doc(doc.page_content, meta=doc.metadata)

    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    result.answer = answer_chain.invoke({"context": format_docs(docs), "question": question})
    return result


if __name__ == "__main__":
    print_result(run())
