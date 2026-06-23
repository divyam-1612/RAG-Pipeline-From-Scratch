"""Advanced Indexing — Multi-Representation Indexing.

Decouple what you embed from what you return: embed concise LLM-generated
summaries for retrieval, but return the full original documents to the LLM.
Uses a MultiVectorRetriever with an in-memory doc store.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryByteStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from common import build_vectorstore, get_llm, load_documents, split_documents
from result import TechniqueResult, print_result

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the following document in 2-3 sentences:\n\n{doc}"
)

ID_KEY = "doc_id"


def build_retriever() -> MultiVectorRetriever:
    # Use larger parent chunks; summarize each for retrieval.
    raw_docs = load_documents()
    parents = split_documents(raw_docs, chunk_size=2000, chunk_overlap=0)

    llm = get_llm()
    summary_chain = {"doc": lambda d: d.page_content} | SUMMARY_PROMPT | llm | StrOutputParser()
    summaries = summary_chain.batch(parents, {"max_concurrency": 4})

    doc_ids = [str(uuid.uuid4()) for _ in parents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    vectorstore = build_vectorstore(
        summary_docs, collection_name="multi_representation", persist=False
    )
    store = InMemoryByteStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, byte_store=store, id_key=ID_KEY
    )
    retriever.docstore.mset(list(zip(doc_ids, parents)))
    return retriever


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(
        title="Advanced Indexing — Multi-Representation", question=question
    )

    retriever = build_retriever()

    # The vector store holds short summaries...
    sub_hits = retriever.vectorstore.similarity_search(question, k=1)
    result.add_step("Top summary hit (what we embed)", sub_hits[0].page_content)

    # ...but the retriever returns the full parent document.
    docs = retriever.invoke(question)
    result.add_step(
        "Returned full document (what the LLM sees)",
        f"length={len(docs[0].page_content)} chars",
    )
    result.add_doc(docs[0].page_content, label="full parent document", meta=docs[0].metadata)
    return result


if __name__ == "__main__":
    print_result(run())
