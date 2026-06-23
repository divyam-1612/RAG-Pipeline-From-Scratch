"""Advanced Indexing — RAPTOR (simplified).

RAPTOR builds a tree of summaries: cluster leaf chunks, summarize each cluster,
then index BOTH the leaves and the higher-level summaries. This captures detail
and high-level synthesis. This is a compact, single-level implementation using
KMeans clustering for clarity.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sklearn.cluster import KMeans

import config
from common import (
    build_vectorstore,
    get_embeddings,
    get_llm,
    load_documents,
    split_documents,
)
from result import TechniqueResult, print_result

CLUSTER_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Write a concise summary that captures the key information across the "
    "following related chunks:\n\n{context}"
)


def build_raptor_index(n_clusters: int = 4):
    raw_docs = load_documents()
    leaves = split_documents(raw_docs, chunk_size=600, chunk_overlap=60)

    embeddings = get_embeddings()
    vectors = np.array(embeddings.embed_documents([d.page_content for d in leaves]))

    n_clusters = min(n_clusters, len(leaves))
    labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(
        vectors
    )

    llm = get_llm()
    summary_chain = CLUSTER_SUMMARY_PROMPT | llm | StrOutputParser()

    summary_docs: list[Document] = []
    for cluster_id in range(n_clusters):
        members = [leaves[i].page_content for i in range(len(leaves)) if labels[i] == cluster_id]
        context = "\n\n".join(members)
        summary = summary_chain.invoke({"context": context})
        summary_docs.append(
            Document(page_content=summary, metadata={"level": "summary", "cluster": cluster_id})
        )

    # Index leaves + cluster summaries together (the RAPTOR "collapsed tree").
    all_docs = leaves + summary_docs
    vectorstore = build_vectorstore(all_docs, collection_name="raptor", persist=False)
    return vectorstore, leaves, summary_docs


def run(question: str | None = None) -> TechniqueResult:
    question = question or config.DEFAULT_QUESTION
    result = TechniqueResult(
        title="Advanced Indexing — RAPTOR (simplified)", question=question
    )

    vectorstore, leaves, summaries = build_raptor_index()
    result.add_step(
        "Index built", f"Leaves: {len(leaves)}, cluster summaries: {len(summaries)}"
    )

    hits = vectorstore.similarity_search(question, k=config.DEFAULT_K)
    for doc in hits:
        level = doc.metadata.get("level", "leaf")
        result.add_doc(doc.page_content, label=level, meta=doc.metadata)
    return result


if __name__ == "__main__":
    print_result(run())
