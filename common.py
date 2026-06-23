"""Shared building blocks used by every technique.

Keeping these here avoids duplicating loader/splitter/LLM/vector-store setup
across the individual technique scripts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

import config


# --- Models ------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> ChatOllama:
    """Return a cached chat model. Temperature 0 for deterministic demos."""
    return ChatOllama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


# --- Loading & splitting -----------------------------------------------------
def load_documents(url: str | None = None) -> list[Document]:
    """Load a web page into LangChain Documents."""
    url = url or config.SOURCE_URL
    loader = WebBaseLoader(url)
    return loader.load()


def split_documents(
    docs: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(list(docs))


# --- Vector store ------------------------------------------------------------
def build_vectorstore(
    documents: list[Document] | None = None,
    collection_name: str = "rag_from_scratch",
    persist: bool = True,
) -> Chroma:
    """Build (or load) a Chroma vector store.

    If ``documents`` is provided, a fresh in-memory/persisted collection is
    created from them. Otherwise an existing persisted collection is opened.
    """
    embeddings = get_embeddings()
    persist_dir = str(config.CHROMA_DIR) if persist else None

    if documents is None:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


@lru_cache(maxsize=1)
def default_index() -> Chroma:
    """Load the source URL, split, and build the default Chroma index.

    Cached so repeated technique calls in one process reuse the same store.
    """
    docs = load_documents()
    splits = split_documents(docs)
    return build_vectorstore(splits, collection_name="rag_from_scratch")


def get_retriever(k: int | None = None):
    return default_index().as_retriever(search_kwargs={"k": k or config.DEFAULT_K})


# --- Helpers -----------------------------------------------------------------
def format_docs(docs: Iterable[Document]) -> str:
    """Join document contents into a single context string."""
    return "\n\n".join(d.page_content for d in docs)


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")
