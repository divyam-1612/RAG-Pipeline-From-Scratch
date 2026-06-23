"""Central configuration for the RAG From Scratch project.

All values can be overridden via environment variables (see .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional at runtime
    pass


# --- Models (Ollama) ---------------------------------------------------------
CHAT_MODEL: str = os.getenv("RAG_CHAT_MODEL", "llama3.2")
EMBED_MODEL: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Data source -------------------------------------------------------------
SOURCE_URL: str = os.getenv(
    "RAG_SOURCE_URL",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
)

# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# --- Storage -----------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CHROMA_DIR: Path = PROJECT_ROOT / os.getenv("RAG_CHROMA_DIR", ".chroma")

# --- Retrieval defaults ------------------------------------------------------
DEFAULT_K: int = int(os.getenv("RAG_DEFAULT_K", "4"))

# --- Demo question -----------------------------------------------------------
DEFAULT_QUESTION: str = os.getenv(
    "RAG_DEFAULT_QUESTION",
    "What is task decomposition for LLM agents?",
)
