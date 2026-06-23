"""CLI dispatcher for the RAG From Scratch project.

Usage:
    python main.py list
    python main.py <technique> [--question "..."]

Examples:
    python main.py core
    python main.py multi_query --question "How does task decomposition work?"
"""

from __future__ import annotations

import argparse
import importlib
from typing import Callable

from result import TechniqueResult, print_result

# technique name -> (module path, callable name)
TECHNIQUES: dict[str, tuple[str, str]] = {
    # Core pipeline
    "indexing": ("techniques.core.indexing", "run"),
    "retrieval": ("techniques.core.retrieval", "run"),
    "generation": ("techniques.core.generation", "run"),
    # Query translation
    "multi_query": ("techniques.query_translation.multi_query", "run"),
    "rag_fusion": ("techniques.query_translation.rag_fusion", "run"),
    "decomposition": ("techniques.query_translation.decomposition", "run"),
    "step_back": ("techniques.query_translation.step_back", "run"),
    "hyde": ("techniques.query_translation.hyde", "run"),
    # Routing
    "logical_routing": ("techniques.routing.logical_routing", "run"),
    "semantic_routing": ("techniques.routing.semantic_routing", "run"),
    # Query construction
    "query_construction": ("techniques.query_construction.metadata_filter", "run"),
    # Advanced indexing
    "multi_representation": ("techniques.advanced_indexing.multi_representation", "run"),
    "raptor": ("techniques.advanced_indexing.raptor", "run"),
    "colbert": ("techniques.advanced_indexing.colbert", "run"),
    # Advanced retrieval
    "reranking": ("techniques.advanced_retrieval.reranking", "run"),
    "crag": ("techniques.advanced_retrieval.crag", "run"),
    # Active generation
    "self_rag": ("techniques.active_generation.self_rag", "run"),
}


def _load(technique: str) -> Callable:
    module_path, func_name = TECHNIQUES[technique]
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def run_core(question: str | None) -> list[TechniqueResult]:
    """Convenience: run the full core pipeline end to end."""
    return [_load(name)(question) for name in ("indexing", "retrieval", "generation")]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG From Scratch — technique runner")
    parser.add_argument(
        "technique",
        help="Technique to run, 'core' for the full core pipeline, or 'list'.",
    )
    parser.add_argument("--question", "-q", default=None, help="Custom question.")
    args = parser.parse_args()

    if args.technique == "list":
        print("Available techniques:\n")
        print("  core   (runs indexing -> retrieval -> generation)")
        for name in TECHNIQUES:
            print(f"  {name}")
        return

    if args.technique == "core":
        for result in run_core(args.question):
            print_result(result)
        return

    if args.technique not in TECHNIQUES:
        print(f"Unknown technique: {args.technique!r}. Try 'python main.py list'.")
        raise SystemExit(1)

    print_result(_load(args.technique)(args.question))


if __name__ == "__main__":
    main()
