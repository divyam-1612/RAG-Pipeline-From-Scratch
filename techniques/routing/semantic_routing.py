"""Routing — Semantic Routing.

Embed several candidate prompts, embed the question, and route to the prompt
whose embedding is most similar (cosine). Useful for picking a persona/template.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import config
from common import get_embeddings, get_llm
from result import TechniqueResult, print_result

PHYSICS_TEMPLATE = """You are a brilliant physics professor. You explain concepts
concisely and clearly. If you don't know something, you admit it.

Question: {query}

Answer:"""

MATH_TEMPLATE = """You are an excellent mathematician. You break problems into
steps and solve them methodically.

Question: {query}

Answer:"""

PROMPT_TEMPLATES = {"physics": PHYSICS_TEMPLATE, "math": MATH_TEMPLATE}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def route(query: str) -> str:
    embeddings = get_embeddings()
    names = list(PROMPT_TEMPLATES)
    template_embeddings = embeddings.embed_documents(
        [PROMPT_TEMPLATES[n] for n in names]
    )
    query_embedding = embeddings.embed_query(query)

    sims = [_cosine(np.array(query_embedding), np.array(te)) for te in template_embeddings]
    return names[int(np.argmax(sims))]


def run(question: str | None = None) -> TechniqueResult:
    question = question or "What is a black hole and how does it form?"
    result = TechniqueResult(title="Routing — Semantic Routing", question=question)

    chosen = route(question)
    result.add_step("Routed to prompt", chosen)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATES[chosen])
    chain = prompt | get_llm() | StrOutputParser()
    result.answer = chain.invoke({"query": question})
    return result


if __name__ == "__main__":
    print_result(run())
