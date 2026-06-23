"""Routing — Logical Routing.

Use the LLM with structured output to pick the most appropriate data source
for a question (e.g., python_docs vs. js_docs vs. golang_docs).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from common import get_llm
from result import TechniqueResult, print_result


class RouteQuery(BaseModel):
    """Route a user question to the most relevant data source."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question, choose which docs would be most relevant.",
    )


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at routing a user question to the appropriate "
            "data source. Based on the programming language the question targets, "
            "route it to the relevant docs.",
        ),
        ("human", "{question}"),
    ]
)


def build_router():
    structured_llm = get_llm().with_structured_output(RouteQuery)
    return ROUTER_PROMPT | structured_llm


def run(question: str | None = None) -> TechniqueResult:
    # A routing-flavored default that clearly targets one language.
    question = question or "How do I use asyncio.gather to run coroutines in parallel?"
    result = TechniqueResult(title="Routing — Logical Routing", question=question)

    route = build_router().invoke({"question": question})
    result.add_step("Routed to data source", route.datasource)
    result.answer = f"This question would be routed to **{route.datasource}**."
    return result


if __name__ == "__main__":
    print_result(run())
