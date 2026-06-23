"""Query Construction — Text to Metadata Filter.

Translate a natural-language question into a structured search query plus
metadata filters (the LLM emits a typed schema via structured output). This
mirrors how you'd build filters over a vector store with metadata such as
publish dates, view counts, or lengths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from common import get_llm
from result import TechniqueResult, print_result


class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos."""

    content_search: str = Field(
        ..., description="Similarity search query applied to video transcripts."
    )
    title_search: str = Field(
        ...,
        description="Short keyword query for the video title.",
    )
    min_view_count: Optional[int] = Field(
        None, description="Minimum view count filter, inclusive."
    )
    max_length_sec: Optional[int] = Field(
        None, description="Maximum video length in seconds, inclusive."
    )
    earliest_publish_date: Optional[str] = Field(
        None, description="Earliest publish date (YYYY-MM-DD), inclusive."
    )
    latest_publish_date: Optional[str] = Field(
        None, description="Latest publish date (YYYY-MM-DD), inclusive."
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You convert user questions into a structured database query. "
            "Only fill filters that the question clearly implies; leave others null.",
        ),
        ("human", "{question}"),
    ]
)


def build_query_constructor():
    structured_llm = get_llm().with_structured_output(TutorialSearch)
    return PROMPT | structured_llm


def run(question: str | None = None) -> TechniqueResult:
    question = (
        question
        or "videos on chat langchain published in 2023 with at least 5000 views and under 5 minutes"
    )
    result = TechniqueResult(title="Query Construction — Metadata Filter", question=question)

    structured = build_query_constructor().invoke({"question": question})
    payload = structured.model_dump_json(indent=2)
    result.add_step("Structured query", payload)
    result.answer = f"```json\n{payload}\n```"
    return result


if __name__ == "__main__":
    print_result(run())
