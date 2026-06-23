"""Streamlit UI for the RAG From Scratch project.

Pick a technique, ask a question, and see the result. Each technique's `run`
function prints its output, so we capture stdout and render it in the app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import importlib
import time

import streamlit as st

import config
from main import TECHNIQUES, run_core
from result import TechniqueResult

# --- Technique grouping for a friendlier menu --------------------------------
GROUPS: dict[str, list[str]] = {
    "Core pipeline": ["core", "indexing", "retrieval", "generation"],
    "Query translation": [
        "multi_query",
        "rag_fusion",
        "decomposition",
        "step_back",
        "hyde",
    ],
    "Routing": ["logical_routing", "semantic_routing"],
    "Query construction": ["query_construction"],
    "Advanced indexing": ["multi_representation", "raptor", "colbert"],
    "Advanced retrieval": ["reranking", "crag"],
    "Active generation": ["self_rag"],
}

DESCRIPTIONS: dict[str, str] = {
    "core": "Full pipeline: index -> retrieve -> generate.",
    "indexing": "Load, split, embed, and store documents in Chroma.",
    "retrieval": "Embed the question and fetch nearest chunks.",
    "generation": "Retrieve context and answer with the LLM.",
    "multi_query": "Rewrite the question 5 ways, union the results.",
    "rag_fusion": "Multi-query + Reciprocal Rank Fusion.",
    "decomposition": "Break into sub-questions, solve, synthesize.",
    "step_back": "Ask a broader question, retrieve both levels.",
    "hyde": "Retrieve using a hypothetical generated answer.",
    "logical_routing": "LLM picks the right data source (structured output).",
    "semantic_routing": "Route to the closest prompt by embedding similarity.",
    "query_construction": "Turn a question into a structured metadata filter.",
    "multi_representation": "Embed summaries, return full documents.",
    "raptor": "Cluster + summarize chunks into a hierarchy.",
    "colbert": "Token-level late-interaction retrieval (optional).",
    "reranking": "Over-retrieve, then LLM re-ranks the best chunks.",
    "crag": "Grade docs; fall back to web search if weak.",
    "self_rag": "Self-grade answers for grounding + usefulness, retry.",
}


def run_technique(technique: str, question: str | None) -> list[TechniqueResult]:
    """Run a technique and return its structured result(s)."""
    if technique == "core":
        return run_core(question)
    module_path, func_name = TECHNIQUES[technique]
    module = importlib.import_module(module_path)
    return [getattr(module, func_name)(question)]


def render_result(result: TechniqueResult) -> None:
    st.subheader(result.title)
    if result.question:
        st.caption(f"Question: {result.question}")

    if result.steps:
        st.markdown("**Steps**")
        for heading, body in result.steps:
            with st.expander(heading, expanded=False):
                if "\n" in body or len(body) > 120:
                    st.code(body, language="text")
                else:
                    st.write(body)

    if result.answer:
        st.markdown("**Answer**")
        st.markdown(result.answer)

    if result.documents:
        st.markdown(f"**Retrieved documents ({len(result.documents)})**")
        for i, doc in enumerate(result.documents, start=1):
            label = f" — {doc.label}" if doc.label else ""
            with st.expander(f"Document {i}{label}", expanded=False):
                if doc.meta:
                    st.caption(", ".join(f"{k}={v}" for k, v in doc.meta.items()))
                st.write(doc.content)


# --- Page setup --------------------------------------------------------------
st.set_page_config(page_title="RAG From Scratch", page_icon="🔎", layout="wide")
st.title("🔎 RAG From Scratch")
st.caption("Local RAG techniques with Ollama + Chroma — based on the freeCodeCamp course.")

# --- Sidebar: config + status ------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    st.markdown(
        f"""
- **Chat model:** `{config.CHAT_MODEL}`
- **Embedding model:** `{config.EMBED_MODEL}`
- **Ollama:** `{config.OLLAMA_BASE_URL}`
- **Source URL:** {config.SOURCE_URL}
- **Chunk size / overlap:** {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}
- **Retrieval k:** {config.DEFAULT_K}
"""
    )

    import os

    tracing_on = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    if tracing_on:
        project = os.getenv("LANGSMITH_PROJECT", "default")
        st.success(f"LangSmith tracing ON → project '{project}'")
        st.markdown("[Open LangSmith](https://smith.langchain.com)")
    else:
        st.info("LangSmith tracing OFF (set LANGSMITH_TRACING=true in .env)")

    st.divider()
    st.caption("Models run locally via Ollama. Make sure `ollama serve` is running.")

# --- Main controls -----------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    group = st.selectbox("Category", list(GROUPS.keys()))
    technique = st.selectbox(
        "Technique",
        GROUPS[group],
        format_func=lambda t: t.replace("_", " ").title(),
    )
    st.caption(DESCRIPTIONS.get(technique, ""))

with col_right:
    question = st.text_area(
        "Question",
        value=config.DEFAULT_QUESTION,
        height=100,
        help="Some techniques (routing, query construction) ignore this and use "
        "their own illustrative example.",
    )
    run_clicked = st.button("Run", type="primary", use_container_width=True)

# --- Run ---------------------------------------------------------------------
if run_clicked:
    q = question.strip() or None
    with st.spinner(f"Running '{technique}' with {config.CHAT_MODEL}..."):
        start = time.time()
        try:
            results = run_technique(technique, q)
            elapsed = time.time() - start
        except Exception as exc:  # surface errors in the UI instead of crashing
            st.error(f"Error while running '{technique}': {exc}")
            st.exception(exc)
        else:
            st.success(f"Done in {elapsed:.1f}s")
            for res in results:
                render_result(res)
                st.divider()

st.divider()
st.caption(
    "Tip: run techniques like multi_query or crag with LangSmith tracing on to "
    "see every individual LLM call."
)
