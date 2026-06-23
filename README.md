# RAG From Scratch — Local (Ollama + Chroma)

A hands-on implementation of the freeCodeCamp **"RAG From Scratch"** course
(by Lance Martin / LangChain), runnable **fully locally** with **Ollama** and **Chroma**.

Each technique lives in its own file so you can read and run them independently.

> Companion notes: [../RAG_From_Scratch_Notes.md](../RAG_From_Scratch_Notes.md)

---

## 1. Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running — https://ollama.com
3. Pull the models used by default:

   ```bash
   ollama pull llama3.2          # chat / generation model
   ollama pull nomic-embed-text  # embedding model
   ```

   (You can change these in `config.py` or via environment variables.)

---

## 2. Setup

```bash
cd RAG_Pipeline/rag_from_scratch
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and tweak model names / source URL.

---

## 3. Run

A single CLI runs any technique:

```bash
python main.py list                 # list available techniques
python main.py core                 # core pipeline (index -> retrieve -> generate)
python main.py multi_query
python main.py rag_fusion
python main.py hyde
python main.py decomposition
python main.py step_back
python main.py logical_routing
python main.py semantic_routing
python main.py query_construction
python main.py multi_representation
python main.py raptor
python main.py reranking
python main.py crag
python main.py self_rag
```

Pass a custom question:

```bash
python main.py multi_query --question "How does task decomposition work for agents?"
```

Or run a module directly:

```bash
python -m techniques.query_translation.hyde
```

### Web UI (Streamlit)

Prefer a UI? Launch the Streamlit app:

```bash
streamlit run app.py
```

It opens in your browser with a category/technique dropdown, a question box,
live config + LangSmith status in the sidebar, and the captured output for each
run. (Requires `ollama serve` to be running.)

---

## 4. Project Structure

```
rag_from_scratch/
├── README.md
├── requirements.txt
├── .env.example
├── config.py                 # model names, source URL, paths
├── common.py                 # shared: loader, splitter, llm, embeddings, vectorstore
├── main.py                   # CLI dispatcher
└── techniques/
    ├── core/
    │   ├── indexing.py
    │   ├── retrieval.py
    │   └── generation.py
    ├── query_translation/
    │   ├── multi_query.py
    │   ├── rag_fusion.py
    │   ├── decomposition.py
    │   ├── step_back.py
    │   └── hyde.py
    ├── routing/
    │   ├── logical_routing.py
    │   └── semantic_routing.py
    ├── query_construction/
    │   └── metadata_filter.py
    ├── advanced_indexing/
    │   ├── multi_representation.py
    │   ├── raptor.py
    │   └── colbert.py
    ├── advanced_retrieval/
    │   ├── reranking.py
    │   └── crag.py
    └── active_generation/
        └── self_rag.py
```

---

## 5. Topic → File Map

| Course Topic | File |
|--------------|------|
| Indexing | `techniques/core/indexing.py` |
| Retrieval | `techniques/core/retrieval.py` |
| Generation | `techniques/core/generation.py` |
| Multi-Query | `techniques/query_translation/multi_query.py` |
| RAG-Fusion | `techniques/query_translation/rag_fusion.py` |
| Decomposition | `techniques/query_translation/decomposition.py` |
| Step-Back | `techniques/query_translation/step_back.py` |
| HyDE | `techniques/query_translation/hyde.py` |
| Logical Routing | `techniques/routing/logical_routing.py` |
| Semantic Routing | `techniques/routing/semantic_routing.py` |
| Query Construction | `techniques/query_construction/metadata_filter.py` |
| Multi-Representation Indexing | `techniques/advanced_indexing/multi_representation.py` |
| RAPTOR | `techniques/advanced_indexing/raptor.py` |
| ColBERT | `techniques/advanced_indexing/colbert.py` |
| Re-ranking | `techniques/advanced_retrieval/reranking.py` |
| CRAG | `techniques/advanced_retrieval/crag.py` |
| Self-RAG | `techniques/active_generation/self_rag.py` |

---

## 6. Tracing with LangSmith (see every LLM hit)

Because every technique is built on LangChain runnables, you can see all LLM
calls, retrievals, and chain steps in [LangSmith](https://smith.langchain.com)
with **no code changes** — just set environment variables.

1. Create an API key in LangSmith (Settings → API Keys).
2. Copy `.env.example` to `.env` and fill in:

   ```bash
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_PROJECT=rag-from-scratch
   ```

3. Run any technique as usual:

   ```bash
   python main.py multi_query -q "How does task decomposition work?"
   ```

4. Open the **rag-from-scratch** project in LangSmith. Each run appears as a
   trace tree: the chain, every sub-LLM call (e.g., the 5 multi-query rewrites),
   the retriever calls, token usage, latency, and inputs/outputs at each step.

`config.py` loads `.env` automatically before any LangChain call, so the
variables above are picked up with no extra wiring. To turn tracing off, set
`LANGSMITH_TRACING=false` or remove the variables.

> Tip: change `LANGSMITH_PROJECT` per experiment to keep traces grouped.

---

## 7. Notes

- **Fully local:** no API keys required. Everything runs through Ollama.
- **ColBERT** (`colbert.py`) optionally uses `ragatouille`; it prints setup
  instructions and falls back gracefully if the package isn't installed.
- First run downloads/loads the source web page and builds a Chroma index in
  `./.chroma/`. Delete that folder to rebuild from scratch.
