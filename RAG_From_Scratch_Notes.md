# RAG From Scratch — Notes

**Course:** freeCodeCamp — "RAG From Scratch" by Lance Martin (LangChain)

These notes follow the structure of the course, moving from the core RAG pipeline
(Indexing → Retrieval → Generation) into advanced techniques for each stage.

---

## 1. Overview — Why RAG?

- **LLMs have not seen all data.** They are trained on public data up to a cutoff date
  and have no access to private/recent/enterprise data.
- **Context windows are growing** (thousands → millions of tokens), making it feasible
  to feed external documents directly into the prompt.
- **RAG = Retrieval Augmented Generation:** connect an LLM to external knowledge so it can
  reason over data it was never trained on.
- **Three core stages:**
  1. **Indexing** — prepare and store documents for search.
  2. **Retrieval** — fetch documents relevant to a question.
  3. **Generation** — produce an answer grounded in the retrieved documents.

```
Question ──► Retrieval ──► Relevant Docs ──► LLM (+ prompt) ──► Answer
                ▲
            Index (vector store)
```

---

## 2. Indexing

- **Goal:** convert documents into a searchable form.
- **Steps:**
  1. **Load** documents (web pages, PDFs, etc.).
  2. **Split** into chunks (documents are too large for context windows / embeddings have
     limited token capacity).
  3. **Embed** each chunk into a vector (numerical representation of meaning).
  4. **Store** vectors in a vector store / index.
- **Embeddings:** compress text into fixed-length vectors so semantically similar text
  lands near each other in vector space.
- **Search methods:**
  - **Statistical** (e.g., TF-IDF, BM25) — keyword/frequency based.
  - **Semantic / embedding** — meaning based; dominant approach for RAG.
- **Similarity metric:** typically cosine similarity / nearest-neighbor search in the
  embedding space.

---

## 3. Retrieval

- **Idea:** embed the question with the **same** embedding model used for documents,
  then find the nearest document vectors.
- **k:** choose how many neighbors (chunks) to return (e.g., k=1..N).
- Vector stores implement efficient nearest-neighbor search (e.g., kNN / ANN).
- Output: the top-k most relevant chunks for the query.

---

## 4. Generation

- **Combine** retrieved chunks + the user question into a single prompt.
- **Prompt template:** instructs the LLM to answer using only the provided context.
- The LLM reasons over the retrieved context (which now lives in its context window)
  and produces a grounded answer.
- **Key chain:** `Question → Retriever → Prompt (context + question) → LLM → Answer`.

---

## 5. Query Translation

> Improve retrieval by **rewriting / reframing the user's question** before retrieval.
> Problem: user questions are often ambiguous or poorly worded for vector search.

### 5.1 Multi-Query
- Use an LLM to rewrite the question into **several different phrasings**.
- Retrieve documents for each variant, then take the **unique union** of results.
- Increases the chance of hitting relevant documents from different angles.

### 5.2 RAG-Fusion
- Like Multi-Query, but **rank** the combined results using
  **Reciprocal Rank Fusion (RRF)**.
- RRF rewards documents that appear high across multiple query result lists.

### 5.3 Decomposition
- Break a complex question into **sub-questions**.
- Solve sub-questions sequentially (using prior answers) or independently, then
  synthesize a final answer.
- Good for multi-hop / multi-part reasoning. (Related work: Least-to-Most prompting, IR-CoT.)

### 5.4 Step-Back Prompting
- Generate a more **abstract / general "step-back" question** from the specific one.
- Retrieve for both the original and the step-back question to get higher-level context.

### 5.5 HyDE (Hypothetical Document Embeddings)
- Use the LLM to generate a **hypothetical answer/document** for the question.
- Embed that hypothetical document (not the raw question) and use it to retrieve.
- Rationale: a full pseudo-document is often closer in embedding space to real
  documents than a short question is.

---

## 6. Routing

> Direct the question to the **right data source or prompt**.

### 6.1 Logical Routing
- Let the LLM **choose** among predefined data sources (e.g., docs DB vs. SQL DB vs. graph).
- Often uses **function/structured output** so the LLM returns a typed choice.

### 6.2 Semantic Routing
- Embed the question and embed candidate **prompts**.
- Route to whichever prompt is most semantically similar (nearest by cosine similarity).

---

## 7. Query Construction

> Translate natural language into a **structured query** for the backing store.

- **Text-to-metadata-filter:** convert a question into a vector store query **plus
  metadata filters** (e.g., date ranges, view counts, categories).
- **Text-to-SQL:** for relational databases.
- **Text-to-Cypher:** for graph databases.
- Typically implemented with LLM **function calling** to emit a validated query schema.

---

## 8. Advanced Indexing

### 8.1 Multi-Representation Indexing
- **Decouple** what you embed from what you return.
- Index a **summary** (or proposition) of a document for retrieval, but return the
  **full raw document** to the LLM.
- Pattern: "Proposition Indexing" — store concise representations optimized for search.

### 8.2 RAPTOR (Recursive Abstractive Processing)
- Build a **hierarchical tree** of summaries:
  - Cluster documents → summarize each cluster → cluster the summaries → summarize again.
- Captures information at multiple levels of abstraction.
- Good when answers require both fine details and high-level synthesis across many docs.

### 8.3 ColBERT (Token-Level / Late Interaction)
- Instead of one vector per document, produce a **vector per token**.
- At query time, compute fine-grained **token-to-token** similarity (late interaction).
- Higher retrieval quality at the cost of more storage/compute.

---

## 9. Advanced Retrieval

### 9.1 Re-Ranking
- Retrieve a larger candidate set, then **re-rank** with a stronger model
  (e.g., RRF, Cohere re-rank, cross-encoder) to surface the best chunks.

### 9.2 CRAG (Corrective RAG)
- **Grade** retrieved documents for relevance.
  - If relevant → proceed to generation.
  - If ambiguous/incorrect → **fall back to web search** (or other source) and retrieve again.
- Adds a self-correcting loop before generation.

---

## 10. Advanced Generation — Active RAG

> The LLM **actively decides** when/what to retrieve and judges its own output.

### 10.1 Self-RAG
- The model grades:
  - **Relevance** of retrieved docs.
  - Whether the generation is **grounded** in the docs (no hallucination).
  - Whether the answer is **useful** for the question.
- If checks fail → **re-retrieve** or **re-generate**.

### 10.2 Flow / State Machines (LangGraph)
- Implement these feedback loops (grade → retrieve → generate → check) as a
  **graph / state machine** rather than a linear chain.
- Enables retries, fallbacks, and conditional branching for robust, agentic RAG.

---

## Quick Glossary

| Term | Meaning |
|------|---------|
| Embedding | Fixed-length vector capturing text meaning |
| Vector store | Index supporting nearest-neighbor search over embeddings |
| Chunk | A split piece of a document |
| k | Number of documents retrieved |
| RRF | Reciprocal Rank Fusion — combines multiple ranked lists |
| HyDE | Retrieve using a hypothetical generated document |
| RAPTOR | Recursive tree of clustered summaries |
| ColBERT | Token-level late-interaction retrieval |
| CRAG | Corrective RAG with doc grading + fallback |
| Self-RAG | Self-grading active RAG with retries |

---

## Mental Model — The Full Advanced Pipeline

```
            Query Translation        Routing       Query Construction
Question ─► (multi-query, HyDE, ─► (logical/    ─► (text-to-SQL/
            decomposition...)        semantic)       metadata/Cypher)
                                                         │
                                                         ▼
   Indexing (multi-rep, RAPTOR, ColBERT) ───────────► Retrieval
                                                         │
                                          Re-rank / CRAG (grade + fallback)
                                                         │
                                                         ▼
                          Generation ◄── Active RAG (Self-RAG, loops via LangGraph)
                                                         │
                                                         ▼
                                                      Answer
```
