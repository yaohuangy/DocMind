# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
python run.py                           # or: streamlit run app.py --server.address 0.0.0.0 --server.port 7860

# Tests
pytest tests/ -v                        # Run all unit tests
pytest tests/ -v --ignore=tests/e2e     # Skip e2e (CI mode)
pytest tests/test_retrieval/test_fusion.py -v  # Single test file

# Lint & type check
ruff check src/                         # Code style (line-length 120, Python 3.10+)
mypy src/                               # Static type checking

# Evaluation
python scripts/generate_eval_dataset.py --num 30 --rewrite --expand-gt
python scripts/run_evaluation.py --dataset data/evaluation/dataset_xxx.json -u <用户名>
python scripts/run_evaluation.py --dataset data/evaluation/dataset_xxx.json -u <用户名> --with-generation  # includes Token cost + RAGAS

# Docker
docker-compose up -d                    # Full stack (Streamlit + Neo4j)
docker-compose up -d docmind            # App only, no Neo4j
```

## Architecture

**Docmind** is a RAG-based document QA system. The stack is **Streamlit** (UI) → **QAEngine** (orchestration) → **core services** (LLM, embeddings, vector store, graph store, metadata store).

### Layer Map

```
app.py + pages/*.py          # Streamlit UI (5 pages, zero HTML/CSS/JS)
       │
src/engine/qa_engine.py      # Facade — the one class the UI touches
       │
src/engine/session.py        # @st.cache_resource singleton factory for QAEngine
       │
  ┌────┴────┬────────┬──────────┐
  │         │        │          │
ingestion  retrieval generation memory
```

### Key Design Patterns

- **Singleton config**: `src/core/config.py` — `load_config()` returns a cached `Settings` dataclass. All env vars prefixed (`LLM_*`, `EMBEDDING_*`, `CHROMA_*`, `NEO4J_*`, `SQLITE_*`, `RETRIEVAL_*`, `CHUNK_*`). Read from `.env` at the project root (3 levels up from `config.py`).
- **Dispatcher pattern for parsers**: 8 format parsers under `src/ingestion/parsers/` each implement a `BaseParser` interface. `MultiFormatLoader` dispatches by file extension / URL scheme.
- **Retrieval strategies**: 4 retrievers all implement `BaseRetriever.retrieve()`. MQE+HyDE uses `asyncio.gather` to run both LLM calls in parallel, then `weighted_merge()` (0.4/0.6 by default) combines results.
- **RRF fusion** (`src/retrieval/fusion.py`): k=60 (Cormack 2009), used by MQE to fuse 5 parallel query variants.
- **Graceful degradation**: Neo4j unavailable → semantic memory auto-disabled, Q&A and notes continue working. `QAEngine._neo4j_available` flag propagates everywhere.
- **Multi-user isolation**: Every data path carries `user_id` — ChromaDB `where` filter, SQLite column, Neo4j node property. User identity persisted via `st.query_params["user_id"]` across page switches.
- **Feedback loop**: 👍/👎 stored in SQLite `feedback` table; the monitoring page (`pages/5_📊_监控.py`) aggregates by retrieval method showing satisfaction rate and latency.

### Data Flow (Question → Answer)

1. `app.py` calls `engine.retrieve(question, method)` → fetches top-k `SourceChunk`s (optionally re-ranked via BGE Reranker v2-m3)
2. `engine.generate_stream(question, sources)` yields tokens → `st.write_stream()` renders them
3. `engine.format_answer(raw_answer, sources)` remaps `[N]` citation markers
4. `engine.record_interaction(question, answer, sources)` writes to all 3 memory systems
5. Feedback button calls `engine.record_feedback()` → SQLite

### Three-Memory System

| Memory | Storage | Purpose |
|--------|---------|---------|
| Working | `st.session_state` | Last N Q&A rounds, FIFO eviction |
| Episodic | ChromaDB `episodic_memory` collection | Persistent Q&A + notes, semantic search |
| Semantic | Neo4j `Concept` nodes + `RELATES_TO` edges | Knowledge graph, concept extraction via LLM |

After each Q&A: `MemoryManager.record_interaction()` → LLM extracts 3-8 concepts → episodic record → Neo4j MERGE nodes → weak pairwise relations.

### Ingestion Pipeline (5 steps)

`IngestPipeline.ingest(source)`:
1. `MultiFormatLoader.load()` → list of documents (dispatches to format-specific parser)
2. `TextChunker.split()` → chunks (per-format adaptive sizes, optional semantic chunking via SemanticSplitterNodeParser)
3. `Embedder.embed()` → vectors
4. `VectorStore.add_chunks()` → ChromaDB `document_chunks` collection
5. `MetadataStore.add_document()` → SQLite `documents` table

Re-ingesting the same source is idempotent — old chunks are deleted first by `doc_id`.

### Configuration System

`Settings` is a dataclass aggregating sub-configs: `LLMConfig`, `EmbeddingConfig`, `ChromaConfig`, `Neo4jConfig`, `SQLiteConfig`, `RetrievalConfig`, `ChunkConfig`. Each sub-config is a flat dataclass with defaults. Environment variables have one-to-one mapping with config fields (e.g., `LLM_MODEL` → `LLMConfig.model`). The settings UI page allows hot-swapping LLM/embedding/retrieval parameters at runtime.

### Chunk Configuration

`ChunkConfig` supports per-format adaptive presets via `chunk_presets` dict (`pdf: (1024, 256)`, `csv/xlsx: (384, 64)`, etc.). Falls back to global `chunk_size`/`chunk_overlap` for unknown formats. Optional semantic chunking (`use_semantic_chunking`) uses LlamaIndex SemanticSplitterNodeParser with configurable buffer size, breakpoint percentile, and max chunk multiplier.

### Important Implementation Details

- ChromaDB uses `PersistentClient` (local disk at `CHROMA_PERSIST_PATH`), not client-server mode. Data lives in `./data/chroma/`.
- ChromaDB returns L2 distance; `VectorStore._parse_results()` converts to similarity score via `1 / (1 + distance)`.
- SQLite runs in WAL mode with thread-local connections for concurrent access.
- LLM client (`src/core/llm_client.py`) uses the OpenAI Python SDK pointed at any compatible endpoint (DashScope, DeepSeek, Ollama, etc.).
- Streamlit `@st.cache_resource` on `get_engine()` ensures single QAEngine per process — all pages share the same instance.
- Re-ranker (`src/retrieval/reranker.py`) is lazily loaded; when enabled, coarse retrieval fetches more candidates (`reranker_top_k` or 2× top_k), then Cross-Encoder re-scores and truncates.
- E2E tests in `tests/e2e/` use Selenium to test the Streamlit app; they are excluded from CI (`--ignore=tests/e2e`).
