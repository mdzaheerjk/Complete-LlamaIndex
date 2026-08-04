# LlamaIndex — The Complete Engineering Notes
### From Zero to Production: RAG & Agentic AI with LlamaIndex

> A comprehensive, production-oriented reference for Generative AI engineers, Agentic AI engineers, and RAG engineers. Covers core concepts, ingestion, indexing, retrieval, query engines, agents, workflows, evaluation, observability, and deployment.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation & Environment Setup](#2-installation--environment-setup)
3. [Core Concepts](#3-core-concepts)
4. [Data Ingestion (Loaders / Readers)](#4-data-ingestion-loaders--readers)
5. [Document → Node Parsing & Chunking](#5-document--node-parsing--chunking)
6. [Embeddings](#6-embeddings)
7. [LLM Integrations](#7-llm-integrations)
8. [Indexes (Deep Dive)](#8-indexes-deep-dive)
9. [Storage: Vector Stores, Docstores, Index Stores](#9-storage-vector-stores-docstores-index-stores)
10. [Retrievers](#10-retrievers)
11. [Node Postprocessors & Rerankers](#11-node-postprocessors--rerankers)
12. [Response Synthesis](#12-response-synthesis)
13. [Query Engines](#13-query-engines)
14. [Chat Engines & Memory](#14-chat-engines--memory)
15. [Prompt Engineering in LlamaIndex](#15-prompt-engineering-in-llamaindex)
16. [Advanced RAG Patterns](#16-advanced-rag-patterns)
17. [Agents](#17-agents)
18. [Tools & Function Calling](#18-tools--function-calling)
19. [Workflows (Event-Driven Orchestration)](#19-workflows-event-driven-orchestration)
20. [Multi-Agent Systems](#20-multi-agent-systems)
21. [Structured Outputs & Pydantic Programs](#21-structured-outputs--pydantic-programs)
22. [Multi-Modal RAG](#22-multi-modal-rag)
23. [LlamaParse & LlamaCloud](#23-llamaparse--llamacloud)
24. [Evaluation](#24-evaluation)
25. [Observability & Tracing](#25-observability--tracing)
26. [Caching & Cost Optimization](#26-caching--cost-optimization)
27. [Fine-Tuning with LlamaIndex](#27-fine-tuning-with-llamaindex)
28. [Production Deployment](#28-production-deployment)
29. [Security & Guardrails](#29-security--guardrails)
30. [Testing Strategy for RAG/Agentic Systems](#30-testing-strategy-for-ragagentic-systems)
31. [LlamaIndex vs LangChain vs Haystack](#31-llamaindex-vs-langchain-vs-haystack)
32. [Common Pitfalls & Debugging Checklist](#32-common-pitfalls--debugging-checklist)
33. [Reference Cheat Sheet](#33-reference-cheat-sheet)
34. [Further Resources](#34-further-resources)

---

## 1. Introduction

### 1.1 What is LlamaIndex?

LlamaIndex is a **data framework** for building LLM applications, especially:

- **RAG (Retrieval-Augmented Generation)** pipelines — connect LLMs to your private/enterprise data.
- **Agentic applications** — LLMs that reason, plan, and call tools/functions to accomplish tasks.
- **Data agents & workflows** — event-driven orchestration of multi-step, multi-agent LLM systems.

It sits between your **raw data** (PDFs, SQL databases, APIs, Slack, Notion, S3, websites, etc.) and your **LLM**, handling:

```
Raw Data → Ingestion → Parsing/Chunking → Embedding → Indexing → Storage
                                                              ↓
User Query → Retrieval → Postprocessing → Synthesis → LLM Response
```

### 1.2 Why LlamaIndex (vs raw prompting)

| Problem | LlamaIndex Solution |
|---|---|
| LLM context window is limited | Chunking + retrieval brings only relevant context |
| LLM has no access to private/live data | Connectors (readers) ingest 160+ data sources |
| Need structured, repeatable pipelines | Indexes, query engines, workflows |
| Need multi-step reasoning + tool use | Agents, Workflows, ReAct |
| Need to evaluate/debug RAG quality | Built-in evaluation & observability integrations |

### 1.3 Two Core Use-Case Families

1. **RAG Engineering** — build search/QA over documents. Focus: ingestion quality, chunking strategy, retrieval precision/recall, reranking, synthesis.
2. **Agentic AI Engineering** — build systems that *act*: call APIs, execute code, use multiple tools, coordinate multi-agent workflows, maintain state across steps.

LlamaIndex is designed so these two worlds compose — an Agent can *use* a RAG query engine as one of its Tools.

### 1.4 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Application Layer                      │
│         (Agents, Workflows, Chat Engines, Query Engines)      │
├─────────────────────────────────────────────────────────────┤
│   Retrieval Layer   │  Synthesis Layer  │   Tool/Agent Layer  │
│  (Retrievers,        │  (Response        │  (FunctionTool,     │
│   Postprocessors)     │   Synthesizers)   │   ReAct, Workflows) │
├─────────────────────────────────────────────────────────────┤
│                        Index Layer                            │
│     (VectorStoreIndex, SummaryIndex, KnowledgeGraphIndex...)  │
├─────────────────────────────────────────────────────────────┤
│                       Storage Layer                           │
│   (VectorStore, DocStore, IndexStore, GraphStore, Cache)      │
├─────────────────────────────────────────────────────────────┤
│                      Ingestion Layer                          │
│         (Readers/Loaders, Node Parsers, Transformations)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Installation & Environment Setup

### 2.1 Package Structure

Since v0.10+, LlamaIndex is **modular** — a thin core package plus many small integration packages. This keeps installs light and avoids dependency bloat.

```bash
# Core framework (includes OpenAI by default as a "starter" integration)
pip install llama-index

# Or install just the core abstractions (no default integrations)
pip install llama-index-core

# Common integrations installed à la carte:
pip install llama-index-llms-openai
pip install llama-index-llms-anthropic
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
pip install llama-index-embeddings-openai
pip install llama-index-vector-stores-chroma
pip install llama-index-vector-stores-pinecone
pip install llama-index-vector-stores-qdrant
pip install llama-index-vector-stores-postgres   # pgvector
pip install llama-index-readers-file
pip install llama-index-readers-web
pip install llama-index-readers-database
pip install llama-index-agent-openai
pip install llama-index-postprocessor-cohere-rerank
```

> **Naming convention:** `llama-index-<category>-<provider>`
> Categories: `llms`, `embeddings`, `vector-stores`, `readers`, `postprocessor`, `agent`, `graph-stores`, `storage`, `tools`, `packs`, `output-parsers`.

### 2.2 Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."
export LLAMA_CLOUD_API_KEY="llx-..."   # for LlamaParse / LlamaCloud
```

Or with `.env` + `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 2.3 Minimal "Hello World" RAG

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. Load documents
documents = SimpleDirectoryReader("./data").load_data()

# 2. Build index (chunk → embed → store, all default settings)
index = VectorStoreIndex.from_documents(documents)

# 3. Query
query_engine = index.as_query_engine()
response = query_engine.query("What are the key findings in these documents?")
print(response)
```

That's the entire pipeline in 3 lines — LlamaIndex fills in defaults (chunk size 1024, OpenAI `text-embedding-ada-002`/`text-embedding-3-small`, in-memory vector store, `gpt-3.5/4` synthesis). **Production systems override every one of these defaults** — the rest of this document shows how and why.

### 2.4 Global Settings (`Settings` object)

Replaces the older, deprecated `ServiceContext`.

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50
Settings.num_output = 512
Settings.context_window = 128000
```

`Settings` is a **global singleton** used as the default whenever a component isn't explicitly passed one. You can still override locally per-index/per-query-engine — local overrides always win.

> ⚠️ **Production tip:** Avoid relying on global `Settings` in multi-tenant or multi-model services (e.g., a FastAPI app serving different models per request). Prefer explicit, per-request construction of `LLM`/`embed_model` objects instead of mutating the global singleton, which is not thread-safe/request-safe.


---

## 3. Core Concepts

### 3.1 Document

A `Document` is the raw unit of ingested content plus metadata.

```python
from llama_index.core import Document

doc = Document(
    text="LlamaIndex is a data framework for LLM applications.",
    metadata={"source": "intro.md", "author": "team-docs", "category": "overview"},
    id_="doc-001",
)
```

Key fields:
- `text`: raw content
- `metadata`: dict, attached to every derived Node (used for filtering & citation)
- `excluded_llm_metadata_keys` / `excluded_embed_metadata_keys`: hide certain metadata from the LLM or embedding model (e.g., internal IDs) while still keeping it for filtering
- `id_`: stable identifier — critical for incremental ingestion/updates

### 3.2 Node

A `Node` is a **chunk** of a Document — the atomic unit that gets embedded, stored, and retrieved.

```python
from llama_index.core.schema import TextNode

node = TextNode(
    text="chunked text content...",
    metadata={"source": "intro.md"},
    relationships={},  # NodeRelationship.SOURCE, PREVIOUS, NEXT, PARENT, CHILD
)
```

Nodes carry **relationships** to other nodes (previous/next chunk, parent document, child nodes) — this is what enables advanced retrieval strategies like sentence-window and auto-merging retrieval (see §16).

### 3.3 Index

An **Index** is a data structure built over Nodes that enables efficient retrieval. LlamaIndex supports several index types (deep dive in §8):

| Index | Structure | Best For |
|---|---|---|
| `VectorStoreIndex` | Embedding vectors + ANN search | Semantic similarity search (default choice, ~90% of use cases) |
| `SummaryIndex` (ListIndex) | Sequential list, no embeddings | Small corpora, "read everything" summarization |
| `TreeIndex` | Hierarchical summary tree | Hierarchical summarization, long documents |
| `KeywordTableIndex` | Keyword → node map | Exact keyword lookup |
| `KnowledgeGraphIndex` / `PropertyGraphIndex` | Entities + relations (triplets) | Relationship-heavy, multi-hop reasoning |
| `DocumentSummaryIndex` | Per-doc summary + full doc | Doc-level retrieval before chunk-level |

### 3.4 The Ingestion Pipeline (mental model)

```
Documents ──[NodeParser/TextSplitter]──▶ Nodes ──[Embedding Model]──▶ Vectors
                                                                          │
                                                                          ▼
                                                                   VectorStore
```

### 3.5 The Query Pipeline (mental model)

```
User Query ─▶ Retriever ─▶ [Node, Node, ...] ─▶ Postprocessors ─▶ Response Synthesizer ─▶ LLM ─▶ Response
                 │                                    │
          (vector search,               (rerank, filter, dedupe,
           hybrid, BM25, etc.)             compress, MMR)
```

A **QueryEngine** = Retriever + Postprocessor(s) + ResponseSynthesizer, wired together.

### 3.6 Object Model Summary

```
Document
   │  (parsed by NodeParser)
   ▼
Node(s) ──has──▶ Embedding, Metadata, Relationships
   │  (added to)
   ▼
Index ──backed by──▶ VectorStore + DocStore + IndexStore
   │  (exposes)
   ▼
Retriever ──feeds──▶ QueryEngine / ChatEngine / Agent Tool
```

---

## 4. Data Ingestion (Loaders / Readers)

### 4.1 SimpleDirectoryReader — the workhorse

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
    required_exts=[".pdf", ".docx", ".txt", ".md"],
    exclude=["**/README.md"],
    filename_as_id=True,
    file_metadata=lambda file_path: {"file_path": file_path, "source": "local_fs"},
)
documents = reader.load_data(num_workers=4)  # parallel loading
```

Supports out-of-the-box: `.pdf, .docx, .pptx, .csv, .epub, .md, .html, .json, images (via CLIP/multi-modal), audio (via whisper)`, etc.

### 4.2 LlamaHub — 300+ Data Connectors

Install any connector à la carte from LlamaHub (llamahub.ai):

```bash
pip install llama-index-readers-web
pip install llama-index-readers-database
pip install llama-index-readers-slack
pip install llama-index-readers-notion
pip install llama-index-readers-google
```

```python
# Website
from llama_index.readers.web import SimpleWebPageReader
docs = SimpleWebPageReader(html_to_text=True).load_data(["https://example.com"])

# SQL Database
from llama_index.readers.database import DatabaseReader
db_reader = DatabaseReader(uri="postgresql://user:pass@host:5432/db")
docs = db_reader.load_data(query="SELECT id, title, body FROM articles")

# Notion
from llama_index.readers.notion import NotionPageReader
docs = NotionPageReader(integration_token="secret_...").load_data(page_ids=["..."])

# Wikipedia
from llama_index.readers.wikipedia import WikipediaReader
docs = WikipediaReader().load_data(pages=["Retrieval-augmented_generation"])
```

### 4.3 Custom Readers

```python
from llama_index.core.readers.base import BaseReader
from llama_index.core import Document

class MyAPIReader(BaseReader):
    def __init__(self, api_client):
        self.api_client = api_client

    def load_data(self, **kwargs):
        records = self.api_client.fetch_all()
        return [
            Document(text=r["body"], metadata={"id": r["id"], "updated_at": r["updated_at"]})
            for r in records
        ]
```

### 4.4 The Ingestion Pipeline API (production pattern)

For production, don't call loaders + parsers manually — use `IngestionPipeline`, which supports **caching, deduplication, and incremental upserts** via a docstore.

```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

client = qdrant_client.QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(client=client, collection_name="docs")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        OpenAIEmbedding(model="text-embedding-3-small"),
    ],
    docstore=SimpleDocumentStore(),          # enables dedup via doc_id + content hash
    vector_store=vector_store,
    cache=IngestionCache(),                   # caches transformation outputs
    docstore_strategy="upserts",              # UPSERTS / DUPLICATES_ONLY / UPSERTS_AND_DELETE
)

nodes = pipeline.run(documents=documents, show_progress=True)
pipeline.persist("./pipeline_storage")  # persist cache+docstore for next run
```

**Why this matters in production:** re-running ingestion on a data source that changed only 5% of documents should *not* re-embed 100% of documents. `docstore_strategy="upserts"` hashes each doc's content; unchanged docs are skipped, changed docs are re-processed and old nodes removed, new docs are added. This alone can cut embedding API costs by 10-100x on recurring ingestion jobs.

### 4.5 Metadata Extraction

Enrich nodes with LLM-generated metadata to improve retrieval quality:

```python
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512),
        TitleExtractor(nodes=5),                       # infer doc title from first 5 nodes
        QuestionsAnsweredExtractor(questions=3),        # "this chunk can answer: ..."
        SummaryExtractor(summaries=["prev", "self", "next"]),
        KeywordExtractor(keywords=10),
    ]
)
nodes = pipeline.run(documents=documents)
```

`QuestionsAnsweredExtractor` is particularly powerful: it embeds hypothetical questions the chunk answers, which often matches user query phrasing better than the raw chunk text (a lightweight form of HyDE, see §16).


---

## 5. Document → Node Parsing & Chunking

Chunking strategy is arguably **the single highest-leverage decision** in a RAG pipeline. Bad chunking → irrelevant retrieval → bad answers, no matter how good the LLM is.

### 5.1 SentenceSplitter (default, most common)

Splits on sentence boundaries while respecting a target chunk size (in tokens), with overlap.

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
```

- **chunk_size**: 256–512 for precise Q&A; 1000–1500 for broader context/summarization.
- **chunk_overlap**: typically 10-20% of chunk_size — preserves context across chunk boundaries.

### 5.2 TokenTextSplitter

Pure token-count-based split, no sentence-boundary awareness — faster, less semantically clean.

### 5.3 SentenceWindowNodeParser (for Sentence-Window Retrieval)

Splits into **single sentences** but stores surrounding sentences in metadata — retrieve small (precise embeddings) but synthesize with a wider window (better context). See §16.2.

```python
from llama_index.core.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,                       # 3 sentences before/after
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
nodes = node_parser.get_nodes_from_documents(documents)
```

### 5.4 HierarchicalNodeParser (for Auto-Merging Retrieval)

Produces multiple granularities of chunks (e.g., 2048 → 512 → 128 tokens) linked parent→child. Enables **auto-merging retrieval**: if enough child chunks under one parent are retrieved, return the parent instead. See §16.3.

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)
```

### 5.5 SemanticSplitterNodeParser

Splits based on **embedding similarity** between sentences rather than a fixed token count — chunk boundaries fall where topic/semantics shift.

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=OpenAIEmbedding(),
)
nodes = splitter.get_nodes_from_documents(documents)
```

Slower and costs embedding calls at ingestion time, but often yields more coherent chunks for narrative or technical text.

### 5.6 Structure-Aware Splitters

```python
from llama_index.core.node_parser import (
    MarkdownNodeParser,     # splits on markdown headers
    JSONNodeParser,         # splits on JSON structure
    CodeSplitter,           # AST-aware code chunking (via tree-sitter)
)

code_splitter = CodeSplitter(language="python", chunk_lines=40, chunk_lines_overlap=15)
```

### 5.7 Chunking Strategy Decision Table

| Data Type | Recommended Parser | Notes |
|---|---|---|
| General prose / articles | `SentenceSplitter` (512, overlap 50) | Safe default |
| Long technical docs, need precise + wide context | `SentenceWindowNodeParser` | Small embed, big synth window |
| Very long docs, need parent context fallback | `HierarchicalNodeParser` + auto-merging retriever | Balances precision & recall |
| Narrative/topic-shifting text | `SemanticSplitterNodeParser` | Higher cost, better coherence |
| Markdown/docs sites | `MarkdownNodeParser` | Preserves header hierarchy as metadata |
| Source code | `CodeSplitter` | AST-aware, avoids splitting mid-function |
| Tables/structured data | Keep as-is + `PandasQueryEngine` / SQL, don't chunk naively | Chunking tables destroys row context |

### 5.8 Chunk Size Tuning — Practical Guidance

- **Too small** (e.g., 128 tokens): high precision, but loses context → LLM hallucinates to fill gaps; more chunks needed at retrieval time (higher `top_k` needed).
- **Too large** (e.g., 4000 tokens): dilutes embedding signal (averaging effect), retrieval precision drops, more irrelevant text sent to LLM (cost + "lost in the middle" effect).
- **Rule of thumb starting point:** 512 tokens, 10% overlap, then tune based on evaluation metrics (§24), not intuition.
- Always **evaluate empirically** — chunk size is a hyperparameter, not a fixed constant. Run a grid ({256, 512, 1024} × overlap {0, 50, 100}) against a golden eval set.


---

## 6. Embeddings

### 6.1 Setting an Embedding Model

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=512)
```

### 6.2 Common Embedding Providers

```python
# OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# HuggingFace (local, free, no API cost — great for on-prem/production cost control)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Cohere
from llama_index.embeddings.cohere import CohereEmbedding
embed_model = CohereEmbedding(model_name="embed-english-v3.0", input_type="search_document")

# Ollama (local)
from llama_index.embeddings.ollama import OllamaEmbedding
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Azure OpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
```

### 6.3 Batch Embedding & Direct Usage

```python
embeddings = embed_model.get_text_embedding_batch(
    ["text one", "text two", "text three"], show_progress=True
)
query_embedding = embed_model.get_query_embedding("What is RAG?")
```

> Some models (BGE, E5, Cohere v3) use **asymmetric embeddings** — different encoding for queries vs. documents (`input_type="search_query"` vs `"search_document"`). Using the wrong mode silently degrades retrieval quality — always check the model card.

### 6.4 Choosing an Embedding Model

| Criteria | Recommendation |
|---|---|
| Best general quality, simplicity | `text-embedding-3-large` (OpenAI) |
| Cost-sensitive / on-prem / data residency | `bge-large-en-v1.5`, `e5-large-v2` (self-hosted) |
| Multilingual | `bge-m3`, `text-embedding-3-large`, `Cohere multilingual-v3` |
| Domain-specific (legal, medical, code) | Fine-tuned embedding model (§27) or specialized model (`voyage-code-2` for code) |
| Long-context chunks | Check model max seq length — many embedding models cap at 512 tokens; check before setting chunk_size > that |

Always check the **[MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)** for current benchmarks — this landscape moves fast; don't assume training-data-era rankings are still accurate.

### 6.5 Embedding Dimensionality & Storage Cost

Vector DB storage/compute cost scales with dimensionality × row count. OpenAI's `text-embedding-3-*` models support **Matryoshka truncation** via the `dimensions` parameter — you can shrink from 3072 → 512 dims with modest quality loss and large storage/latency savings. Benchmark the tradeoff for your corpus rather than assuming full dimensionality is required.

---

## 7. LLM Integrations

### 7.1 Setting an LLM

```python
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

Settings.llm = OpenAI(model="gpt-4o", temperature=0.1, max_tokens=1024)
```

### 7.2 Common Providers

```python
from llama_index.llms.anthropic import Anthropic
llm = Anthropic(model="claude-sonnet-4-5", max_tokens=1024)

from llama_index.llms.ollama import Ollama
llm = Ollama(model="llama3.1", request_timeout=120.0)

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.vertex import Vertex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.groq import Groq   # fast inference
```

### 7.3 LLM Call Patterns

```python
# Simple completion
resp = llm.complete("What is retrieval-augmented generation?")

# Chat
from llama_index.core.llms import ChatMessage
messages = [
    ChatMessage(role="system", content="You are a helpful RAG assistant."),
    ChatMessage(role="user", content="Explain vector search."),
]
resp = llm.chat(messages)

# Streaming
for chunk in llm.stream_complete("Explain agentic AI."):
    print(chunk.delta, end="")

# Async
resp = await llm.acomplete("...")
```

### 7.4 Structured / Function-Calling LLMs

Most production agentic pipelines require an LLM that supports native function/tool calling (OpenAI, Anthropic, Gemini, Mistral-large, etc.) — this determines which Agent classes are available (§17).

```python
llm.metadata.is_function_calling_model  # True/False, check before choosing agent type
```

### 7.5 Local / Self-Hosted LLMs

```python
from llama_index.llms.ollama import Ollama
Settings.llm = Ollama(model="mistral", request_timeout=300)

# or via vLLM / TGI OpenAI-compatible endpoint
from llama_index.llms.openai_like import OpenAILike
Settings.llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key="fake",
    is_chat_model=True,
)
```

### 7.6 LLM Selection Guidance (Production)

| Need | Guidance |
|---|---|
| Best reasoning / complex agents | Frontier models (GPT-4-class, Claude-class) |
| Low latency, high throughput, simple tasks | Smaller/faster models (mini/haiku-class), or Groq-hosted open models |
| Data residency / air-gapped | Self-hosted via Ollama/vLLM + open-weight model |
| Cost at scale | Route by task complexity — use a cheap model for routing/classification, escalate to a strong model only for final synthesis (see RouterQueryEngine, §13.3) |
| Deterministic structured extraction | Lower temperature (0–0.2) + structured output mode (§21) |


---

## 8. Indexes (Deep Dive)

### 8.1 VectorStoreIndex

The default, most-used index — embeds each node, stores in a vector store, retrieves via ANN (approximate nearest neighbor) similarity search.

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
# or build incrementally
index = VectorStoreIndex([])
index.insert_nodes(nodes)
index.insert(document)
index.delete_ref_doc("doc-001", delete_from_docstore=True)
```

### 8.2 SummaryIndex (formerly ListIndex)

Stores nodes in a simple sequential list — no embedding-based retrieval; at query time, by default it **sends every node to the LLM** (or uses an LLM-based node selector). Best for small corpora or a single-document summarization use case; expensive/impractical at scale.

```python
from llama_index.core import SummaryIndex
index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine(response_mode="tree_summarize")
```

### 8.3 TreeIndex

Builds a hierarchical tree of summaries (leaf nodes → parent summary nodes → root). Query traverses top-down. Useful for long single documents needing multi-level summarization; less common in modern stacks (largely superseded by hierarchical retrieval + `tree_summarize` response mode).

### 8.4 KeywordTableIndex

Extracts keywords per node (via LLM or simple regex `SimpleKeywordTableIndex`), builds keyword → node-id map. Good for exact-term lookup, poor for semantic queries. Rarely used alone in production — usually combined with vector search in a hybrid retriever.

### 8.5 DocumentSummaryIndex

Stores a **summary per document** plus the full document; retrieval can first match at the document-summary level, then drill into the matched document's nodes. Useful when you have many documents and want doc-level relevance ranking before chunk-level retrieval.

```python
from llama_index.core import DocumentSummaryIndex, get_response_synthesizer

response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
doc_summary_index = DocumentSummaryIndex.from_documents(
    documents, response_synthesizer=response_synthesizer,
)
```

### 8.6 KnowledgeGraphIndex & PropertyGraphIndex

Extracts `(subject, relation, object)` triplets from text (via LLM), builds a graph. `PropertyGraphIndex` (newer, more flexible) supports typed nodes/edges, custom extractors, and hybrid vector+graph retrieval.

```python
from llama_index.core import PropertyGraphIndex
from llama_index.llms.openai import OpenAI

index = PropertyGraphIndex.from_documents(
    documents,
    llm=OpenAI(model="gpt-4o-mini"),
    embed_kg_nodes=True,   # also embed entities for hybrid retrieval
)
query_engine = index.as_query_engine(include_text=True)
```

Best for: multi-hop reasoning ("who reports to the person who approved project X?"), relationship-dense domains (org charts, supply chains, biomedical literature, legal case law).

### 8.7 Composability — ComposableGraph / Multi-Index

You can combine multiple indexes (e.g., one VectorStoreIndex per data source) under a single top-level index or router, so queries are routed to the right sub-index. See RouterQueryEngine (§13.3) — this is the modern, preferred pattern over the older `ComposableGraph`.

### 8.8 Choosing an Index — Decision Table

| Scenario | Index |
|---|---|
| General semantic search over docs (default) | `VectorStoreIndex` |
| Small corpus, need full summarization | `SummaryIndex` |
| Long single document, hierarchical summary | `TreeIndex` or hierarchical chunking + `tree_summarize` |
| Exact keyword match required | `KeywordTableIndex` (usually hybridized) |
| Many documents, need doc-level relevance first | `DocumentSummaryIndex` |
| Relationship/multi-hop reasoning | `PropertyGraphIndex` |
| Multiple heterogeneous data sources | Multiple indexes + `RouterQueryEngine` |

---

## 9. Storage: Vector Stores, Docstores, Index Stores

### 9.1 The StorageContext

```python
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=SimpleDocumentStore(),
    index_store=SimpleIndexStore(),
)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
storage_context.persist(persist_dir="./storage")
```

`StorageContext` bundles three separate stores:
- **VectorStore** — embeddings + similarity search
- **DocStore** — raw node/document content + metadata (needed to reconstruct text, dedupe, etc.)
- **IndexStore** — index metadata (which nodes belong to which index, index config)

### 9.2 Reloading a Persisted Index

```python
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

For external vector DBs (Pinecone, Qdrant, etc.) that are already persistent, you don't "load" — you reconnect:

```python
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
```

### 9.3 Vector Store Integrations (production options)

| Vector Store | Notes |
|---|---|
| `Chroma` | Easiest local/dev, embeddable, good for prototyping |
| `Qdrant` | Production-grade, filtering, hybrid search, on-prem or cloud |
| `Pinecone` | Fully managed, scales well, popular in production |
| `Weaviate` | Managed or self-hosted, built-in hybrid search + modules |
| `pgvector` (Postgres) | Great if you already run Postgres — single system to operate |
| `Milvus` / `Zilliz` | High-scale, large enterprise deployments |
| `Elasticsearch` / `OpenSearch` | Best when you need existing full-text search infra + vectors combined |
| `Redis` | Low-latency, good when Redis is already in your stack |
| `LanceDB` | Embedded, serverless-friendly, good for edge/local-first apps |

```python
# Example: pgvector
from llama_index.vector_stores.postgres import PGVectorStore

vector_store = PGVectorStore.from_params(
    database="ragdb", host="localhost", password="pw", port=5432, user="pguser",
    table_name="doc_embeddings", embed_dim=1536,
    hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
)
```

### 9.4 Metadata Filtering at the Vector Store Level

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="finance", operator=FilterOperator.EQ),
        MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE),
    ]
)
retriever = index.as_retriever(filters=filters, similarity_top_k=5)
```

This pushes filtering **down to the vector store** (pre-filter), which is far more efficient than retrieving broadly and filtering in application code — essential for multi-tenant RAG (filter by `tenant_id`/`user_id` at the DB level, never trust app-layer filtering alone for security-sensitive isolation).


---

## 10. Retrievers

### 10.1 Basic Vector Retriever

```python
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("What is hybrid search?")
for n in nodes:
    print(n.score, n.node.get_content()[:100])
```

### 10.2 Retrieval Modes per Index Type

```python
# SummaryIndex retrieval modes
retriever = summary_index.as_retriever(retriever_mode="embedding")  # or "default" (all nodes), "llm"

# KeywordTableIndex
retriever = keyword_index.as_retriever(retriever_mode="simple")  # or "rake", "llm"
```

### 10.3 Hybrid Search (Dense + Sparse/BM25)

Combines semantic (vector) search with lexical (keyword/BM25) search — catches both "meaning matches" and "exact term matches" (critical for acronyms, product codes, proper nouns that embeddings often under-weight).

```python
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

vector_retriever = index.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)

hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=4,              # generates query variations for better recall
    mode="reciprocal_rerank",   # fuse rankings via Reciprocal Rank Fusion
    use_async=True,
)
nodes = hybrid_retriever.retrieve("error code E4021 troubleshooting")
```

Many vector DBs (Qdrant, Weaviate, Elasticsearch, Pinecone) also support **native hybrid search** server-side — prefer that over app-level fusion when available, for lower latency.

### 10.4 Auto-Retrieval (LLM infers metadata filters from natural language)

The LLM parses the user's query into a structured filter + semantic query automatically.

```python
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

vector_store_info = VectorStoreInfo(
    content_info="Product reviews",
    metadata_info=[
        MetadataInfo(name="rating", type="int", description="Rating 1-5"),
        MetadataInfo(name="category", type="str", description="Product category"),
    ],
)
retriever = VectorIndexAutoRetriever(index, vector_store_info=vector_store_info)
# Query: "Show me 5-star reviews for electronics"
# → auto-generates: filter(category=="electronics", rating==5) + semantic query
```

### 10.5 Recursive Retriever

Retrieves small "reference" chunks that point to a larger object (a table, a sub-index, a full document) and recursively fetches the referenced content — key mechanism for auto-merging retrieval and small-to-big retrieval over tables/nested docs.

```python
from llama_index.core.retrievers import RecursiveRetriever

retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    node_dict=node_mapping,  # maps IndexNode.index_id -> referenced object
)
```

### 10.6 Custom Retriever

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

class HybridCustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever, keyword_retriever):
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        combined = {n.node.node_id: n for n in vector_nodes}
        for n in keyword_nodes:
            combined.setdefault(n.node.node_id, n)
        return list(combined.values())
```

### 10.7 Retrieval Tuning Parameters

| Parameter | Effect | Guidance |
|---|---|---|
| `similarity_top_k` | Number of nodes retrieved | Start at 5-10; too high dilutes context / raises cost |
| `similarity_cutoff` (postprocessor) | Score threshold | Filters weak matches; tune against eval set, don't guess |
| `vector_store_query_mode` | `default`, `sparse`, `hybrid` | Use hybrid when exact-term queries matter |
| `alpha` (hybrid weighting) | Balance dense vs sparse | 0.5 typical start; tune per corpus |

---

## 11. Node Postprocessors & Rerankers

Postprocessors run **after retrieval, before synthesis** — filter, reorder, compress, or transform retrieved nodes.

### 11.1 Similarity/Keyword Filtering

```python
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor

postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.75),
    KeywordNodePostprocessor(required_keywords=["contract"], exclude_keywords=["draft"]),
]
```

### 11.2 Reranking (huge quality lever)

Initial vector retrieval (bi-encoder) is fast but approximate. A **cross-encoder reranker** re-scores the top-k candidates with much higher precision — retrieve broad (top_k=25-50), rerank down to top_n=5.

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank
reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)

# or local cross-encoder (no API cost, good for on-prem)
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)

query_engine = index.as_query_engine(
    similarity_top_k=25,
    node_postprocessors=[reranker],
)
```

> **This is one of the highest ROI additions to any RAG pipeline.** Retrieve wide (recall), rerank narrow (precision). Almost every serious production RAG system uses a reranking stage.

### 11.3 LLM-Based Reranking / Filtering

```python
from llama_index.core.postprocessor import LLMRerank
reranker = LLMRerank(top_n=5, choice_batch_size=5)  # uses LLM to judge relevance, more expensive
```

### 11.4 Long Context Reorder ("Lost in the Middle" mitigation)

LLMs attend more strongly to content at the start/end of context than the middle. This postprocessor reorders retrieved nodes so the most relevant ones sit at the edges.

```python
from llama_index.core.postprocessor import LongContextReorder
node_postprocessors = [LongContextReorder()]
```

### 11.5 Prompt Compression

```python
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor
compressor = LongLLMLinguaPostprocessor(target_token=300)  # compress retrieved text before sending to LLM
```

Useful when retrieved context is large and you want to cut token cost/latency without losing key information — trades a small extra LLM/compression-model call for large main-LLM token savings.

### 11.6 Full Postprocessor Chain (typical production order)

```python
query_engine = index.as_query_engine(
    similarity_top_k=25,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.6),
        CohereRerank(top_n=6),
        LongContextReorder(),
    ],
)
```


---

## 12. Response Synthesis

The **ResponseSynthesizer** takes retrieved nodes + the query and generates the final answer — handling the case where retrieved content exceeds the LLM's context window.

### 12.1 Response Modes

```python
from llama_index.core import get_response_synthesizer

synthesizer = get_response_synthesizer(response_mode="compact")
```

| Mode | Behavior | Use When |
|---|---|---|
| `refine` | Sequentially processes each node, refining the answer one chunk at a time | Need highest accuracy on long context, tolerant of latency; original default |
| `compact` (default) | Like refine but packs as many chunks as fit per LLM call first — fewer calls | Best balance of speed/cost/quality — most common choice |
| `tree_summarize` | Recursively summarizes nodes in a tree (bottom-up) until one answer remains | Summarization over many/large chunks, better than refine for holistic summaries |
| `simple_summarize` | Truncates/concats all nodes into one call, no chunk-by-chunk refinement | Small number of short nodes only |
| `no_text` | Skips LLM call, returns retrieved nodes only | When you just want retrieval, not generation (e.g., building your own downstream logic) |
| `accumulate` | Runs LLM separately per node, concatenates all outputs | Need per-chunk independent answers (e.g., "list all mentions of X") |
| `compact_accumulate` | Accumulate but batched like compact | Faster accumulate |

### 12.2 Direct Usage

```python
from llama_index.core import get_response_synthesizer
from llama_index.core.schema import NodeWithScore, TextNode

synthesizer = get_response_synthesizer(response_mode="tree_summarize")
response = synthesizer.synthesize(
    query="Summarize the key risks mentioned across these reports.",
    nodes=[NodeWithScore(node=n, score=1.0) for n in retrieved_nodes],
)
```

### 12.3 Streaming Responses

```python
query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Explain the architecture.")
streaming_response.print_response_stream()
```

### 12.4 Citations / Source Attribution

```python
response = query_engine.query("What does the contract say about termination?")
for sn in response.source_nodes:
    print(sn.node.metadata.get("file_name"), sn.score, sn.node.get_content()[:150])
```

For strict citation formatting (e.g., `[1]`, `[2]` inline), use `CitationQueryEngine`:

```python
from llama_index.core.query_engine import CitationQueryEngine

query_engine = CitationQueryEngine.from_args(
    index, similarity_top_k=5, citation_chunk_size=256,
)
response = query_engine.query("What are the payment terms?")
print(response.response)          # includes inline [1], [2] markers
print(response.source_nodes)      # matches citation numbers to source chunks
```

---

## 13. Query Engines

A **QueryEngine** wraps Retriever + Postprocessors + ResponseSynthesizer into a single callable interface — the standard "ask a question, get an answer" abstraction.

### 13.1 Basic Query Engine

```python
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
    response_mode="compact",
    streaming=False,
)
response = query_engine.query("...")
```

### 13.2 Sub-Question Query Engine (decompose complex questions)

Breaks a complex multi-part question into sub-questions, answers each against the relevant tool/index, then synthesizes a combined answer.

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(
        query_engine=sales_index.as_query_engine(),
        metadata=ToolMetadata(name="sales_data", description="Quarterly sales figures"),
    ),
    QueryEngineTool(
        query_engine=hr_index.as_query_engine(),
        metadata=ToolMetadata(name="hr_data", description="Employee headcount and policies"),
    ),
]

sub_question_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
response = sub_question_engine.query(
    "How did sales per employee change between Q1 and Q3, given headcount changes?"
)
```

### 13.3 Router Query Engine (choose the right index/tool per query)

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        QueryEngineTool(query_engine=summary_index.as_query_engine(),
                         metadata=ToolMetadata(name="summary", description="Good for summarization questions")),
        QueryEngineTool(query_engine=vector_index.as_query_engine(),
                         metadata=ToolMetadata(name="vector_search", description="Good for specific factual lookups")),
    ],
)
```

This is the standard pattern for **multi-source RAG** — e.g., route "summarize this doc" queries to a SummaryIndex and "what does clause 5.2 say" queries to a VectorStoreIndex.

### 13.4 SQL Query Engine (Text-to-SQL RAG)

```python
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase

engine = create_engine("postgresql://user:pass@host/db")
sql_database = SQLDatabase(engine, include_tables=["sales", "customers"])

query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=["sales", "customers"])
response = query_engine.query("What were total sales by region last quarter?")
```

> ⚠️ **Production caution:** text-to-SQL engines execute LLM-generated SQL against real databases. Always run against a **read-only, least-privilege DB user**, validate/sandbox generated queries, set query timeouts, and consider a human-in-the-loop confirmation step for anything beyond `SELECT`.

### 13.5 Pandas Query Engine (structured data Q&A)

```python
from llama_index.experimental.query_engine import PandasQueryEngine
import pandas as pd

df = pd.read_csv("sales.csv")
query_engine = PandasQueryEngine(df=df, verbose=True)
response = query_engine.query("What is the average order value by month?")
```

> ⚠️ This executes LLM-generated Python (`df.eval`/`exec`-like behavior) — treat as untrusted code execution; sandbox it (see §29 Security).

### 13.6 Retriever Query Engine (manual composition)

```python
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
    response_synthesizer=get_response_synthesizer(response_mode="compact"),
)
```

### 13.7 Query Engine as a Tool (bridge to Agents)

Every query engine can be wrapped as a `QueryEngineTool` and handed to an Agent — this is *the* core pattern connecting RAG engineering to Agentic AI engineering (§17-18).


---

## 14. Chat Engines & Memory

Chat engines add **conversational state** on top of a query engine — they handle multi-turn context, follow-up questions, and query condensation.

### 14.1 Chat Engine Modes

```python
chat_engine = index.as_chat_engine(chat_mode="context", similarity_top_k=5)
response = chat_engine.chat("What is covered under warranty?")
response = chat_engine.chat("What about accidental damage?")  # uses prior turn as context
```

| Mode | Behavior |
|---|---|
| `condense_question` | Rewrites follow-up into a standalone question using chat history, then does standard RAG retrieval |
| `context` | Retrieves context per turn and stuffs it + chat history into the system prompt |
| `condense_plus_context` | Combines both — condenses the question AND retrieves fresh context each turn (most robust default) |
| `react` | Wraps a ReAct agent as a chat engine (tool-using conversational agent) |
| `simple` | No retrieval, plain LLM chat |

```python
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)
```

### 14.2 Memory

```python
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", memory=memory)
```

Newer, more capable memory (v0.11+): `Memory` class with configurable **short-term (token buffer) + long-term (fact extraction / vector memory)** blocks:

```python
from llama_index.core.memory import Memory

memory = Memory.from_defaults(
    session_id="user-123",
    token_limit=40000,
    chat_history_token_ratio=0.7,   # short-term buffer proportion
)
```

### 14.3 Persisting Chat Sessions

```python
import json
chat_history = memory.get_all()
# persist chat_history to your session store (Redis/DB) keyed by session_id
# on reload:
memory = ChatMemoryBuffer.from_defaults(chat_history=restored_messages, token_limit=3000)
```

> **Production pattern:** Never keep chat memory only in-process for a multi-replica service — persist per-session history to Redis/Postgres, reload into `Memory` at the start of each request, so conversations survive across load-balanced instances/restarts.

---

## 15. Prompt Engineering in LlamaIndex

### 15.1 Inspecting Default Prompts

```python
prompts = query_engine.get_prompts()
print(prompts.keys())
print(prompts["response_synthesizer:text_qa_template"].get_template())
```

### 15.2 Custom Prompt Templates

```python
from llama_index.core import PromptTemplate

qa_prompt_tmpl = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query. If the answer isn't in the context, say you don't know.\n"
    "Query: {query_str}\n"
    "Answer: "
)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
```

### 15.3 Chat Prompt Templates

```python
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

chat_template = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content="You are a precise financial analyst assistant."),
        ChatMessage(role=MessageRole.USER, content="Context:\n{context_str}\nQuestion: {query_str}"),
    ]
)
```

### 15.4 Few-Shot Prompting

```python
from llama_index.core import PromptTemplate

few_shot_prompt = PromptTemplate(
    "Examples:\n"
    "Q: What is the refund window? A: 30 days from purchase, per Section 4.2.\n"
    "Q: Is shipping included? A: No, shipping is calculated at checkout, per Section 2.1.\n\n"
    "Context: {context_str}\n"
    "Q: {query_str}\nA:"
)
```

### 15.5 Guardrail / System Prompts for RAG

Common production system prompt pattern to reduce hallucination:

```
You are a support assistant. Answer ONLY using the provided context.
If the context does not contain the answer, respond exactly with:
"I don't have enough information to answer that."
Do not use outside knowledge. Always cite the source document name.
```

Encode this via `update_prompts` on `text_qa_template` and `refine_template` — both are used depending on response mode, so update both if changing behavior consistently across all synthesis paths.


---

## 16. Advanced RAG Patterns

### 16.1 Basic RAG Recap

```
Query → Embed → Vector Search (top_k) → Stuff into prompt → LLM → Answer
```
Simple, but suffers: imprecise chunk boundaries, no reranking, no query understanding, single-hop only.

### 16.2 Sentence-Window Retrieval

Embed **small** units (single sentences) for precise matching, but expand to a **wider window** of surrounding sentences when sending to the LLM — best of both worlds (precise retrieval, sufficient context for synthesis).

```python
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

node_parser = SentenceWindowNodeParser.from_defaults(window_size=3)
nodes = node_parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

query_engine = index.as_query_engine(
    similarity_top_k=6,
    node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],
)
```
`MetadataReplacementPostProcessor` swaps the matched single sentence back out for its stored window before synthesis.

### 16.3 Auto-Merging Retrieval

Chunk hierarchically (parent → child). Retrieve at the child (fine-grained) level; if enough children of the same parent are retrieved, **merge up** and return the parent chunk instead — gives the LLM cohesive context instead of fragmented siblings.

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext, VectorStoreIndex

node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)  # store ALL nodes (leaf + parents) for merging lookups
storage_context = StorageContext.from_defaults(docstore=docstore)

index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
base_retriever = index.as_retriever(similarity_top_k=12)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
```

### 16.4 HyDE (Hypothetical Document Embeddings)

Instead of embedding the raw user query, first ask the LLM to generate a **hypothetical answer**, embed *that*, and retrieve using it. Hypothetical answers often resemble the target documents more closely than terse user questions do.

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query("What causes catalytic converter failure?")
```

### 16.5 Query Rewriting / Multi-Query

Generate several paraphrased versions of the query to broaden recall, then fuse results (as in `QueryFusionRetriever`, §10.3).

### 16.6 Small-to-Big / Recursive Retrieval over Tables

For documents containing tables, store a **summary node** referencing the full table as a linked object; retrieve the summary, recursively fetch the full table for synthesis (via `RecursiveRetriever`, §10.5).

### 16.7 Contextual Retrieval (Chunk-Level Context Injection)

Prepend LLM-generated context (what this chunk is about, in relation to the whole document) to each chunk **before** embedding — improves retrieval when chunks are ambiguous out of context (e.g., "the company reported a 3% increase" — increase in *what*, from *when*?).

```python
# Pseudocode pattern (build via a custom Transformation in IngestionPipeline)
def add_contextual_prefix(node, full_doc_text, llm):
    context = llm.complete(
        f"Document:\n{full_doc_text}\n\nChunk:\n{node.text}\n\n"
        f"Give a 1-2 sentence context situating this chunk within the document."
    )
    node.text = f"{context.text}\n\n{node.text}"
    return node
```

### 16.8 Corrective / Self-RAG Patterns

Add a **grading step**: after retrieval, an LLM (or classifier) judges whether retrieved context is actually relevant/sufficient; if not, fall back to a broader search, query rewrite, or web search tool. Implement via a `Workflow` (§19) with conditional branching — this is exactly the kind of multi-step, stateful logic Workflows are designed for.

### 16.9 Advanced RAG Pattern Selection Guide

| Symptom | Likely Fix |
|---|---|
| Retrieves right doc, wrong section | Reduce chunk size, add reranking |
| Retrieves fragmented, incomplete context | Sentence-window or auto-merging retrieval |
| Misses exact terms/codes/acronyms | Hybrid search (BM25 + vector) |
| Query phrasing mismatches document phrasing | HyDE, query rewriting, `QuestionsAnsweredExtractor` |
| Chunks are ambiguous without document context | Contextual retrieval (context prefix injection) |
| Multi-hop reasoning needed | PropertyGraphIndex, SubQuestionQueryEngine, or an Agent/Workflow |
| Irrelevant retrieval passed through to LLM | Add grading/corrective step, similarity cutoff, reranker |
