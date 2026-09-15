# MiniVec

A lightweight C++ vector retrieval engine implementing approximate nearest-neighbor search, hybrid retrieval, metadata filtering, persistence, and concurrent REST serving.

## Features

- **HNSW** approximate nearest-neighbor search
- Exact **brute-force** search for recall evaluation
- Configurable distance metrics
- **Metadata filtering**
- Persistent index / graph storage
- **BM25** lexical retrieval
- Hybrid semantic + lexical retrieval using **RRF**
- Multithreaded REST API
- Concurrent search/write stress testing
- Benchmarking across dataset sizes and recall/latency trade-offs

## Architecture

```text
Client
  │
  ▼
REST API
  │
  ▼
Retrieval Engine
  ├── HNSW ANN Search
  ├── Exact Search
  ├── Metadata Filtering
  ├── BM25 Search
  └── Hybrid Search
        │
        └── Reciprocal Rank Fusion (RRF)
              │
              ▼
        Persistent Index
```

## Performance

Benchmarked on 384-dimensional vectors.

| Workload    | Configuration | Latency     | Recall     |
| ----------- | ------------- | ----------- | ---------- |
| 1K vectors  | HNSW, ef=50   | **0.84 ms** | **93.04%** |
| 1K vectors  | HNSW, ef=100  | **1.02 ms** | **99.34%** |
| 1K vectors  | HNSW, ef=200  | **1.17 ms** | **100%**   |
| 10K vectors | HNSW, ef=200  | **5.29 ms** | **81.54%** |

The engine was also validated under concurrent search/write workloads and persistence restart tests.

## RAG Integration

MiniVec was integrated into a RAG pipeline and evaluated using RAGAS:

| Metric            | Previous | MiniVec   |
| ----------------- | -------- | --------- |
| Faithfulness      | 0.561    | **0.759** |
| Context Precision | 0.720    | **0.833** |
| Context Recall    | 0.670    | 0.587     |

The retrieval backend improved faithfulness and context precision, while context recall decreased in the tested configuration.

## Build

<pre class="overflow-visible! px-0!" data-start="1906" data-end="1949"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>g</span><span class="ͼv">++</span><span></span><span class="ͼ12">-std</span><span class="ͼv">=</span><span>c</span><span class="ͼv">++</span><span class="ͼy">17</span><span></span><span class="ͼ12">-O2</span><span></span><span class="ͼ12">-pthread</span><span> ...</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

See the source files and benchmark scripts for the complete build and evaluation workflow.

## Tech Stack

**C++17 · HNSW · BM25 · REST · Multithreading · Persistence · RRF · JSON · Linux/Windows**
