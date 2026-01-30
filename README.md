# GraphRAG-Lite

<p align="center">
  <img src="https://github.com/shibing624/graphrag-lite/blob/main/docs/logo.svg" alt="GraphRAG-Lite Logo" width="400">
</p>

<p align="center">
  <b>Minimal GraphRAG implementation in ~500 lines of Python code.</b>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/graphrag-lite"><img src="https://badge.fury.io/py/graphrag-lite.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/shibing624/graphrag-lite/blob/main/README.md"><img src="https://img.shields.io/badge/wechat-group-green.svg?logo=wechat" alt="Chat Group"></a>
</p>

<p align="center">
  <a href="https://github.com/shibing624/graphrag-lite/blob/main/README_zh.md">中文文档</a>
</p>

GraphRAG-Lite is a clean, educational implementation of GraphRAG (Graph-based Retrieval-Augmented Generation). Perfect for learning the core principles of knowledge graph enhanced RAG systems.

## Why GraphRAG-Lite?

- **Learn by Reading**: Clean, well-documented code you can understand in an afternoon
- **Production Patterns**: Real-world optimizations like batch embeddings and LLM caching
- **Flexible Retrieval**: 4 query modes for different use cases
- **Minimal Dependencies**: Just `openai`, `numpy`, `tiktoken`, and `loguru`

## Features

| Feature | Description |
|---------|-------------|
| **4 Query Modes** | `local`, `global`, `mix`, `naive` - choose the right strategy |
| **Batch Embeddings** | Reduce API calls with intelligent batching |
| **LLM Caching** | Avoid redundant LLM requests |
| **Streaming Output** | Real-time response streaming |
| **NumPy Acceleration** | Fast vector similarity search |
| **Persistent Storage** | JSON-based storage, no external database needed |

## Installation

```bash
pip install graphrag-lite
```

Or install from source:

```bash
git clone https://github.com/shibing624/graphrag-lite.git
cd graphrag-lite
pip install -e .
```

## Quick Start

```python
import os
from graphrag_lite import GraphRAGLite

# Initialize
graph = GraphRAGLite(
    storage_path="./my_graph",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),  # Optional: for compatible APIs
)

# Insert documents
graph.insert("""
Charles Dickens wrote "A Christmas Carol" in 1843.
The story features Ebenezer Scrooge, a miserly old man,
and the ghost of his former business partner Jacob Marley.
""")

# Query with knowledge graph context
answer = graph.query("What is the relationship between Scrooge and Marley?")
print(answer)
```

## Query Modes

| Mode | Strategy | Best For |
|------|----------|----------|
| `local` | Entity → Related relations | "Who is X?" questions |
| `global` | Relation → Related entities | "How are X and Y related?" |
| `mix` | Entity + Relation + Chunks | **General purpose (recommended)** |
| `naive` | Text chunks only | Baseline comparison |

```python
# Choose the right mode for your question
answer = graph.query("Who is Scrooge?", mode="local")
answer = graph.query("How are Scrooge and Marley connected?", mode="global")
answer = graph.query("Tell me about the story", mode="mix")      # Recommended
answer = graph.query("What happened?", mode="naive")
```

## Streaming Output

```python
for chunk in graph.query("Who is Scrooge?", stream=True):
    print(chunk, end="", flush=True)
```

## API Reference

### GraphRAGLite

```python
GraphRAGLite(
    storage_path: str = "./graphrag_data",  # Data storage directory
    api_key: str = None,                     # OpenAI API key
    base_url: str = None,                    # OpenAI-compatible API base URL
    model: str = "gpt-4o-mini",              # LLM model
    embedding_model: str = "text-embedding-3-small",  # Embedding model
    enable_cache: bool = True,               # Enable LLM response caching
)
```

### Methods

| Method | Description |
|--------|-------------|
| `insert(text, doc_id=None)` | Insert document and build knowledge graph |
| `query(question, mode="mix", top_k=10, stream=False)` | Query the knowledge graph |
| `has_data()` | Check if graph has data |
| `get_stats()` | Get graph statistics |
| `list_entities()` | List all entities |
| `list_relations()` | List all relations |
| `clear()` | Clear all data |

## How It Works

<p align="center">
  <img src="https://github.com/shibing624/graphrag-lite/blob/main/docs/workflow.svg" alt="GraphRAG-Lite Workflow" width="800">
</p>

**Insert Pipeline:**
```
Document → Chunking → LLM Entity Extraction → Batch Embedding → Storage
```

**Query Pipeline:**
```
Question → Vector Search → Context Building → LLM Generation → Answer
```

## Use Cases

- **Learning GraphRAG**: Understand how knowledge graphs enhance RAG
- **Prototyping**: Quickly validate GraphRAG for your domain
- **Research**: Baseline for comparing retrieval strategies
- **Education**: Teaching material for RAG concepts

## Community & Support

*   **GitHub Issues**: [Submit an issue](https://github.com/shibing624/graphrag-lite/issues)
*   **WeChat**: Add `xuming624` with note "llm" to join the LLM tech wechat group

<img src="https://github.com/shibing624/graphrag-lite/blob/main/docs/wechat.jpeg" width="200" />

## License

Apache License 2.0

## Citation

```bibtex
@software{graphrag-lite,
  author = {Xu Ming},
  title = {GraphRAG-Lite: Minimal GraphRAG Implementation},
  year = {2025},
  url = {https://github.com/shibing624/graphrag-lite}
}
```
