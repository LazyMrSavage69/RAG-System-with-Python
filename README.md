# RAG System — DeepSeek + ChromaDB

A clean, dependency-light Retrieval-Augmented Generation (RAG) pipeline.

## Architecture

```
┌─────────────┐    chunk    ┌──────────────┐   embed   ┌────────────┐
│  Documents  │ ──────────► │    Loader    │ ────────► │  ChromaDB  │
│ PDF/TXT/MD  │             │  (loader.py) │           │ (local DB) │
│   / URLs    │             └──────────────┘           └─────┬──────┘
└─────────────┘                                              │ top-k
                                                             │ retrieve
┌──────────────────────┐    answer    ┌──────────────┐       │
│     User Question    │ ◄─────────── │   DeepSeek   │ ◄─────┘
│                      │              │  (generator) │
└──────────────────────┘              └──────────────┘
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and paste your DeepSeek API key
```

Get your key at: https://platform.deepseek.com/api_keys

### 3. Project structure

```
rag_deepseek/
├── loader.py        # Document loading & chunking (PDF, TXT, MD, URL)
├── vector_store.py  # ChromaDB wrapper + local embeddings
├── generator.py     # DeepSeek API calls
├── rag.py           # Pipeline orchestrator + CLI
├── chat.py          # Interactive chat interface
├── requirements.txt
├── .env.example
└── chroma_db/       # Auto-created by ChromaDB
```

---

## Usage

### Interactive chat (recommended)

```bash
python chat.py
```

Then use commands inside the chat:
```
/ingest path/to/file.pdf
/ingest https://example.com/article
/status
Your question here…
```

### CLI

```bash
# Ingest documents
python rag.py ingest data/report.pdf
python rag.py ingest data/notes.md
python rag.py ingest https://en.wikipedia.org/wiki/Retrieval-augmented_generation

# Ask a question
python rag.py ask "What is the main conclusion of the report?"

# Check how many chunks are indexed
python rag.py status
```

### Python API

```python
from rag import RAGPipeline

pipeline = RAGPipeline()

# Ingest one or many sources
pipeline.ingest("data/report.pdf")
pipeline.ingest("https://example.com")
pipeline.ingest_many(["file1.txt", "file2.md"])

# Ask questions
answer = pipeline.ask("Summarize the key findings.", verbose=True)
print(answer)
```

---

## Configuration

`RAGPipeline` accepts these optional arguments:

| Parameter         | Default                | Description                                  |
|-------------------|------------------------|----------------------------------------------|
| `api_key`         | from `.env`            | DeepSeek API key                             |
| `persist_dir`     | `./chroma_db`          | Where ChromaDB stores its files              |
| `collection_name` | `rag_collection`       | ChromaDB collection name                     |
| `embed_model`     | `all-MiniLM-L6-v2`     | Sentence-transformers model for embeddings   |
| `deepseek_model`  | `deepseek-chat`        | DeepSeek model (`deepseek-chat` or `deepseek-reasoner`) |
| `top_k`           | `5`                    | Number of chunks retrieved per query         |
| `chunk_size`      | `500`                  | Words per chunk                              |
| `chunk_overlap`   | `50`                   | Overlap words between consecutive chunks     |

---

## Notes

- **Embeddings run locally** — no API key or internet needed for indexing.
- **ChromaDB persists to disk** — your indexed documents survive restarts.
- **Duplicate-safe** — re-ingesting the same file won't create duplicates.
- Swap `deepseek_model="deepseek-reasoner"` for harder reasoning tasks.
# RAG-System-with-Python
