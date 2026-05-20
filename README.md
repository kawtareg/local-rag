# 📚 RAG PDF — Question Answering on pdfs

Ask questions to your PDF documents using a local LLM.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Ollama](https://img.shields.io/badge/Ollama-local-black)
![Mistral](https://img.shields.io/badge/Model-Mistral_7B-purple)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Demo

![Demo RAG](<assets/Demo_rag.gif>)

---

## Features

- Load an entire folder of PDFs in one command
- Semantic search across all documents using embeddings
- Multi-turn conversation with memory
- Streaming responses token by token
- Source display — see which file and page was used for each answer
- Fully local — no API key, no data sent to the cloud

---

## Stack

| Tool | Role |
|---|---|
| Python 3.12 | Language |
| [Ollama](https://ollama.com) + Mistral 7B | Local LLM inference |
| [ChromaDB](https://www.trychroma.com) | Vector database |
| [sentence-transformers](https://www.sbert.net) | Embeddings (all-MiniLM-L6-v2) |
| [pypdf](https://pypdf.readthedocs.io) | PDF text extraction |
| openai SDK | API client (Ollama-compatible) |

---

## How it works

```
PDFs → Text extraction → Chunking → Embeddings → ChromaDB
                                                      ↓
Question → Embedding → Semantic search → Top 3 chunks
                                              ↓
                          Question + Context → Mistral → Answer
```

1. **Indexing** — PDFs are split into overlapping chunks, embedded and stored in ChromaDB
2. **Retrieval** — the question is embedded and the most similar chunks are retrieved
3. **Generation** — the LLM answers based only on the retrieved context

---

## Installation

### Prerequisites

- Mac with [Homebrew](https://brew.sh) or equivalent
- Python 3.12+

### 1. Clone the project

```bash
git clone https://github.com/TON_PSEUDO/rag-pdf
cd rag-pdf
```

### 2. Install Ollama and the model

```bash
brew install ollama
brew services start ollama
ollama pull mistral
```

### 3. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Add your HuggingFace token (optional but recommended)

```bash
echo "HF_TOKEN=hf_..." > .env
```

Get a free token at [huggingface.co](https://huggingface.co) → Settings → Access Tokens.

---

## Usage

```bash
python3 main.py --folder ./your-pdf-folder/
```

```
Loading ./your-pdf-folder/...
548 chunks created
Ready! Type 'quit' to exit

>>> What is the difference between TCP and UDP?
TCP is a connection-oriented protocol that guarantees delivery...

Sources:
  [1] RI_CM3_TCP.pdf — page 4 : "TCP establishes a connection via three-way handshake..."
  [2] RI_CM3_TCP.pdf — page 7 : "UDP is a connectionless protocol, faster but unreliable..."

>>> quit
```

### Commands

| Command | Action |
|---|---|
| Any question | Ask the model |
| `quit` | Exit |

---

## Project structure

```
rag-pdf/
├── main.py              # CLI entry point
├── config.py            # model and chunking parameters
├── rag/
│   ├── loader.py        # PDF loading and text chunking
│   ├── embedder.py      # embedding generation and ChromaDB storage
│   ├── retriever.py     # semantic search
│   └── generator.py     # LLM answer generation with streaming
├── db/                  # ChromaDB vector store (git-ignored)
├── requirements.txt
└── .gitignore
```

---

## Configuration

Edit `config.py` to tune the pipeline:

```python
MODEL = "mistral"           # any Ollama model
TEMPERATURE = 0.3           # 0 = deterministic, 1 = creative
CHUNK_SIZE = 500            # characters per chunk
CHUNK_OVERLAP = 100         # overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # HuggingFace embedding model
```

---

## What I learned

- RAG pipeline architecture (indexing vs retrieval vs generation)
- Vector embeddings and semantic search with ChromaDB
- PDF text extraction and chunking strategies
- Metadata storage and retrieval in vector databases
- Streaming LLM responses
- Modular Python project structure

---

## Roadmap

- [ ] Re-indexing detection (skip if already indexed)
- [ ] Semantic chunking
- [ ] Relevance score display
- [ ] Support for more document formats (txt, md, docx)

---

## Author

**Kawtar El Gueddari** — [GitHub](https://github.com/kawtareg) · INSA Rouen Normandie
