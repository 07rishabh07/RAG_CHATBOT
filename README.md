# RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers questions from the markdown documents in `RAG/data`.

The project works out of the box with Python's standard library. If you install the optional ML dependencies, it can also use SentenceTransformers, FAISS, and Ollama.

## Features

- Lexical retrieval with no required third-party packages
- Optional semantic search with SentenceTransformers and FAISS
- Optional LLM answer generation through Ollama
- Basic grounding check to reduce unsupported answers
- Standard-library HTTP backend
- Optional FastAPI backend
- Simple HTML frontend

## Project Structure

```text
RAG_CHATBOT/
  RAG/
    api.py
    rag.py
    server.py
    index.html
    data/
      doc1.md
      doc2.md
      doc3.md
  requirements.txt
```

## Run the Working App

From the repository root:

```bash
python RAG/server.py
```

Open this URL in your browser:

```text
http://127.0.0.1:8000
```

This mode works without installing any dependencies.

## Optional ML Setup

```bash
git clone https://github.com/07rishabh07/RAG_CHATBOT.git
cd RAG_CHATBOT
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install and start Ollama if you want LLM-generated answers, then pull the model used by the app:

```bash
ollama pull phi
```

## Run the Optional FastAPI API

From the repository root:

```bash
uvicorn RAG.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## Test the RAG Pipeline Directly

```bash
python RAG/rag.py
```

## API Example

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What causes construction delays?\"}"
```
