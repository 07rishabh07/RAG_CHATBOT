# 🧠 RAG Chatbot (End-to-End AI System)

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers questions using custom documents while strictly avoiding hallucinations.

---

## 🚀 Features

- 🔍 Semantic search using SentenceTransformers
- ⚡ Fast similarity search with FAISS
- 🧠 LLM-based answer generation
- ❌ Hallucination prevention using grounding validation
- 🌐 FastAPI backend for API access
- 💻 Simple frontend for user interaction

---

## 🏗️ Architecture

User Query  
→ Embedding (SentenceTransformers)  
→ FAISS Retrieval  
→ Context Selection  
→ LLM (Ollama / API)  
→ Grounding Check  
→ Final Answer  

---

## 🛠️ Tech Stack

- Python  
- FastAPI  
- FAISS  
- SentenceTransformers  
- Ollama / OpenRouter (LLM)  
- HTML, JavaScript  

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/07rishabh07/RAG_CHATBOT.git
cd RAG_CHATBOT
