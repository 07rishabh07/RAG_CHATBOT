from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 🔥 import your RAG functions
from rag import (
    load_documents,
    chunk_text,
    create_embeddings,
    build_faiss_index,
    retrieve,
    generate_answer,
    is_grounded,
    model
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str


# ---------------- LOAD RAG ON START ----------------
documents = load_documents()

all_chunks = []
for doc in documents:
    all_chunks.extend(chunk_text(doc))

embeddings = create_embeddings(all_chunks)
index = build_faiss_index(embeddings)

print("✅ RAG system loaded")


# ---------------- ROUTES ----------------
@app.get("/")
def home():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask_question(q: Query):
    query = q.question

    # retrieval
    retrieved_chunks = retrieve(query, index, all_chunks)

    # generation
    raw_answer = generate_answer(query, retrieved_chunks)

    # hallucination guard
    if not is_grounded(raw_answer, retrieved_chunks):
        final_answer = "I don't know"
    else:
        final_answer = raw_answer

    return {
        "question": query,
        "answer": final_answer,
        "retrieved_context": retrieved_chunks
    }