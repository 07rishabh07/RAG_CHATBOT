from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from .rag import (
        answer_question,
        build_rag,
    )
except ImportError:
    from rag import (
        answer_question,
        build_rag,
    )

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


all_chunks, index = build_rag()
print("RAG system loaded")


@app.get("/")
def home():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask_question(q: Query):
    return answer_question(q.question, all_chunks, index)
