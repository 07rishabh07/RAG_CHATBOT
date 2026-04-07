import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


# ---------------- CLEANING ----------------
def clean_text(text):
    text = re.sub(r'#.*', '', text)
    text = re.sub(r'Version:.*', '', text)
    text = re.sub(r'Audience:.*', '', text)
    text = re.sub(r'Last Updated:.*', '', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()


# ---------------- CHUNKING ----------------
def chunk_text(text, chunk_size=80, overlap=20):
    text = clean_text(text)

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ---------------- LOAD DOCS ----------------
def load_documents():
    docs = []
    folder_path = "data"

    for file in os.listdir(folder_path):
        if file.endswith(".md"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                docs.append(f.read())

    return docs


# ---------------- EMBEDDINGS ----------------
def create_embeddings(chunks):
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return embeddings


# ---------------- FAISS ----------------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)  # cosine similarity
    index.add(np.array(embeddings))

    return index


# ---------------- RETRIEVAL ----------------
def retrieve(query, index, chunks, k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)

    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]
    return results


# ---------------- LLM ----------------
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a strict extraction-based AI system.

Rules:
- Answer ONLY using the context
- Do NOT add outside knowledge
- Do NOT explain
- If answer is not present, say "I don't know"
- Return bullet points only

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model='phi',
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']


# ---------------- HALLUCINATION CHECK ----------------
def is_grounded(answer, context_chunks):
    context = " ".join(context_chunks).lower()
    answer_words = answer.lower().split()

    match_count = sum(1 for word in answer_words if word in context)

    # allow some flexibility (not 100% strict)
    return match_count / max(len(answer_words), 1) > 0.6


# ---------------- MAIN ----------------
if __name__ == "__main__":
    documents = load_documents()

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    embeddings = create_embeddings(all_chunks)
    index = build_faiss_index(embeddings)

    query = "What causes construction delays?"

    retrieved_chunks = retrieve(query, index, all_chunks)

    raw_answer = generate_answer(query, retrieved_chunks)

    # 🔥 hallucination guard
    if not is_grounded(raw_answer, retrieved_chunks):
        final_answer = "I don't know"
    else:
        final_answer = raw_answer

    # ---------------- OUTPUT ----------------
    print("\nQuery:", query)

    print("\nRetrieved Context:\n")
    for i, r in enumerate(retrieved_chunks):
        print(f"{i+1}. {r}\n")

    print("\nFinal Answer:\n")
    print(final_answer)