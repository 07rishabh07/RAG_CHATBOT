import re
from pathlib import Path

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    np = None
    SentenceTransformer = None

try:
    import ollama
except ImportError:
    ollama = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "phi"

_embedding_model = None

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def normalize_text(text):
    replacements = {
        "â€”": "-",
        "â€œ": '"',
        "â€": '"',
        "â€™": "'",
        "â€˜": "'",
        "â‚¹": "Rs.",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_text(text):
    text = normalize_text(text)
    text = re.sub(r"#.*", "", text)
    text = re.sub(r"Version:.*", "", text)
    text = re.sub(r"Audience:.*", "", text)
    text = re.sub(r"Last Updated:.*", "", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def chunk_text(text, chunk_size=80, overlap=20):
    text = clean_text(text)
    words = text.split()
    chunks = []

    if not words:
        return chunks

    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def load_documents(folder_path=DATA_DIR):
    docs = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    for file_path in sorted(folder.glob("*.md")):
        docs.append(file_path.read_text(encoding="utf-8"))

    return docs


def get_embedding_model():
    global _embedding_model

    if SentenceTransformer is None:
        return None

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model


def create_embeddings(chunks):
    model = get_embedding_model()

    if model is None:
        return None

    return model.encode(chunks, normalize_embeddings=True)


def build_faiss_index(embeddings):
    if embeddings is None or faiss is None or np is None:
        return None

    if len(embeddings) == 0:
        raise ValueError("Cannot build a FAISS index without embeddings.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))

    return index


def lexical_retrieve(query, chunks, k=3):
    query_terms = set(tokenize(query))

    if not query_terms:
        return chunks[:k]

    scored_chunks = []
    for chunk in chunks:
        chunk_terms = tokenize(chunk)
        chunk_term_set = set(chunk_terms)
        overlap = query_terms & chunk_term_set
        score = len(overlap) + sum(chunk_terms.count(term) for term in overlap) * 0.1
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:k]] or chunks[:k]


def score_text(text, query_terms):
    text_terms = tokenize(text)
    if not text_terms:
        return 0

    text_term_set = set(text_terms)
    overlap = query_terms & text_term_set
    return len(overlap) + sum(text_terms.count(term) for term in overlap) * 0.1


def retrieve(query, index, chunks, k=3):
    if not chunks:
        return []

    model = get_embedding_model() if index is not None else None

    if model is None or index is None:
        return lexical_retrieve(query, chunks, k)

    query_embedding = model.encode([query], normalize_embeddings=True)
    result_count = min(k, len(chunks))
    distances, indices = index.search(query_embedding, result_count)

    return [chunks[i] for i in indices[0] if i != -1]


def clean_answer_candidate(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\s*[-*\d.)]+\s*", "", text)
    text = re.sub(r"^A:\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"\s+Q:\s+", text, maxsplit=1)[0].strip()
    return text.strip(" -")


def extract_qa_answers(query, context_chunks):
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    text = "\n".join(context_chunks)
    qa_matches = re.findall(
        r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\s+Q:|\s+\(End of document\)|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    scored_answers = []
    for question, answer in qa_matches:
        question_score = score_text(question, query_terms)
        if question_score <= 0:
            continue

        score = question_score
        answer = clean_answer_candidate(answer)
        if answer:
            scored_answers.append((score, answer))

    scored_answers.sort(key=lambda item: item[0], reverse=True)
    if not scored_answers:
        return []

    top_score = scored_answers[0][0]
    return [answer for score, answer in scored_answers if score == top_score]


def extractive_answer(query, context_chunks):
    query_terms = set(tokenize(query))

    if not context_chunks or not query_terms:
        return "I don't know"

    qa_answers = extract_qa_answers(query, context_chunks)
    if qa_answers:
        return "\n".join(format_bullets(qa_answers[:2]))

    sentences = []
    for chunk in context_chunks:
        chunk = re.sub(r"\s+Q:\s+.*", "", chunk)
        sentences.extend(re.split(r"(?<=[.!?])\s+|(?=\s+-\s+)", chunk))

    scored_sentences = []
    for sentence in sentences:
        sentence = clean_answer_candidate(sentence)
        if not sentence:
            continue
        score = score_text(sentence, query_terms)
        if score:
            scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    best_sentences = []
    seen = set()
    for _, sentence in scored_sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        best_sentences.append(sentence)
        seen.add(normalized)
        if len(best_sentences) == 3:
            break

    if not best_sentences:
        return "I don't know"

    return "\n".join(format_bullets(best_sentences))


def format_bullets(items):
    bullets = []
    seen = set()

    for item in items:
        item = clean_answer_candidate(item)
        if not item:
            continue

        normalized = item.lower()
        if normalized in seen:
            continue

        bullets.append(f"- {item}")
        seen.add(normalized)

    return bullets


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

    if ollama is not None:
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception:
            pass

    return extractive_answer(query, context_chunks)


def is_grounded(answer, context_chunks):
    if answer == "I don't know":
        return True

    context = " ".join(context_chunks).lower()
    answer_words = tokenize(answer)

    if not answer_words:
        return False

    match_count = sum(1 for word in answer_words if word in context)
    return match_count / len(answer_words) > 0.45


def build_rag():
    documents = load_documents()

    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc))

    if not chunks:
        raise RuntimeError("No document chunks were loaded. Add markdown files to RAG/data.")

    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    return chunks, index


def answer_question(query, chunks, index):
    retrieved_chunks = retrieve(query, index, chunks)
    raw_answer = generate_answer(query, retrieved_chunks)
    final_answer = raw_answer if is_grounded(raw_answer, retrieved_chunks) else "I don't know"

    return {
        "question": query,
        "answer": final_answer,
        "retrieved_context": retrieved_chunks,
    }


if __name__ == "__main__":
    all_chunks, faiss_index = build_rag()
    result = answer_question("What causes construction delays?", all_chunks, faiss_index)

    print("\nQuery:", result["question"])
    print("\nRetrieved Context:\n")
    for i, context in enumerate(result["retrieved_context"]):
        print(f"{i + 1}. {context}\n")
    print("\nFinal Answer:\n")
    print(result["answer"])
