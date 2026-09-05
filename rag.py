"""
Phases 4-7: embeddings, ChromaDB storage/retrieval, and LLM answer generation.

An embedding is a fixed-length vector (list of numbers) that represents the
*meaning* of a piece of text. Texts with similar meaning end up with vectors
that are close together in space, even if they share no exact words. That's
what lets us search by meaning instead of exact keyword matching.
"""
import re

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config
import ingest

_model = None
_client = None
_openai_client = None
_bm25 = None
_bm25_records = []
_bm25_collection_name = None
RRF_K = 60

# The LLM is only allowed to answer from these chunks - never from its own
# training knowledge. This is what stops it from inventing recipes.
SYSTEM_PROMPT = """You are "Ask My Cookbook", an assistant that answers questions using ONLY the cookbook excerpts given to you as context below.

Rules you must follow:
- Answer strictly using the provided context. Never use outside knowledge.
- Never invent or guess a recipe, ingredient, quantity, cooking time, or step that is not explicitly present in the context.
- If the context does not contain enough information to answer, respond with exactly: "I couldn't find that in the cookbook." Do not try to answer anyway.
- Keep answers concise and directly grounded in the text provided."""


def get_embedding_model() -> SentenceTransformer:
    """Load the local embedding model once and reuse it (loading is slow)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of text strings into a list of embedding vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Open (or create) the on-disk ChromaDB store at config.CHROMA_DB_DIR.

    PersistentClient writes to disk, so embeddings survive between app runs —
    we only have to embed the cookbook once, not every time Streamlit starts.
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    return _client


def get_collection():
    """
    Get (or create) the collection for the currently active chunk config.

    Named via config.COLLECTION_NAME (e.g. "cookbook_small"), so switching
    config.ACTIVE_CHUNK_CONFIG points us at a different, separate collection
    instead of mixing chunks from two different chunk sizes together.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name=config.COLLECTION_NAME)


def store_chunks(chunks: list[dict]) -> int:
    """
    Embed a list of chunks (from ingest.chunk_pages) and store them in ChromaDB.

    For each chunk we store four things together, keyed by the same id:
    - the embedding (what ChromaDB actually searches over)
    - the raw chunk text (so we can hand it to the LLM later)
    - metadata: source filename, page number, recipe name (so we can cite it)
    """
    if not chunks:
        return 0

    collection = get_collection()
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    # Chroma metadata values can't be None, so an undetected recipe name
    # becomes "" rather than being left out.
    metadatas = [
        {
            "source": c["source"],
            "page": c["page"],
            "recipe_name": c["recipe_name"] or "",
        }
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    _set_bm25_index(chunks)
    return len(chunks)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _set_bm25_index(chunks: list[dict]) -> None:
    global _bm25, _bm25_records, _bm25_collection_name
    _bm25_records = chunks
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks]) if chunks else None
    _bm25_collection_name = config.COLLECTION_NAME


def _get_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Load a BM25 index from the active persisted Chroma collection when needed."""
    if _bm25_collection_name == config.COLLECTION_NAME and _bm25 is not None:
        return _bm25, _bm25_records

    collection = get_collection()
    stored = collection.get(include=["documents", "metadatas"])
    records = [
        {
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "recipe_name": metadata.get("recipe_name") or None,
            "id": chunk_id,
        }
        for chunk_id, text, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"]
        )
    ]
    _set_bm25_index(records)
    return _bm25, _bm25_records


def build_index(reset: bool = True) -> int:
    """
    Full Phase 5 pipeline: load PDFs -> chunk -> embed -> store in ChromaDB.

    reset=True deletes any existing collection for the active chunk config
    first. Without this, re-running ingestion would add duplicate copies of
    every chunk on top of what's already stored.
    """
    client = get_chroma_client()
    if reset:
        global _bm25, _bm25_records, _bm25_collection_name
        _bm25 = None
        _bm25_records = []
        _bm25_collection_name = None
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet - nothing to delete

    pages = ingest.load_all_pdfs()
    chunks = ingest.chunk_pages(pages)
    stored = store_chunks(chunks)
    return stored


def retrieve_semantic(question: str, top_k: int = None) -> list[dict]:
    """
    Phase 6: embed a question and find the TOP_K most similar chunks.

    This is the "R" in RAG - we don't hand the whole cookbook to the LLM,
    just the handful of chunks most likely to contain the answer.
    """
    top_k = top_k or config.TOP_K
    collection = get_collection()
    query_vector = embed_texts([question])
    results = collection.query(query_embeddings=query_vector, n_results=top_k)

    retrieved = []
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "id": chunk_id,
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "recipe_name": meta["recipe_name"] or None,
            "distance": dist,
            "semantic_rank": len(retrieved) + 1,
            "bm25_rank": None,
            "rrf_score": None,
            "search_type": "semantic",
        })
    return retrieved


def retrieve_bm25(question: str, top_k: int = None) -> list[dict]:
    """Find keyword-overlapping chunks with BM25."""
    top_k = top_k or config.TOP_K
    index, records = _get_bm25_index()
    if index is None:
        return []

    scores = index.get_scores(_tokenize(question))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            **records[i],
            "distance": None,
            "semantic_rank": None,
            "bm25_rank": rank,
            "bm25_score": float(scores[i]),
            "rrf_score": None,
            "search_type": "bm25",
        }
        for rank, i in enumerate(ranked_indices, start=1)
    ]


def retrieve(question: str, top_k: int = None, hybrid: bool = True) -> list[dict]:
    """Retrieve semantic-only results or combine semantic and BM25 with RRF."""
    top_k = top_k or config.TOP_K
    semantic = retrieve_semantic(question, top_k)
    if not hybrid:
        return semantic

    bm25 = retrieve_bm25(question, top_k)
    combined = {}
    for result in semantic + bm25:
        key = result["id"] if "id" in result else (
            result["source"], result["page"], result["text"]
        )
        item = combined.setdefault(
            key,
            {**result, "semantic_rank": None, "bm25_rank": None, "rrf_score": 0.0},
        )
        if result["search_type"] == "semantic":
            item["semantic_rank"] = result["semantic_rank"]
            item["distance"] = result["distance"]
        else:
            item["bm25_rank"] = result["bm25_rank"]
            item["bm25_score"] = result["bm25_score"]
        rank = result["semantic_rank"] or result["bm25_rank"]
        item["rrf_score"] += 1 / (RRF_K + rank)
        item["search_type"] = "hybrid"

    return sorted(combined.values(), key=lambda item: item["rrf_score"], reverse=True)[:top_k]


def get_llm_client() -> OpenAI:
    """
    Both providers are reached through the OpenAI SDK - Ollama exposes an
    OpenAI-compatible endpoint, so the only difference is base_url/api_key.
    """
    global _openai_client
    if _openai_client is None:
        if config.LLM_PROVIDER == "ollama":
            _openai_client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
        else:
            if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-openai-api-key-here":
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add your real key to the .env file "
                    "(never commit it - .env is already gitignored)."
                )
            _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into labeled blocks the LLM can cite from."""
    blocks = []
    for c in chunks:
        label = f"[{c['source']} - Page {c['page']}"
        if c["recipe_name"]:
            label += f" - Recipe: {c['recipe_name']}"
        label += "]"
        blocks.append(f"{label}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Phase 7: ask the LLM to answer using ONLY the retrieved chunks.

    If nothing was retrieved, we don't even call the LLM - there's nothing
    it could truthfully answer from, so we return the refusal directly.
    """
    if not chunks:
        return "I couldn't find that in the cookbook."

    context = build_context(chunks)
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
            ],
        )
    except Exception as e:
        if config.LLM_PROVIDER == "ollama":
            raise RuntimeError(
                f"Could not reach Ollama at {config.OLLAMA_BASE_URL}. Make sure Ollama is "
                f"installed and running, and the model is pulled "
                f"(`ollama pull {config.LLM_MODEL}`). Original error: {e}"
            ) from e
        raise
    return response.choices[0].message.content.strip()


def answer_question(question: str, top_k: int = None) -> dict:
    """Full RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""
    chunks = retrieve(question, top_k)
    answer = generate_answer(question, chunks)
    return {"question": question, "answer": answer, "chunks": chunks}


if __name__ == "__main__":
    import numpy as np

    samples = [
        "2 cups basmati rice",              # a fact from a recipe
        "How much rice do I need?",          # semantically related question
        "Preheat the oven to 180 degrees",   # unrelated cooking instruction
    ]
    vectors = embed_texts(samples)
    print(f"Embedding dimension: {len(vectors[0])}")

    a, b, c = (np.array(v) for v in vectors)
    print(f"\nSimilarity('{samples[0]}', '{samples[1]}') = {np.dot(a, b):.3f}  <- related, should be higher")
    print(f"Similarity('{samples[0]}', '{samples[2]}') = {np.dot(a, c):.3f}  <- unrelated, should be lower")

    print(f"\nActive config: {config.ACTIVE_CHUNK_CONFIG} -> collection '{config.COLLECTION_NAME}'")
    stored = build_index(reset=True)
    print(f"Stored {stored} chunks in ChromaDB at {config.CHROMA_DB_DIR}")

    # Phase 6 check: retrieve() using the real function the app will call.
    test_question = "How do I make pasta primavera?"
    chunks = retrieve(test_question)
    print(f"\nTest query: '{test_question}'")
    print(f"TOP_K = {config.TOP_K} -> the {config.TOP_K} most similar chunks:")
    for i, c in enumerate(chunks, start=1):
        print(f"\n#{i} distance={c['distance']:.3f} | {c['source']} p.{c['page']} | recipe: {c['recipe_name'] or 'unknown'}")
        print(c["text"][:150])

    # Phase 7 check: full RAG answer, only if a real OpenAI key is configured.
    print("\n--- Phase 7: LLM answer generation ---")
    try:
        result = answer_question(test_question)
        print(f"\nAnswer: {result['answer']}")
    except RuntimeError as e:
        print(f"Skipped: {e}")
