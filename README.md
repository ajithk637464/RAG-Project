# Ask My Cookbook

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a cookbook PDF, using only the cookbook's own text — never invented information.

## How it works

1. **Ingest** (`ingest.py`) — extracts text from `data/*.pdf` page by page, detects a likely recipe name per page, and splits each page's text into overlapping chunks.
2. **Embed + store** (`rag.py`) — each chunk is turned into a 384-dimension vector with a local embedding model (`all-MiniLM-L6-v2`) and stored in ChromaDB alongside its source/page/recipe metadata.
3. **Retrieve** (`rag.py: retrieve()`) — a question is embedded the same way, and ChromaDB returns the `TOP_K` most similar chunks (nearest vectors = most relevant text).
4. **Generate** (`rag.py: generate_answer()`) — only those retrieved chunks are sent to the LLM (`gpt-4o-mini`), with a system prompt that forbids answering from anything else. If nothing relevant was retrieved, the app returns "I couldn't find that in the cookbook." instead of guessing.
5. **UI** (`app.py`) — a Streamlit chat interface that shows the answer plus a citation for every source chunk used (filename, page, recipe name), with the raw chunks viewable in a debug expander.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`.

**LLM provider** (set via `LLM_PROVIDER` in `.env`, defaults to `ollama`):

- **`ollama` (default, free)** — install [Ollama](https://ollama.com), then pull a model:
  ```
  ollama pull llama3.2
  ```
  Ollama runs its own local server (`http://localhost:11434`) — no API key needed.
- **`openai` (paid)** — set `LLM_PROVIDER=openai` and put a real key in `OPENAI_API_KEY`:
  ```
  OPENAI_API_KEY=sk-...
  ```
  Requires billing set up on the OpenAI account, or every request fails with a `429 insufficient_quota` error.

Add a cookbook PDF to `data/`.

## Running

Build the vector index, then launch the chat UI:

```
python rag.py        # builds/rebuilds the ChromaDB index for the active chunk config
streamlit run app.py
```

## Configuration (`config.py`)

Everything tunable lives in one file:

- `ACTIVE_CHUNK_CONFIG` — `"small"` (500 chars / 50 overlap) or `"large"` (1000 chars / 100 overlap). Each config gets its own ChromaDB collection, so they never mix.
- `TOP_K` — how many chunks are retrieved per question. Higher = more context but more chance of irrelevant text diluting the answer; lower = more precise but may miss the right chunk.
- `EMBEDDING_MODEL_NAME`, `LLM_MODEL` — model names.

## Chunk size experiment (Phase 10)

`test_rag.py` runs the same 4 test questions against both the `small` and `large` chunk configs and prints what each retrieves, without touching `config.py` on disk. Observed on the sample cookbook:

- Both configs correctly surfaced the right recipe's page as the #1 result for direct (A), reworded (B), and other-recipe (C) questions.
- For a question with no answer in the cookbook (D), retrieval distances were noticeably higher (~1.27–1.45) than for in-book questions (~0.47–0.95) in both configs — the LLM's system prompt is what turns that into an explicit refusal, since distance alone isn't used as a hard cutoff.
- `small` produces more, more targeted chunks (51 total); `large` produces fewer, larger chunks with more surrounding context per chunk (31 total). With this cookbook's short recipe pages, both configs retrieved the same top recipe, but `small` chunks are tighter around the exact matching sentence while `large` chunks carry more of the page around it.

## Known limitations

- Recipe-name detection is a heuristic (title-case lines near the top of a page) and doesn't resolve every page layout — some pages return no detected recipe name.
- A few special characters (e.g. accented letters) can render as `�` due to a font-decoding limitation in `pypdf` for this particular PDF.
- Full end-to-end answer generation requires a real `OPENAI_API_KEY` in `.env`.
