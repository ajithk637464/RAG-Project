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
- End-to-end answers use whichever provider `LLM_PROVIDER` selects. The default is a local Ollama model. The OpenAI path needs a real `OPENAI_API_KEY` in `.env`.
- Ingredient checks, substitutes, and calorie numbers in the Week 7 tools are simple lookups. They are not a full nutrition database, and a substitute is not a cookbook fact.

## Week 7: recipe agent and fixed workflow

The cookbook chat is unchanged. Two new ways of answering sit beside it and call the same RAG code.

### Agent architecture

```
User question
    |
    v
Short-term memory (this task only)
    |
    v
Think -- the LLM picks the next action as one JSON object
    |
    v
Tool -- search_recipes / check_ingredients / find_substitute / calculate_nutrition
    |
    v
Observe -- the tool result is written into short-term memory
    |
    +-- repeat until a final answer, or a safeguard stops the loop
```

`agent.py` is a normal Python loop. It does not use LangChain or LangGraph. The model never runs tools itself. It only proposes the next JSON action, and Python executes it.

The fixed workflow in `workflow.py` does the same task without letting the model choose. The order is hard-coded.

Both paths share:

- `rag.py` for retrieval, ChromaDB, and the LLM client
- `tools.py` for the four tools
- `memory.py` for short-term memory and token/cost counters
- `config.py` for `AGENT_MAX_STEPS`, `AGENT_TIMEOUT_SECONDS`, and `AGENT_MAX_TOTAL_TOKENS`

### Tools

| Tool | What it does | Uses the LLM? |
| --- | --- | --- |
| `search_recipes(query)` | Calls the existing RAG pipeline (`retrieve` + `generate_answer`) and saves the cookbook excerpt | Yes |
| `check_ingredients(recipe_text, available_ingredients)` | Lists known ingredients in that excerpt and marks which ones the user already has | No |
| `find_substitute(ingredient)` | Looks up one ingredient in a small built-in table | No |
| `calculate_nutrition(recipe_text)` | Adds a rough calorie estimate for recognized ingredients | No |

`search_recipes` is the only tool that reads the cookbook. When the question names a retrieved recipe, only that recipe's chunks are kept, so the contents page is not treated as an ingredient list. If no retrieved recipe name matches the question, search returns "I couldn't find that in the cookbook." and does not call the model. Substitutes and calories are labeled as table lookups, not cookbook facts. "Pasta" covers pasta shapes, "cheese" covers named cheeses, and "tomato" covers tomato forms, so "I have pasta and cheese" matches conchiglie and cheddar.

If the agent omits `recipe_text`, the loop fills it from short-term memory. The step shown in the UI says `recipe_text=(from short-term memory)`.

### Agent loop

Each step is recorded and shown in the UI:

1. **Think** — the model sees the question plus memory and returns JSON.
2. **Tool** — Python runs that function.
3. **Observe** — the result is stored in memory for the next Think.
4. Repeat until the JSON says `"action": "final"`.

The loop refuses a final answer until `search_recipes` has been called, and it will not search a second time. A repeated call with the same arguments is skipped. On the last allowed step the prompt tells the model to give the final answer. Short strings passed as `recipe_text` are replaced with the excerpt saved in memory.

Safeguards, checked before every new step:

- **MAX_STEPS** (default 8) — stops a model that keeps calling tools
- **Timeout** (default 180 seconds) — stops a run that takes too long
- **Token budget** (default 12,000 total tokens) — stops a run that gets expensive

A stopped run is a failure in the comparison. Its answer explains which safeguard fired.

### Short-term memory

`TaskMemory` holds the goal, the saved recipe excerpt, and the tool observations for the current question. It is created at the start of `run_agent` / `run_workflow` and thrown away when that call returns. It is not a database and it is not shared across questions.

### Fixed workflow

Every question follows the same five stages:

1. **RAG search** — `search_recipes(question)`
2. **Ingredient check** — pantry phrase taken from "I have ..."
3. **Substitute** — one lookup for each missing ingredient, at most three
4. **Nutrition** — `calculate_nutrition` on the saved excerpt
5. **Final answer** — assembled in code from the four tool results. Search is the only model call. If search refused, the final answer is that refusal.

The workflow always attempts this sequence. The agent may skip a tool or stop after search when the cookbook has no matching recipe. That difference is what the comparison measures.

### Comparison

`compare.py` sends the same three questions to both systems:

1. Tuna and Pasta Bake, with tuna, pasta, and cheese on hand. Success: the answer mentions tuna, that recipe, conchiglie, or the Knorr herb stock pot from the page.
2. Pasta Primavera. Success: the answer mentions primavera.
3. Chocolate lava cake, which is not in the cookbook. Success: the answer says it could not be found.

A case is a success only when `status` is `completed` and the final answer contains the expected phrase. Time, tokens, estimated cost, tool calls, and success/failure are printed together.

Cost is $0 for Ollama. For `LLM_PROVIDER=openai` the estimate uses the gpt-4o-mini rates in `config.py` (`OPENAI_INPUT_COST_PER_1M` and `OPENAI_OUTPUT_COST_PER_1M`). If a server does not report usage, token counts are estimated at about 4 characters per token and the UI says so.

### How to run

From the project folder, with the virtual environment active and Ollama running (`ollama pull llama3.2` for the default model):

```
python rag.py          # only needed to rebuild the ChromaDB index
streamlit run app.py   # cookbook chat, agent, workflow, and comparison
python compare.py      # same comparison, printed in the terminal
python tools.py        # ingredient, substitute, and nutrition checks with no LLM
```

In the Streamlit sidebar, choose **Recipe agent**, **Fixed workflow**, or **Comparison**. The cookbook chat is the original RAG page.

Optional safeguards can be set in `.env`:

```
AGENT_MAX_STEPS=8
AGENT_TIMEOUT_SECONDS=180
AGENT_MAX_TOTAL_TOKENS=12000
```
