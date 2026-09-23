"""
Central place for every tunable setting in this project.

Why centralize this? During the chunking experiment (Phase 9/10) we need to
change chunk_size/overlap and re-run ingestion. Keeping every knob here means
we only ever edit one file, and app.py/ingest.py/rag.py stay untouched.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = str(BASE_DIR / "chroma_db")

# --- Chunking experiments ---
# Two named configs so we can compare retrieval quality across chunk sizes.
# Switch ACTIVE_CHUNK_CONFIG below and re-run ingest.py to build the other index.
CHUNK_CONFIGS = {
    "small": {"chunk_size": 500, "chunk_overlap": 50},
    "large": {"chunk_size": 1000, "chunk_overlap": 100},
}
ACTIVE_CHUNK_CONFIG = "small"

CHUNK_SIZE = CHUNK_CONFIGS[ACTIVE_CHUNK_CONFIG]["chunk_size"]
CHUNK_OVERLAP = CHUNK_CONFIGS[ACTIVE_CHUNK_CONFIG]["chunk_overlap"]

# Each chunk config gets its own ChromaDB collection, so switching configs
# never mixes chunks from different experiments in the same search.
COLLECTION_NAME = f"cookbook_{ACTIVE_CHUNK_CONFIG}"

# --- Embeddings ---
# Local model (downloaded once, then runs on your machine — no API cost/latency).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 3

# --- LLM (answer generation) ---
# "ollama" = free, runs locally, no API key needed (default).
# "openai" = paid, needs a real OPENAI_API_KEY with billing set up.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

LLM_MODEL = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else "gpt-4o-mini"

# --- Week 7 agent safeguards ---
# The manual loop stops when any one of these is hit, so a confused model
# cannot keep calling tools or run up a bill.
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
AGENT_TIMEOUT_SECONDS = float(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))
AGENT_MAX_TOTAL_TOKENS = int(os.getenv("AGENT_MAX_TOTAL_TOKENS", "12000"))

# Public list prices for gpt-4o-mini. Used only when LLM_PROVIDER=openai.
# A local Ollama model is reported as $0.
OPENAI_INPUT_COST_PER_1M = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0.15"))
OPENAI_OUTPUT_COST_PER_1M = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0.60"))
