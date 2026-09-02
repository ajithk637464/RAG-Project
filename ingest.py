"""
Phase 2: PDF ingestion.

Loads every PDF in data/, extracts text page-by-page, and attaches metadata
to each page: source filename, page number, and a best-effort recipe name.

We keep metadata attached from this very first step because later stages
(chunking, embedding, storage) all need to carry it forward so we can show
the user exactly where an answer came from (source citations).
"""
from pathlib import Path
from pypdf import PdfReader

import config

GENERIC_HEADERS = {
    "ingredients", "instructions", "directions", "method", "contents",
    "notes", "serves", "prep time", "cook time", "servings", "cook",
    "cooking time", "difficulty", "add+",
}
# Small connector words are allowed to stay lowercase inside a title
# (e.g. "Tuna and Pasta Bake") without failing the Title Case check.
LOWERCASE_OK = {"and", "with", "in", "of", "the", "a", "for", "on", "to", "or", "&"}


def _is_title_like(line: str) -> bool:
    words = line.split()
    significant = [w for w in words if w.lower() not in LOWERCASE_OK]
    if not significant:
        return False
    capitalized = sum(1 for w in significant if w[0].isupper())
    return capitalized / len(significant) >= 0.6


def detect_recipe_name(page_text: str):
    """
    Best-effort recipe name detection.

    Heuristic: a recipe title is usually 1-3 short lines near the top of the
    page, in Title Case, without a trailing period/colon, not a generic
    section header, and not an ingredient line (which tends to start with a
    quantity, e.g. "2 cups basmati rice"). Titles that wrap across multiple
    lines (a common PDF layout) are merged back into one string.
    """
    lines = [line.replace("\xa0", " ").strip() for line in page_text.splitlines()]
    lines = [line for line in lines if line]

    title_lines = []
    for line in lines[:6]:
        first_word = line.split()[0].lower() if line.split() else ""
        looks_like_title = (
            len(line) <= 60
            and not line.endswith((".", ":"))
            and line.lower() not in GENERIC_HEADERS
            and not line[0].isdigit()
            and _is_title_like(line)
            # Don't let a title start mid-phrase on a connector word
            # (e.g. skip "with Kale" if "Prawn pasta" above it was rejected).
            and not (not title_lines and first_word in LOWERCASE_OK)
        )
        if looks_like_title:
            title_lines.append(line)
            if len(title_lines) >= 3:
                break
        elif title_lines:
            # We already started collecting a title; this line breaks the
            # pattern (e.g. it's body text), so stop merging.
            break

    return " ".join(title_lines) if title_lines else None


def load_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from every page of one PDF, with metadata attached."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "text": text,
            "source": pdf_path.name,
            "page": i,
            "recipe_name": detect_recipe_name(text),
        })
    return pages


def load_all_pdfs(data_dir: Path = None) -> list[dict]:
    """Load every PDF found in the data/ folder."""
    data_dir = data_dir or config.DATA_DIR
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Add a cookbook PDF there and try again."
        )
    all_pages = []
    for pdf_path in pdf_files:
        all_pages.extend(load_pdf(pdf_path))
    return all_pages


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.

    We slide a window of `chunk_size` characters across the text, stepping
    forward by (chunk_size - chunk_overlap) each time. The overlap means the
    tail of one chunk reappears at the head of the next, so a fact that sits
    right on a chunk boundary still ends up whole in at least one chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_pages(pages: list[dict], chunk_size: int = None, chunk_overlap: int = None) -> list[dict]:
    """
    Chunk every page's text, carrying each page's metadata onto every chunk
    it produces. We chunk per-page (rather than across the whole document)
    so page-number metadata always stays accurate for citations.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    all_chunks = []
    for page in pages:
        pieces = chunk_text(page["text"], chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "id": f"{page['source']}_p{page['page']}_c{i}",
                "text": piece,
                "source": page["source"],
                "page": page["page"],
                "recipe_name": page["recipe_name"],
            })
    return all_chunks


if __name__ == "__main__":
    pages = load_all_pdfs()
    print(f"Loaded {len(pages)} pages from {config.DATA_DIR}")
    for p in pages[:3]:
        print(f"--- {p['source']} page {p['page']} (recipe: {p['recipe_name']}) ---")
        print(p["text"][:200])
        print()

    chunks = chunk_pages(pages)
    print(f"\nActive config: {config.ACTIVE_CHUNK_CONFIG} "
          f"(chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP})")
    print(f"Produced {len(chunks)} chunks from {len(pages)} pages")
    print(f"\nExample chunk: {chunks[5]}")
