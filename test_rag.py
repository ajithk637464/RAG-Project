"""
Phase 10: test question categories + chunk-size comparison.

Not a pytest suite - just a script that runs the 4 required question
categories against both chunk configs and prints what was retrieved, so the
results are easy to read and compare by eye:

  A - answer exists verbatim in the cookbook
  B - semantically related question, worded differently than the text
  C - question about a different recipe
  D - question with no answer anywhere in the cookbook (must be refused)
"""
import config
import rag

TEST_QUESTIONS = [
    ("A - verbatim answer", "What is Pasta Primavera tossed in?"),
    ("B - reworded / semantic", "Which vegetarian pasta dish has a creamy cheese sauce?"),
    ("C - different recipe", "What is in the Tuna and Pasta Bake?"),
    ("D - not in the cookbook", "How do I make a chocolate lava cake?"),
]


def run_for_config(config_name: str):
    """Temporarily point rag/config at a different chunk config, in-process only - config.py on disk is never edited."""
    cfg = config.CHUNK_CONFIGS[config_name]
    config.ACTIVE_CHUNK_CONFIG = config_name
    config.CHUNK_SIZE = cfg["chunk_size"]
    config.CHUNK_OVERLAP = cfg["chunk_overlap"]
    config.COLLECTION_NAME = f"cookbook_{config_name}"

    print(f"\n{'=' * 70}\nCONFIG: {config_name} (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})\n{'=' * 70}")
    stored = rag.build_index(reset=True)
    print(f"Indexed {stored} chunks into collection '{config.COLLECTION_NAME}'")

    for category, question in TEST_QUESTIONS:
        print(f"\n--- [{category}] \"{question}\" ---")
        chunks = rag.retrieve(question)
        for i, c in enumerate(chunks, start=1):
            print(f"  #{i} distance={c['distance']:.3f} | p.{c['page']} | recipe: {c['recipe_name'] or 'unknown'}")

        try:
            result = rag.answer_question(question)
            print(f"  ANSWER: {result['answer']}")
        except RuntimeError as e:
            print(f"  ANSWER: [skipped - {e}]")


if __name__ == "__main__":
    run_for_config("small")
    run_for_config("large")
    print("\nDone. Note: config.py on disk is untouched - it still defaults to 'small'.")
