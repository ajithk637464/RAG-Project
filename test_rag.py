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
    ("A - verbatim answer", "What is Pasta Primavera tossed in?", "Pasta Primavera"),
    ("B - reworded / semantic", "Which vegetarian pasta dish has a creamy cheese sauce?", "Pasta Primavera"),
    ("C - different recipe", "What is in the Tuna and Pasta Bake?", "Tuna and Pasta Bake"),
    ("D - not in the cookbook", "How do I make a chocolate lava cake?", None),
]


def is_relevant(chunk: dict, expected_recipe: str) -> bool:
    """Treat a result as relevant when its detected recipe matches the target."""
    return bool(
        expected_recipe
        and chunk.get("recipe_name")
        and expected_recipe.lower() in chunk["recipe_name"].lower()
    )


def score_results(result_sets: list[list[dict]], expected_recipes: list[str], top_k: int) -> tuple[float, float]:
    """Calculate Hit Rate@K and mean reciprocal rank for one retrieval method."""
    hits = 0
    reciprocal_ranks = []
    for chunks, expected_recipe in zip(result_sets, expected_recipes):
        first_hit = next(
            (rank for rank, chunk in enumerate(chunks[:top_k], start=1)
             if is_relevant(chunk, expected_recipe)),
            None,
        )
        if first_hit:
            hits += 1
            reciprocal_ranks.append(1 / first_hit)
        else:
            reciprocal_ranks.append(0.0)
    count = len(expected_recipes)
    return hits / count, sum(reciprocal_ranks) / count


def evaluate_retrieval(top_k: int = None) -> None:
    """Compare the existing semantic search with the new hybrid search."""
    top_k = top_k or config.TOP_K
    evaluated = [case for case in TEST_QUESTIONS if case[2]]
    expected_recipes = [case[2] for case in evaluated]
    semantic_results = [rag.retrieve(case[1], top_k, hybrid=False) for case in evaluated]
    hybrid_results = [rag.retrieve(case[1], top_k, hybrid=True) for case in evaluated]
    semantic_scores = score_results(semantic_results, expected_recipes, top_k)
    hybrid_scores = score_results(hybrid_results, expected_recipes, top_k)

    print(f"\nEvaluation on {len(evaluated)} known-answer questions (K={top_k})")
    print(f"  Semantic: Hit Rate@{top_k}={semantic_scores[0]:.3f}, MRR={semantic_scores[1]:.3f}")
    print(f"  Hybrid:   Hit Rate@{top_k}={hybrid_scores[0]:.3f}, MRR={hybrid_scores[1]:.3f}")
    print(f"  Change:   Hit Rate@{top_k}={hybrid_scores[0] - semantic_scores[0]:+.3f}, MRR={hybrid_scores[1] - semantic_scores[1]:+.3f}")


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

    evaluate_retrieval()

    for category, question, _ in TEST_QUESTIONS:
        print(f"\n--- [{category}] \"{question}\" ---")
        chunks = rag.retrieve(question)
        for i, c in enumerate(chunks, start=1):
            distance = f"{c['distance']:.3f}" if c.get("distance") is not None else "n/a"
            print(f"  #{i} distance={distance} | p.{c['page']} | recipe: {c['recipe_name'] or 'unknown'}")

        try:
            result = rag.answer_question(question)
            print(f"  ANSWER: {result['answer']}")
        except RuntimeError as e:
            print(f"  ANSWER: [skipped - {e}]")


if __name__ == "__main__":
    run_for_config("small")
    run_for_config("large")
    print("\nDone. Note: config.py on disk is untouched - it still defaults to 'small'.")
