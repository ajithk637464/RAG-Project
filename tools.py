"""
Tools the recipe agent and the fixed workflow are allowed to call.

search_recipes is the existing RAG pipeline. The other three tools are small
and deterministic so each step is easy to explain:

- check_ingredients compares a pantry list with words in the retrieved recipe
- find_substitute looks up one ingredient in a fixed table
- calculate_nutrition adds a rough calorie table for ingredients it recognizes
"""
import inspect
import json
import re

import rag

# Longer names are matched first so "olive oil" wins over a shorter fragment
# and "tomatoes" is not also counted as "tomato".
COMMON_INGREDIENTS = [
    "olive oil", "cherry tomatoes", "chopped tomatoes", "mozzarella", "parmesan",
    "ricotta", "cheddar", "breadcrumbs", "pine nuts", "linguine", "penne",
    "tortellini", "lasagne", "conchiglie", "spaghetti", "pasta", "tuna",
    "chicken", "shrimp", "prawn", "mushrooms", "mushroom", "spinach", "kale",
    "lentils", "broccoli", "carrot", "peas", "onion", "garlic", "butter",
    "flour", "milk", "cream", "cheese", "tomato", "tomatoes", "olives",
    "basil", "oregano", "lemon", "egg", "rice", "mayonnaise", "yogurt",
]

SUBSTITUTES = {
    "butter": "olive oil, or margarine",
    "milk": "unsweetened plant milk",
    "cream": "milk mixed with a little butter, or plain yogurt",
    "cheese": "another hard cheese, or nutritional yeast",
    "cheddar": "another hard cheese such as parmesan",
    "parmesan": "another hard cheese, or nutritional yeast",
    "mozzarella": "another mild melting cheese",
    "ricotta": "cottage cheese or blended silken tofu",
    "egg": "a flax egg (1 tbsp ground flax plus 3 tbsp water) in baking",
    "tuna": "cooked chicken or chickpeas",
    "chicken": "chickpeas or extra mushrooms",
    "mayonnaise": "plain yogurt",
    "breadcrumbs": "crushed crackers",
    "olive oil": "any neutral cooking oil",
    "onion": "a shallot, or a pinch of onion powder",
    "garlic": "a pinch of garlic powder",
    "lemon": "a small splash of vinegar",
    "basil": "parsley, or dried Italian herbs",
    "cream cheese": "plain yogurt",
    "tomato": "fresh tomatoes, canned tomatoes, or passata",
    "tomatoes": "fresh tomatoes, canned tomatoes, or passata",
    "chopped tomatoes": "fresh tomatoes, canned tomatoes, or passata",
    "olives": "capers, or leave them out",
    "flour": "cornstarch, using about half as much",
}

# Rough kilocalories for a typical home quantity. This is a teaching estimate,
# not a dietary measurement.
NUTRITION_KCAL = {
    "pasta": 220, "linguine": 220, "penne": 220, "spaghetti": 220,
    "conchiglie": 220, "tortellini": 250, "lasagne": 200,
    "tuna": 180, "chicken": 180, "shrimp": 100, "prawn": 100,
    "cheese": 110, "cheddar": 110, "parmesan": 80, "mozzarella": 120, "ricotta": 150,
    "butter": 100, "olive oil": 120, "flour": 50, "milk": 60, "cream": 100,
    "tomato": 25, "tomatoes": 30, "cherry tomatoes": 30, "chopped tomatoes": 40,
    "onion": 40, "garlic": 5, "spinach": 20, "kale": 20, "mushroom": 15,
    "mushrooms": 20, "broccoli": 30, "lentils": 160, "olives": 40,
    "breadcrumbs": 80, "pine nuts": 90, "egg": 70, "rice": 200, "lemon": 10,
}


def extract_pantry(question: str) -> str:
    """Pull a pantry list out of phrases like 'I have tuna, pasta, and cheese'."""
    patterns = [
        r"i have ([^.?\n]+)",
        r"i've got ([^.?\n]+)",
        r"i’ve got ([^.?\n]+)",
        r"available ingredients?(?: are| include)?[:\s]+([^.?\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _split_items(text: str) -> list[str]:
    parts = re.split(r",|;|\band\b", (text or "").lower())
    return [part.strip(" .") for part in parts if part.strip(" .")]


def ingredients_in_text(text: str) -> list[str]:
    """Ingredient names from COMMON_INGREDIENTS that appear as whole words."""
    lower = (text or "").lower()
    found = []
    for name in sorted(COMMON_INGREDIENTS, key=len, reverse=True):
        if any(name in earlier or earlier in name for earlier in found):
            continue
        if re.search(rf"\b{re.escape(name)}\b", lower):
            found.append(name)
    # A more specific name already covers the generic word, so don't count both.
    if "pasta" in found and any(item in PASTA_WORDS and item != "pasta" for item in found):
        found = [item for item in found if item != "pasta"]
    if "cheese" in found and any(item in CHEESE_WORDS and item != "cheese" for item in found):
        found = [item for item in found if item != "cheese"]
    if "tomato" in found and any(item in TOMATO_WORDS and item not in ("tomato",) for item in found):
        found = [item for item in found if item != "tomato"]
    return found


PASTA_WORDS = {
    "pasta", "linguine", "penne", "tortellini", "lasagne", "conchiglie", "spaghetti",
}
CHEESE_WORDS = {"cheese", "cheddar", "parmesan", "mozzarella", "ricotta"}
TOMATO_WORDS = {"tomato", "tomatoes", "cherry tomatoes", "chopped tomatoes"}


def _same_family(recipe_item: str, pantry_item: str) -> bool:
    """'pasta' covers shapes, 'cheese' covers named cheeses, 'tomato' covers tomato forms."""
    for family in (PASTA_WORDS, CHEESE_WORDS, TOMATO_WORDS):
        if recipe_item in family and pantry_item in family:
            return True
    return False


def _covers(recipe_item: str, pantry_item: str) -> bool:
    return (
        recipe_item in pantry_item
        or pantry_item in recipe_item
        or _same_family(recipe_item, pantry_item)
    )


GENERIC_PAGE_NAMES = {
    "recipes", "pasta dishes", "pasta recipe guide", "contents",
    "soup stock cubes available",
}


def _mentioned_in_query(recipe_name: str, query: str) -> bool:
    """True when the user named this recipe, not just a generic index page."""
    name = (recipe_name or "").strip().lower()
    if len(name) < 4 or name in GENERIC_PAGE_NAMES:
        return False
    if name in query:
        return True
    words = [word for word in re.findall(r"[a-z0-9]+", name) if len(word) > 3 and word not in {"pasta", "with"}]
    if not words:
        return False
    hits = sum(1 for word in words if word in query)
    # Allow one missing word so "tuna pasta" still matches "Tuna and Pasta Bake".
    return hits >= max(1, len(words) - 1)


def focus_chunks(query: str, chunks: list[dict]) -> tuple[list[dict], bool]:
    """
    Keep chunks for the recipe the user actually named.

    Retrieval also returns the contents page, and that page lists every dish.
    Later tools would treat those titles as ingredients. When the query names
    a retrieved recipe, only that recipe's chunks are kept. The bool is False
    when no retrieved recipe name matches the question.
    """
    query_l = (query or "").lower()
    mentioned = [chunk for chunk in chunks if _mentioned_in_query(chunk.get("recipe_name"), query_l)]
    if not mentioned:
        return chunks, False
    chosen = mentioned[0]["recipe_name"]
    focused = [chunk for chunk in chunks if chunk.get("recipe_name") == chosen] or mentioned
    return focused, True


def search_recipes(query: str) -> dict:
    """Existing RAG: retrieve cookbook chunks, then generate a grounded answer."""
    query = (query or "").strip()
    if not query:
        return {"error": "search_recipes needs a query."}

    chunks, matched = focus_chunks(query, rag.retrieve(query))
    if not matched:
        # No retrieved recipe name matches the question. Skip the model so a
        # nearby but different dish is not presented as the answer.
        return {
            "answer": "I couldn't find that in the cookbook.",
            "recipes": [],
            "sources": [],
            "recipe_text": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "usage_estimated": False,
        }

    # Ask only for the recipe text. Substitutes and calories belong to other tools.
    lookup_question = (
        "List the recipe name and the ingredients written in the context. "
        "Do not discuss substitutes or nutrition."
    )
    generated = rag.generate_answer_with_usage(lookup_question, chunks)
    recipes = []
    sources = []
    for chunk in chunks:
        name = chunk.get("recipe_name") or "unknown"
        if name not in recipes:
            recipes.append(name)
        sources.append(f"{chunk['source']} p.{chunk['page']} ({name})")

    return {
        "answer": generated["answer"],
        "recipes": recipes,
        "sources": sources,
        "recipe_text": "\n\n".join(chunk["text"] for chunk in chunks),
        "prompt_tokens": generated["prompt_tokens"],
        "completion_tokens": generated["completion_tokens"],
        "usage_estimated": generated["usage_estimated"],
    }


def check_ingredients(recipe_text: str = "", available_ingredients: str = "") -> dict:
    """Which known ingredients in the recipe text are in the user's pantry."""
    recipe_text = recipe_text or ""
    if not recipe_text.strip():
        return {
            "present": [],
            "missing": [],
            "mentioned_but_not_in_recipe": _split_items(available_ingredients),
            "note": "No recipe text yet. Search the cookbook first.",
        }

    recipe_items = ingredients_in_text(recipe_text)
    available = _split_items(available_ingredients)
    present = [item for item in recipe_items if any(_covers(item, have) for have in available)]
    missing = [item for item in recipe_items if item not in present]
    unused = [have for have in available if not any(_covers(item, have) for item in recipe_items)]
    return {
        "present": present,
        "missing": missing,
        "mentioned_but_not_in_recipe": unused,
        "recipe_ingredients_found": recipe_items,
    }


def find_substitute(ingredient: str = "") -> dict:
    """Look up one ingredient in the built-in substitute table."""
    key = (ingredient or "").strip().lower()
    if not key:
        return {"ingredient": "", "substitute": None, "note": "No ingredient was provided."}
    if key in SUBSTITUTES:
        return {"ingredient": key, "substitute": SUBSTITUTES[key], "source": "built-in table"}
    for name, substitute in SUBSTITUTES.items():
        if name in key or key in name:
            return {"ingredient": name, "substitute": substitute, "source": "built-in table"}
    return {
        "ingredient": key,
        "substitute": None,
        "source": "built-in table",
        "note": "No substitute is listed for that ingredient.",
    }


def calculate_nutrition(recipe_text: str = "") -> dict:
    """Sum the rough calorie table for ingredients recognized in the recipe text."""
    items = ingredients_in_text(recipe_text or "")
    breakdown = []
    for item in items:
        kcal = NUTRITION_KCAL.get(item)
        if kcal is None:
            continue
        breakdown.append({"ingredient": item, "kcal": kcal})
    return {
        "estimated_kcal": sum(item["kcal"] for item in breakdown),
        "breakdown": breakdown,
        "note": "Rough teaching estimate for typical home amounts, not a dietary measurement.",
    }


TOOLS = {
    "search_recipes": search_recipes,
    "check_ingredients": check_ingredients,
    "find_substitute": find_substitute,
    "calculate_nutrition": calculate_nutrition,
}

ALIASES = {
    "search": "search_recipes",
    "search_recipe": "search_recipes",
    "rag_search": "search_recipes",
    "ingredients": "check_ingredients",
    "check_ingredient": "check_ingredients",
    "substitute": "find_substitute",
    "nutrition": "calculate_nutrition",
}


def _coerce_arguments(fn, arguments: dict) -> dict:
    """Keep only parameters the tool accepts, and accept a few obvious aliases."""
    allowed = set(inspect.signature(fn).parameters)
    clean = {}
    for key, value in (arguments or {}).items():
        if key in allowed and value is not None:
            clean[key] = str(value)

    aliases = {
        "query": ("question", "q"),
        "ingredient": ("item", "missing"),
        "available_ingredients": ("available", "pantry", "ingredients"),
        "recipe_text": ("recipe", "text", "context"),
    }
    for target, names in aliases.items():
        if target in clean or target not in allowed:
            continue
        for name in names:
            if arguments.get(name):
                clean[target] = str(arguments[name])
                break
    return clean


def run_tool(name: str, arguments: dict = None) -> tuple[dict, dict]:
    """
    Run one tool by name.

    Returns (result, usage). Usage is zero for the deterministic tools.
    Errors are returned as {"error": "..."} so a bad call does not crash the loop.
    """
    arguments = arguments or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    tool_name = ALIASES.get(name, name)
    fn = TOOLS.get(tool_name)
    zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "usage_estimated": False}
    if fn is None:
        known = ", ".join(TOOLS)
        return {"error": f"Unknown tool '{name}'. Use one of: {known}."}, zero_usage

    try:
        raw = fn(**_coerce_arguments(fn, arguments))
    except Exception as exc:
        return {"error": f"{tool_name} failed: {exc}"}, zero_usage

    usage = {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "usage_estimated": bool(raw.get("usage_estimated")),
    }
    public = {
        key: value
        for key, value in raw.items()
        if key not in ("prompt_tokens", "completion_tokens", "usage_estimated")
    }
    return public, usage


def to_text(payload) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    sample = (
        "Tuna and Pasta Bake. Ingredients: pasta, butter, onion, garlic, "
        "flour, tomatoes, olives, tuna, cheddar cheese."
    )
    print(check_ingredients(sample, "tuna, pasta, and cheese"))
    print(find_substitute("butter"))
    print(find_substitute("saffron"))
    print(calculate_nutrition(sample))
