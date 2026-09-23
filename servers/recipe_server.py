"""
Our recipe-search MCP server.

It exposes cookbook tools. It does not call a language model. The host does.
find_substitute's description is the prompt the model sees, and a miss names
the next ingredient to try instead of returning "Error 3".
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_framing import serve

LEGACY = os.environ.get("SUBSTITUTE_ERROR_MODE") == "legacy"

FIND_SUBSTITUTE_DESCRIPTION = (
    "Look up one ingredient in the local substitute table. "
    "Input: {\"ingredient\": \"the ingredient name\"}. "
    "If nothing matches, the text names a simpler ingredient after the word 'try'. "
    "Call find_substitute again with that exact name. "
    "Do not invent a substitute, and do not treat the miss as a dead server."
)
LEGACY_FIND_SUBSTITUTE_DESCRIPTION = "Find a substitute. Returns Error 3 on failure."

EXTRA_SUBSTITUTES = {
    "creme fraiche": "sour cream, or equal parts heavy cream and plain yogurt",
    "crème fraîche": "sour cream, or equal parts heavy cream and plain yogurt",
}


def _strip_qualifier(name: str) -> str:
    """'creme fraiche lite' -> 'creme fraiche'. The retry target, not a guess."""
    lowered = " ".join(name.lower().split())
    for suffix in (" lite", " light", " low-fat", " low fat"):
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)].strip()
    return lowered


def _text_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _find_substitute(ingredient: str) -> dict:
    import tools

    key = " ".join((ingredient or "").lower().split())
    if not key:
        return _text_result("find_substitute needs an ingredient name.", is_error=True)
    if LEGACY:
        hit = tools.find_substitute(key)
        if hit.get("substitute"):
            return _text_result(tools.to_text(hit))
        return _text_result("Error 3", is_error=True)
    if key in EXTRA_SUBSTITUTES:
        return _text_result(tools.to_text({
            "ingredient": key,
            "substitute": EXTRA_SUBSTITUTES[key],
            "source": "recipe-search substitute table",
        }))
    hit = tools.find_substitute(key)
    if hit.get("substitute"):
        return _text_result(tools.to_text(hit))
    suggestion = _strip_qualifier(key)
    if suggestion != key:
        return _text_result(
            f"no substitute matched {key!r}: try {suggestion!r}",
            is_error=True,
        )
    return _text_result(
        f"no substitute matched {key!r}: try a single common ingredient name",
        is_error=True,
    )


def _search_recipes(query: str) -> dict:
    """Retrieval only. The host model writes the answer."""
    import rag
    import tools

    query = (query or "").strip()
    if not query:
        return _text_result("search_recipes needs a query.", is_error=True)
    chunks, matched = tools.focus_chunks(query, rag.retrieve(query))
    if not matched:
        return _text_result("I couldn't find that in the cookbook.")
    recipes = []
    lines = []
    for chunk in chunks:
        name = chunk.get("recipe_name") or "unknown"
        if name not in recipes:
            recipes.append(name)
        lines.append(f"{chunk['source']} p.{chunk['page']} ({name})\n{chunk['text']}")
    return _text_result(tools.to_text({"recipes": recipes, "excerpts": "\n\n".join(lines)[:2000]}))


def _check_ingredients(arguments: dict) -> dict:
    import tools

    result = tools.check_ingredients(
        arguments.get("recipe_text") or "",
        arguments.get("available_ingredients") or "",
    )
    return _text_result(tools.to_text(result))


def _calculate_nutrition(arguments: dict) -> dict:
    import tools

    result = tools.calculate_nutrition(arguments.get("recipe_text") or "")
    return _text_result(tools.to_text(result))


def _tool_list() -> list[dict]:
    substitute_description = (
        LEGACY_FIND_SUBSTITUTE_DESCRIPTION if LEGACY else FIND_SUBSTITUTE_DESCRIPTION
    )
    return [
        {
            "name": "search_recipes",
            "description": (
                "Search the cookbook and return matching recipe excerpts. "
                "Input: {\"query\": \"dish or ingredient words\"}. "
                "This does not write the final answer."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "check_ingredients",
            "description": (
                "Compare a pantry list with recipe text. "
                "Input: {\"recipe_text\": \"excerpt\", \"available_ingredients\": \"what the user has\"}."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "recipe_text": {"type": "string"},
                    "available_ingredients": {"type": "string"},
                },
            },
        },
        {
            "name": "find_substitute",
            "description": substitute_description,
            "inputSchema": {
                "type": "object",
                "properties": {"ingredient": {"type": "string"}},
                "required": ["ingredient"],
            },
        },
        {
            "name": "calculate_nutrition",
            "description": (
                "Sum the local calorie table for ingredients in recipe text. "
                "Input: {\"recipe_text\": \"excerpt\"}. Use this number, do not invent one."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"recipe_text": {"type": "string"}},
            },
        },
    ]


HANDLERS = {
    "search_recipes": lambda arguments: _search_recipes(arguments.get("query") or ""),
    "check_ingredients": _check_ingredients,
    "find_substitute": lambda arguments: _find_substitute(arguments.get("ingredient") or ""),
    "calculate_nutrition": _calculate_nutrition,
}


def handle(message: dict):
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}, "resources": {}},
                "serverInfo": {"name": "recipe-search", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tool_list()}}
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        handler = HANDLERS.get(name)
        if handler is None:
            result = _text_result(f"Unknown tool {name!r}.", is_error=True)
        else:
            result = handler(params.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


if __name__ == "__main__":
    serve(handle)
