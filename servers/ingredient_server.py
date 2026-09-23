"""
Stand-in for the content team's ingredient database MCP server.

Lookup tools return allergen flags or per-100g nutrition for one name.
The full allergen matrix is a resource, not a tool: the host attaches it.
This process does not call a language model.
"""
import json
import os
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_framing import serve

LOG_PATH = ROOT / "week9" / "ingredient_db.log"
FULL_TOKEN = "ingredient-db-full"
SCOPED_TOKEN = "ingredient-db-allergen-only"

# Per 100g, rough reference figures for the assignment, not dietary advice.
DATABASE = {
    "butter": {"allergens": ["milk"], "per_100g": {"kcal": 717, "fat_g": 81, "protein_g": 0.9}},
    "cream": {"allergens": ["milk"], "per_100g": {"kcal": 340, "fat_g": 36, "protein_g": 2.1}},
    "creme fraiche": {"allergens": ["milk"], "per_100g": {"kcal": 292, "fat_g": 30, "protein_g": 2.4}},
    "milk": {"allergens": ["milk"], "per_100g": {"kcal": 61, "fat_g": 3.3, "protein_g": 3.2}},
    "cheddar": {"allergens": ["milk"], "per_100g": {"kcal": 403, "fat_g": 33, "protein_g": 25}},
    "parmesan": {"allergens": ["milk"], "per_100g": {"kcal": 431, "fat_g": 29, "protein_g": 38}},
    "tuna": {"allergens": ["fish"], "per_100g": {"kcal": 132, "fat_g": 1.3, "protein_g": 28}},
    "pasta": {"allergens": ["gluten"], "per_100g": {"kcal": 371, "fat_g": 1.5, "protein_g": 13}},
    "flour": {"allergens": ["gluten"], "per_100g": {"kcal": 364, "fat_g": 1.0, "protein_g": 10}},
    "egg": {"allergens": ["egg"], "per_100g": {"kcal": 143, "fat_g": 10, "protein_g": 13}},
    "tomato": {"allergens": [], "per_100g": {"kcal": 18, "fat_g": 0.2, "protein_g": 0.9}},
    "olive oil": {"allergens": [], "per_100g": {"kcal": 884, "fat_g": 100, "protein_g": 0}},
}


def _token() -> str:
    return os.environ.get("INGREDIENT_DB_TOKEN", "")


def _scope() -> str:
    token = _token()
    if token == FULL_TOKEN:
        return "full"
    if token == SCOPED_TOKEN:
        return "allergen-only"
    return "invalid"


def _log(tool: str, ingredient: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"tool={tool} ingredient={ingredient} token_scope={_scope()}\n")


def _normalize(name: str) -> str:
    return " ".join((name or "").lower().replace("è", "e").replace("é", "e").split())


def _suggest(name: str) -> str:
    lowered = _normalize(name)
    for suffix in (" lite", " light", " low-fat", " low fat"):
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)].strip()
    return ""


def _text(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _lookup_row(name: str, tool: str):
    _log(tool, name)
    if _scope() == "invalid":
        return None, _text(
            "ingredient database token was rejected. Do not invent allergen flags or nutrition.",
            is_error=True,
        )
    key = _normalize(name)
    if not key:
        return None, _text(f"{tool} needs an ingredient name.", is_error=True)
    row = DATABASE.get(key)
    if row is None:
        suggestion = _suggest(key)
        if suggestion and suggestion in DATABASE:
            return None, _text(
                f"no ingredient matched {key!r}: try {suggestion!r}",
                is_error=True,
            )
        known = ", ".join(sorted(DATABASE))
        return None, _text(
            f"no ingredient matched {key!r}: try one of: {known}",
            is_error=True,
        )
    return row, None


def lookup_allergen(arguments: dict) -> dict:
    row, error = _lookup_row(arguments.get("ingredient") or "", "lookup_allergen")
    if error:
        return error
    return _text(json.dumps({
        "ingredient": _normalize(arguments.get("ingredient") or ""),
        "allergens": row["allergens"],
    }))


def lookup_nutrition(arguments: dict) -> dict:
    if _scope() == "allergen-only":
        ingredient = arguments.get("ingredient") or ""
        _log("lookup_nutrition", ingredient)
        return _text(
            "nutrition lookup is not allowed for this token; allergen lookup still works. "
            "Call lookup_allergen, and do not invent per-100g numbers.",
            is_error=True,
        )
    row, error = _lookup_row(arguments.get("ingredient") or "", "lookup_nutrition")
    if error:
        return error
    return _text(json.dumps({
        "ingredient": _normalize(arguments.get("ingredient") or ""),
        "per_100g": row["per_100g"],
        "note": "Per 100g reference figures from the ingredient table, not a medical measurement.",
    }))


TOOLS = [
    {
        "name": "lookup_allergen",
        "description": (
            "Look up allergen flags for one ingredient name. "
            "Input: {\"ingredient\": \"butter\"}. "
            "If the text says try another spelling, call lookup_allergen again with that name. "
            "Do not invent allergen flags."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"ingredient": {"type": "string"}},
            "required": ["ingredient"],
        },
    },
    {
        "name": "lookup_nutrition",
        "description": (
            "Look up per-100g nutrition for one ingredient name. "
            "Input: {\"ingredient\": \"butter\"}. "
            "If the token cannot read nutrition, say so and do not invent numbers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"ingredient": {"type": "string"}},
            "required": ["ingredient"],
        },
    },
]

RESOURCE = {
    "uri": "ingredient://allergen-matrix",
    "name": "allergen-matrix",
    "description": (
        "Names the database knows. Flags and per-100g numbers are not in this resource. "
        "The host attaches this once. The model must call lookup_allergen or lookup_nutrition "
        "for a specific ingredient."
    ),
    "mimeType": "text/plain",
}


def _matrix_text() -> str:
    names = ", ".join(sorted(DATABASE))
    return (
        "Ingredient database resource (not a tool). "
        f"Known names: {names}. "
        "Allergen flags and per-100g nutrition are not listed here. "
        "Call lookup_allergen or lookup_nutrition with one name."
    )


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
                "serverInfo": {"name": "ingredient-db", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": [RESOURCE]}}
    if method == "resources/read":
        uri = ((message.get("params") or {}).get("uri"))
        if uri != RESOURCE["uri"]:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32002, "message": f"Resource not found: {uri}"},
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"contents": [{"uri": uri, "mimeType": "text/plain", "text": _matrix_text()}]},
        }
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "lookup_allergen":
            result = lookup_allergen(arguments)
        elif name == "lookup_nutrition":
            result = lookup_nutrition(arguments)
        else:
            result = _text(f"Unknown tool {name!r}.", is_error=True)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


if __name__ == "__main__":
    serve(handle)
