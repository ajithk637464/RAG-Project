"""
One front door in front of the recipe-search and ingredient-db servers.

The agent speaks MCP to this process only. tools/list is the union of the
backends. Every tools/call writes one audit line: caller, tool, ingredient.
A scoped token denies lookup_nutrition here, before the backend runs.
This process does not call a language model.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_framing import serve
from mcp_host import McpHost

AUDIT_PATH = ROOT / "week9" / "gateway_audit.log"
BACKENDS = [
    {
        "name": "recipe-search",
        "command": "python",
        "args": ["servers/recipe_server.py"],
        "env": {},
    },
    {
        "name": "ingredient-db",
        "command": "python",
        "args": ["servers/ingredient_server.py"],
        "env": {"INGREDIENT_DB_TOKEN": os.environ.get("INGREDIENT_DB_TOKEN", "")},
    },
]

_host = None
_caller = "unknown"


def _audit(tool: str, arguments: dict, outcome: str) -> None:
    ingredient = arguments.get("ingredient") or arguments.get("query") or arguments.get("name") or ""
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"caller={_caller} tool={tool} ingredient={ingredient} outcome={outcome}\n"
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _host_open() -> McpHost:
    global _host
    if _host is None:
        _host = McpHost(servers=BACKENDS)
        _host.open()
        _host.discover()
    return _host


def _text(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle(message: dict):
    global _caller
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        client = ((message.get("params") or {}).get("clientInfo") or {})
        _caller = client.get("name") or "unknown"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "recipe-gateway", "version": "1.0.0"},
            },
        }
    host = _host_open()
    if method == "tools/list":
        tools = [
            {"name": tool["name"], "description": tool["description"], "inputSchema": tool["inputSchema"]}
            for tool in host.tools
        ]
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}}
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name") or ""
        arguments = params.get("arguments") or {}
        token = os.environ.get("INGREDIENT_DB_TOKEN", "")
        if name == "lookup_nutrition" and token == "ingredient-db-allergen-only":
            _audit(name, arguments, "denied")
            result = _text(
                "nutrition lookup is not allowed for this token; allergen lookup still works. "
                "Call lookup_allergen, and do not invent per-100g numbers.",
                is_error=True,
            )
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        text, _server, is_error = host.call(name, arguments)
        _audit(name, arguments, "denied" if is_error else "ok")
        return {"jsonrpc": "2.0", "id": request_id, "result": _text(text, is_error)}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


if __name__ == "__main__":
    try:
        serve(handle)
    finally:
        if _host is not None:
            _host.close()
