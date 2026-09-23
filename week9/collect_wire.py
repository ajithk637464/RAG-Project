"""Capture tools/list counts and the raw ingredient-db JSON-RPC exchange."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_host import McpHost, McpSession

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def names(config_path: Path) -> list[str]:
    host = McpHost(config_path=config_path)
    host.open()
    try:
        return [f"{tool['name']} [{tool['server']}]" for tool in host.discover()]
    finally:
        host.close()


def main() -> None:
    before = names(OUT / "mcp_config.server1.json")
    after = names(ROOT / "mcp_config.json")
    (OUT / "tool_counts.txt").write_text(
        f"{len(before)} before -> {len(after)} after\n"
        f"before: {', '.join(before)}\n"
        f"after: {', '.join(after)}\n",
        encoding="utf-8",
    )
    print((OUT / "tool_counts.txt").read_text(encoding="utf-8"))

    session = McpSession({
        "name": "ingredient-db",
        "command": "python",
        "args": ["servers/ingredient_server.py"],
        "env": {"INGREDIENT_DB_TOKEN": "ingredient-db-full"},
    })
    try:
        session.initialize()
        session.list_tools()
        session.call_tool("lookup_allergen", {"ingredient": "butter"})
    finally:
        session.close()
    (OUT / "wire_raw.json").write_text(
        json.dumps(session.trace, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(session.trace)} wire messages")


if __name__ == "__main__":
    main()
