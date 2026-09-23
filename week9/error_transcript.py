"""Same failing find_substitute call, legacy Error 3 vs the recoverable docstring."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mcp_agent

QUESTION = "What can I use instead of creme fraiche lite?"


def spec(legacy: bool) -> list[dict]:
    env = {"SUBSTITUTE_ERROR_MODE": "legacy"} if legacy else {}
    return [{
        "name": "recipe-search",
        "command": "python",
        "args": ["servers/recipe_server.py"],
        "env": env,
    }]


def main() -> None:
    before = mcp_agent.run(QUESTION, servers=spec(legacy=True))
    after = mcp_agent.run(QUESTION, servers=spec(legacy=False))
    out = Path(__file__).resolve().parent / "error_runs.json"
    out.write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    print("BEFORE tools", before["tools"])
    for step in before["steps"]:
        print(" ", step)
    print("BEFORE answer", before["answer"])
    print("AFTER tools", after["tools"])
    for step in after["steps"]:
        print(" ", step)
    print("AFTER answer", after["answer"])


if __name__ == "__main__":
    main()
