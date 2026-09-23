"""Print the raw model reply after the recoverable miss, without changing the agent."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag
from mcp_agent import system_prompt
from mcp_host import load_host

QUESTION = "What can I use instead of creme fraiche lite?"


def once(legacy: bool) -> None:
    env = {"SUBSTITUTE_ERROR_MODE": "legacy"} if legacy else {}
    host = load_host(servers=[{
        "name": "recipe-search",
        "command": "python",
        "args": ["servers/recipe_server.py"],
        "env": env,
    }])
    try:
        messages = [
            {"role": "system", "content": system_prompt(host.tools, "")},
            {"role": "user", "content": QUESTION},
        ]
        first = rag.complete_chat(messages, max_tokens=300)
        print("MODE", "legacy" if legacy else "new")
        print("RAW1", first["content"])
        text, _server, _err = host.call("find_substitute", {"ingredient": "creme fraiche lite"})
        messages.append({"role": "user", "content": f"Tool find_substitute returned:\n{text}"})
        second = rag.complete_chat(messages, max_tokens=300)
        print("OBS", text)
        print("RAW2", second["content"])
        if not legacy and "creme fraiche" in second["content"]:
            text2, _s, _e = host.call("find_substitute", {"ingredient": "creme fraiche"})
            messages.append({"role": "user", "content": f"Tool find_substitute returned:\n{text2}"})
            third = rag.complete_chat(messages, max_tokens=300)
            print("OBS2", text2)
            print("RAW3", third["content"])
    finally:
        host.close()


if __name__ == "__main__":
    once(True)
    once(False)
