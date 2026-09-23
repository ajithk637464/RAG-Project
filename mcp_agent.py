"""
Week 9 recipe agent.

The tool list is the result of tools/list. This file does not name the
ingredient-database tools. Adding that server is a config change.
"""
import json

import rag
from mcp_host import load_host

MAX_STEPS = 4


def system_prompt(tools: list[dict], attached: str) -> str:
    lines = [
        "You are a recipe assistant. Use only the tools listed here.",
        "They were discovered from MCP servers. Do not invent tools or numbers.",
        "Reply with one JSON object.",
        'Call a tool: {"action":"tool","tool":"TOOL_NAME","arguments":{...}}',
        'Finish: {"action":"final","final_answer":"..."}',
        "If a tool result says to try a different name, call that tool again with that name.",
        "If a tool says a lookup is not allowed, tell the user and do not invent the missing data.",
        "",
        "Tools:",
    ]
    for tool in tools:
        lines.append(f"- {tool['name']} [server {tool['server']}]: {tool['description']}")
    if attached:
        lines.extend(["", "Attached resource (already loaded, not a tool call):", attached])
    return "\n".join(lines)


def _parse(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start:end + 1])


def _tool_name(decision: dict, known: set[str]) -> str:
    """The name is one the server advertised. action may be that name or the word tool."""
    named = str(decision.get("tool") or "")
    if named:
        return named
    action = str(decision.get("action") or "")
    if action in known:
        return action
    return ""


def _arguments(decision: dict) -> dict:
    arguments = decision.get("arguments")
    if isinstance(arguments, dict) and arguments:
        return arguments
    skip = {"action", "tool", "thought", "final_answer", "final"}
    return {key: value for key, value in decision.items() if key not in skip}


def run(question: str, config_path=None, servers=None) -> dict:
    """Discover tools, then Think -> tools/call -> Observe. The model runs here, not in a server."""
    host = load_host(config_path=config_path, servers=servers)
    steps = []
    try:
        tools = host.tools
        known = {tool["name"] for tool in tools}
        attached = host.resources_text()
        messages = [
            {"role": "system", "content": system_prompt(tools, attached)},
            {"role": "user", "content": question},
        ]
        answer = ""
        for step_number in range(1, MAX_STEPS + 1):
            completion = rag.complete_chat(messages, max_tokens=300)
            try:
                decision = _parse(completion["content"])
            except (ValueError, json.JSONDecodeError):
                messages.append({"role": "user", "content": "Reply with one JSON object."})
                steps.append({
                    "step": step_number,
                    "thought": "The reply was not JSON.",
                    "observation": completion["content"][:500],
                })
                continue
            name = _tool_name(decision, known)
            if decision.get("action") == "final" or (decision.get("final_answer") and not name):
                answer = str(decision.get("final_answer") or "").strip()
                steps.append({
                    "step": step_number,
                    "thought": decision.get("thought") or "",
                    "final": True,
                    "observation": answer,
                })
                break
            arguments = _arguments(decision)
            text, server, is_error = host.call(name, arguments)
            steps.append({
                "step": step_number,
                "thought": decision.get("thought") or "",
                "tool": name,
                "server": server,
                "arguments": arguments,
                "is_error": is_error,
                "observation": text,
            })
            messages.append({
                "role": "user",
                "content": f"Tool {name} returned:\n{text}",
            })
        else:
            answer = "Stopped before a final answer."
        return {
            "question": question,
            "answer": answer,
            "steps": steps,
            "tools": [{"name": tool["name"], "server": tool["server"]} for tool in tools],
        }
    finally:
        host.close()


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip() or "What allergens does butter have?"
    result = run(question)
    print("Discovered:", ", ".join(f"{tool['name']}@{tool['server']}" for tool in result["tools"]))
    for step in result["steps"]:
        if step.get("tool"):
            print(f"step {step['step']} tool={step['tool']} server={step['server']} args={step['arguments']}")
            print(f"  observe: {step['observation'][:300]}")
        else:
            print(f"step {step['step']} {step.get('observation', '')[:300]}")
    print("ANSWER:", result["answer"])
