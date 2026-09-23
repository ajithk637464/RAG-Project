"""
Manual recipe agent.

The loop is ordinary Python, not a framework:

    Think → Tool → Observe → Repeat → Final Answer

Think asks the existing LLM what to do next and expects one JSON object.
Tool runs a function in tools.py. Observe writes that result into short-term
memory, which is shown to the model on the next Think.
"""
import json
import re
import time

import config
import rag
import safety
import tools
from memory import RunStats, TaskMemory, limit_reason, make_step, package_run, stop_step

AGENT_SYSTEM = """You are a recipe agent for one cookbook. You do not answer from your own cooking knowledge. You call tools, read what they return, and then answer.

Reply with ONE JSON object and no other text.

Call a tool:
{"thought":"why this call is needed","action":"tool","tool":"search_recipes","arguments":{"query":"tuna pasta bake"}}

Final answer:
{"thought":"why this is enough","action":"final","final_answer":"short answer using only tool results"}

Tools:
- search_recipes: arguments {"query": string}. Searches the cookbook. Call this first.
- check_ingredients: arguments {"available_ingredients": string}. Compares the user's pantry with the saved recipe. Omit recipe_text; short-term memory supplies it.
- find_substitute: arguments {"ingredient": string}. Looks up one ingredient in a small substitute table. Call this at most once.
- calculate_nutrition: arguments {}. Rough calorie estimate from the saved recipe. Call this at most once.

Rules:
- Call search_recipes before any final answer.
- Do not call the same tool with the same arguments twice.
- After search, ingredient check, one substitute, and nutrition, give the final answer.
- If search says the cookbook does not contain the answer, the final answer must say you could not find it. Do not invent a recipe.
- Nutrition numbers must come from calculate_nutrition.
"""


def _extract_json(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in the model reply")
    return cleaned[start:end + 1]


def parse_decision(text: str) -> dict:
    """Turn the model's reply into one decision dict, or raise ValueError."""
    blob = _extract_json(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*}", "}", blob)
        repaired = re.sub(r",\s*]", "]", repaired)
        data = json.loads(repaired)
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), None)
    if not isinstance(data, dict):
        raise ValueError("JSON was not an object")
    return data


def _action_of(decision: dict) -> str:
    action = str(decision.get("action") or decision.get("type") or "").strip().lower()
    if action in ("final", "finish", "answer"):
        return "final"
    if decision.get("final_answer") and not (decision.get("tool") or decision.get("function")):
        return "final"
    return "tool"


def _tool_name(decision: dict) -> str:
    name = decision.get("tool") or decision.get("function") or decision.get("name")
    if not name:
        action = str(decision.get("action") or "").strip()
        if action in tools.TOOLS or action in tools.ALIASES:
            name = action
    return str(name or "").strip()


def _arguments(decision: dict) -> dict:
    arguments = decision.get("arguments")
    if arguments is None:
        arguments = decision.get("args") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    for key in ("query", "ingredient", "available_ingredients", "recipe_text", "question"):
        if key in decision and key not in arguments and decision[key] is not None:
            arguments[key] = decision[key]
    return arguments


def _think(question: str, memory: TaskMemory, step_number: int) -> tuple[str, dict]:
    reminder = ""
    if step_number >= config.AGENT_MAX_STEPS - 1:
        reminder = (
            "\n\nThis is your last step. Reply with action final. "
            "Use only the tool results in memory. If search said the recipe "
            "is not in the cookbook, say you could not find it."
        )
    document_block = ""
    if getattr(memory, "show_poison", False):
        document_block = (
            "\n\nRetrieved document note:\n"
            f"{safety.POISON_NOTE}\n"
        )
    result = rag.complete_chat(
        [
            {"role": "system", "content": AGENT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Task:\n{question}\n\n"
                    f"Short-term memory for this task:\n{memory.as_text()}"
                    f"{document_block}"
                    f"{reminder}"
                ),
            },
        ],
        max_tokens=700,
    )
    return result["content"], result


def _fallback_answer(status: str, memory: TaskMemory) -> str:
    return (
        f"The agent stopped ({status}) before a final answer.\n\n"
        f"Short-term memory for this task:\n{memory.as_text()}"
    )


def _remember_tool(memory: TaskMemory, name: str, public: dict) -> None:
    """Keep the fields the controller and the output check need."""
    if name == "search_recipes":
        memory.search_answer = public.get("answer") or ""
        memory.recipe_text = public.get("recipe_text") or ""
        memory.search_refused = memory.search_answer.strip().lower() in {
            "i couldn't find that in the cookbook.",
            "i couldn't find that in the cookbook",
        }
    elif name == "check_ingredients":
        memory.missing = list(public.get("missing") or [])
    elif name == "find_substitute":
        if public.get("substitute"):
            memory.substitute_note = f"Substitute for {public.get('ingredient')}: {public['substitute']}."
        elif public.get("note"):
            memory.substitute_note = public["note"]
        elif isinstance(public.get("substitutes"), list):
            memory.substitute_note = tools.to_text(public["substitutes"])
    elif name == "calculate_nutrition":
        if public.get("estimated_kcal") is not None:
            memory.nutrition_kcal = public.get("estimated_kcal")


def _controller_tool(name: str, memory: TaskMemory, question: str) -> tuple[dict, dict]:
    """Run the next required tool with saved memory. The model does not choose the arguments."""
    if name == "check_ingredients":
        return tools.run_tool("check_ingredients", {
            "recipe_text": memory.recipe_text,
            "available_ingredients": tools.extract_pantry(question),
        })
    if name == "find_substitute":
        missing = memory.missing or []
        if not missing:
            return {"note": "No missing ingredients, so no substitute was looked up."}, {
                "prompt_tokens": 0, "completion_tokens": 0, "usage_estimated": False,
            }
        return tools.run_tool("find_substitute", {"ingredient": missing[0]})
    if name == "calculate_nutrition":
        return tools.run_tool("calculate_nutrition", {"recipe_text": memory.recipe_text})
    return {"error": f"No controller path for {name}."}, {
        "prompt_tokens": 0, "completion_tokens": 0, "usage_estimated": False,
    }


def _document_note(defenses: bool) -> str:
    """Poisoned document note, or a short line saying the defense removed it."""
    if not defenses:
        return safety.POISON_NOTE
    return "Removed a document note that looked like an instruction to the agent."


def _guard_observation(observation: str, poison: bool = False) -> str:
    """Strip instruction-shaped lines before they are written into memory."""
    del poison
    return safety.sanitize_document(observation)


def _visible_tool_result(public: dict, from_memory: bool) -> str:
    visible = dict(public)
    if "recipe_text" in visible:
        visible["recipe_text"] = (
            f"saved to short-term memory ({len(public.get('recipe_text') or '')} characters)"
        )
    observation = tools.to_text(visible)
    if from_memory:
        observation = "Used recipe text from short-term memory. " + observation
    return observation


def run_agent(question: str, defenses: bool = True, poison: bool = False) -> dict:
    """Run Think → Tool → Observe until a final answer or a safeguard."""
    memory = TaskMemory(question)
    stats = RunStats()
    steps = []
    started = time.perf_counter()
    status = "max_steps"
    answer = ""
    seen = set()
    searched = False
    called = []
    rejected_finals = 0

    if defenses and safety.is_direct_injection(question):
        answer = (
            "I can't follow an instruction to ignore the cookbook. "
            "Ask a recipe question and I will search it."
        )
        steps.append(make_step(
            1,
            "The user message tried to override the agent instructions.",
            observation=answer,
            final=True,
        ))
        result = package_run("agent", question, answer, "completed", steps, memory, stats, started)
        result["defenses"] = True
        return result

    for step_number in range(1, config.AGENT_MAX_STEPS + 1):
        reason = limit_reason(started, stats.total_tokens)
        if reason:
            status = reason
            steps.append(stop_step(step_number, reason))
            break

        try:
            raw, usage = _think(question, memory, step_number)
        except RuntimeError as exc:
            status = "error"
            answer = str(exc)
            steps.append(make_step(step_number, "The language model call failed.", observation=answer))
            break

        stats.add_usage(usage["prompt_tokens"], usage["completion_tokens"], usage["usage_estimated"])

        try:
            decision = parse_decision(raw)
        except (ValueError, json.JSONDecodeError):
            observation = "That reply was not valid JSON. Reply with one JSON object."
            memory.add("observe", observation)
            steps.append(make_step(
                step_number,
                thought="The model reply could not be parsed.",
                observation=f"{observation}\n\nRaw reply: {raw[:500]}",
            ))
            continue

        thought = str(decision.get("thought") or "").strip() or "No thought was provided."

        if _action_of(decision) == "final":
            if not searched:
                observation = "search_recipes has not been called yet. Call it before the final answer."
                memory.add("observe", observation)
                steps.append(make_step(step_number, thought, observation=observation))
                continue
            proposed = str(decision.get("final_answer") or "").strip()
            if not proposed:
                observation = "final_answer was empty. Send the final answer, or call another tool."
                memory.add("observe", observation)
                steps.append(make_step(step_number, thought, observation=observation))
                continue

            if defenses:
                missing = safety.missing_tools(question, called, memory.search_refused)
                if missing and not memory.search_refused:
                    next_tool = missing[0]
                    public, tool_usage = _controller_tool(next_tool, memory, question)
                    stats.add_usage(
                        tool_usage["prompt_tokens"],
                        tool_usage["completion_tokens"],
                        tool_usage["usage_estimated"],
                    )
                    stats.tool_calls += 1
                    called.append(next_tool)
                    _remember_tool(memory, next_tool, public)
                    observation = _visible_tool_result(public, next_tool != "find_substitute")
                    if defenses:
                        observation = _guard_observation(observation, poison=False)
                    memory.add(next_tool, observation)
                    steps.append(make_step(
                        step_number,
                        f"The model tried to answer before {next_tool}. The controller ran that tool.",
                        next_tool,
                        {"source": "controller"},
                        observation,
                    ))
                    continue

                problems = safety.validate_final(proposed, memory.search_refused, memory.nutrition_kcal)
                if problems:
                    rejected_finals += 1
                    if rejected_finals >= 2 or step_number >= config.AGENT_MAX_STEPS:
                        answer = safety.answer_from_tools(memory)
                        memory.add("final_answer", answer)
                        steps.append(make_step(
                            step_number,
                            "Output check replaced the answer with the tool results. "
                            + " ".join(problems),
                            observation=answer,
                            final=True,
                        ))
                        status = "completed"
                        break
                    observation = (
                        "Final answer rejected: " + " ".join(problems) + ". "
                        "Use only the tool results in memory."
                    )
                    if memory.nutrition_kcal is not None:
                        observation += f" The nutrition tool said {memory.nutrition_kcal} kcal."
                    memory.add("observe", observation)
                    steps.append(make_step(step_number, thought, observation=observation))
                    continue

            answer = proposed
            memory.add("final_answer", answer)
            steps.append(make_step(step_number, thought, observation=answer, final=True))
            status = "completed"
            break

        tool_name = _tool_name(decision)
        arguments = _arguments(decision)
        resolved_name = tools.ALIASES.get(tool_name, tool_name)
        if resolved_name != "search_recipes" and not searched:
            observation = "Call search_recipes before any other tool."
            memory.add("observe", observation)
            steps.append(make_step(step_number, thought, tool_name or resolved_name, arguments, observation))
            continue

        if resolved_name == "search_recipes" and searched:
            observation = "search_recipes was already used. Call the next tool, or give the final answer."
            memory.add("observe", observation)
            steps.append(make_step(step_number, thought, tool_name or resolved_name, arguments, observation))
            continue

        signature = (tool_name, json.dumps(arguments, sort_keys=True, default=str))
        if signature in seen:
            observation = (
                "That tool was already called with the same arguments. "
                "Use the memory above or give the final answer."
            )
            memory.add("observe", observation)
            steps.append(make_step(step_number, thought, tool_name, arguments, observation))
            continue
        seen.add(signature)

        call_args = dict(arguments)
        from_memory = False
        if resolved_name in ("check_ingredients", "calculate_nutrition"):
            supplied = str(call_args.get("recipe_text") or "").strip()
            # A recipe name or the phrase "from short-term memory" is not the excerpt.
            if memory.recipe_text and len(supplied) < 80:
                call_args["recipe_text"] = memory.recipe_text
                from_memory = True
        if tool_name in ("check_ingredients", "ingredients"):
            if not str(call_args.get("available_ingredients") or call_args.get("available") or "").strip():
                pantry = tools.extract_pantry(question)
                if pantry:
                    call_args["available_ingredients"] = pantry

        public, tool_usage = tools.run_tool(tool_name, call_args)
        stats.add_usage(
            tool_usage["prompt_tokens"],
            tool_usage["completion_tokens"],
            tool_usage["usage_estimated"],
        )
        stats.tool_calls += 1

        if public.get("error") and "Could not reach Ollama" in str(public["error"]):
            status = "error"
            answer = public["error"]
            steps.append(make_step(step_number, thought, tool_name, arguments, answer))
            break

        if resolved_name == "search_recipes":
            searched = True
            _remember_tool(memory, resolved_name, public)
            if poison:
                public = dict(public)
                public["document_note"] = _document_note(defenses)
                if not defenses:
                    memory.show_poison = True
                    memory.recipe_text = safety.POISON_NOTE + "\n" + (memory.recipe_text or "")
        elif resolved_name in ("check_ingredients", "find_substitute", "calculate_nutrition"):
            _remember_tool(memory, resolved_name, public)
        if resolved_name:
            called.append(resolved_name)

        observation = _visible_tool_result(public, from_memory)
        if defenses:
            observation = _guard_observation(observation, poison=False)

        memory.add(resolved_name or tool_name or "tool", observation)
        display_args = dict(arguments)
        if from_memory:
            display_args["recipe_text"] = "(from short-term memory)"
        steps.append(make_step(step_number, thought, tool_name or resolved_name, display_args, observation))
    else:
        steps.append(stop_step(len(steps) + 1, "max_steps"))

    if not answer and defenses:
        missing = safety.missing_tools(question, called, memory.search_refused)
        if memory.search_refused or not missing:
            answer = safety.answer_from_tools(memory)
            status = "completed"
            steps.append(make_step(
                len(steps) + 1,
                "The step limit was reached after the required tools, so the answer was taken from those results.",
                observation=answer,
                final=True,
            ))
    if not answer:
        answer = _fallback_answer(status, memory)

    result = package_run("agent", question, answer, status, steps, memory, stats, started)
    result["defenses"] = defenses
    return result


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip() or (
        "I have tuna, pasta, and cheese. Can I make the Tuna and Pasta Bake? "
        "Tell me what I am missing, suggest a substitute, and estimate nutrition."
    )
    result = run_agent(question)
    print(f"status={result['status']} tools={result['tool_calls']} "
          f"tokens={result['total_tokens']} seconds={result['elapsed_seconds']}")
    print(result["answer"])
    for step in result["steps"]:
        label = step["tool"] or ("final" if step["final"] else "note")
        print(f"  step {step['step']} [{label}] {step['thought'][:120]}")
