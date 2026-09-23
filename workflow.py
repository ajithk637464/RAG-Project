"""
Fixed recipe workflow.

The same task as the agent, but the program — not the model — chooses the
order. Every run follows:

    RAG Search → Ingredient Check → Substitute → Nutrition → Final Answer

The only model call is inside search_recipes. The final answer is assembled
in code from the tool results, so a small model cannot overwrite the
ingredient check, the substitute table, or the calorie estimate. If search
already refused, that refusal is the final answer.
"""
import time

import tools
from memory import RunStats, TaskMemory, limit_reason, make_step, package_run, stop_step


def _search_refused(search_public: dict) -> bool:
    """True only when search returned the cookbook refusal and nothing else."""
    answer = " ".join((search_public.get("answer") or "").split()).strip().lower()
    return answer in {
        "i couldn't find that in the cookbook.",
        "i couldn't find that in the cookbook",
        "i could not find that in the cookbook.",
        "i could not find that in the cookbook",
    }


def _compose_answer(search_public: dict, check_public: dict,
                    substitutes: list[dict], nutrition_public: dict) -> str:
    """Final answer from tool results only. No second language-model call."""
    if _search_refused(search_public):
        return "I couldn't find that in the cookbook."

    present = ", ".join(check_public.get("present") or []) or "none"
    missing = ", ".join(check_public.get("missing") or []) or "none"
    lines = [(search_public.get("answer") or "I couldn't find that in the cookbook.").strip(), ""]
    lines.append(f"Ingredient check - already have: {present}.")
    lines.append(f"Ingredient check - missing: {missing}.")
    if substitutes:
        lines.append("Substitutes:")
        for item in substitutes:
            substitute = item.get("substitute") or item.get("note") or "none listed"
            lines.append(f"- {item.get('ingredient')}: {substitute}")
    else:
        lines.append("Substitutes: none needed.")
    kcal = nutrition_public.get("estimated_kcal", 0)
    lines.append(
        f"Nutrition tool: about {kcal} kcal for the recognized ingredients. "
        "This is a rough teaching estimate, not a dietary measurement."
    )
    return "\n".join(lines)


def _record_tool(stats: RunStats, usage: dict) -> None:
    stats.tool_calls += 1
    stats.add_usage(usage["prompt_tokens"], usage["completion_tokens"], usage["usage_estimated"])


def _visible_search(public: dict) -> str:
    visible = dict(public)
    if "recipe_text" in visible:
        visible["recipe_text"] = (
            f"saved to short-term memory ({len(public.get('recipe_text') or '')} characters)"
        )
    return tools.to_text(visible)


def run_workflow(question: str) -> dict:
    """Run the four tools in a fixed order, then assemble the final answer."""
    memory = TaskMemory(question)
    stats = RunStats()
    steps = []
    started = time.perf_counter()
    status = "completed"
    answer = ""
    pantry = tools.extract_pantry(question)
    search_public = {}
    check_public = {}
    substitute_results = []
    nutrition_public = {}

    def halted(step_number: int) -> bool:
        nonlocal status
        reason = limit_reason(started, stats.total_tokens)
        if not reason:
            return False
        status = reason
        steps.append(stop_step(step_number, reason))
        return True

    # 1. RAG search — always the existing retrieval + grounded answer.
    if not halted(1):
        public, usage = tools.run_tool("search_recipes", {"query": question})
        _record_tool(stats, usage)
        if public.get("error") and "Could not reach Ollama" in str(public["error"]):
            status = "error"
            answer = public["error"]
            steps.append(make_step(
                1,
                "Fixed workflow: always search the cookbook first.",
                "search_recipes",
                {"query": question},
                answer,
            ))
        else:
            search_public = public
            memory.recipe_text = public.get("recipe_text") or ""
            observation = _visible_search(public)
            memory.add("search_recipes", observation)
            steps.append(make_step(
                1,
                "Fixed workflow: always search the cookbook first.",
                "search_recipes",
                {"query": question},
                observation,
            ))

    # 2. Ingredient check against the pantry phrase in the question.
    if status == "completed" and not halted(2):
        public, usage = tools.run_tool("check_ingredients", {
            "recipe_text": memory.recipe_text,
            "available_ingredients": pantry,
        })
        _record_tool(stats, usage)
        check_public = public
        observation = tools.to_text(public)
        memory.add("check_ingredients", observation)
        steps.append(make_step(
            2,
            "Fixed workflow: compare the saved recipe with the ingredients the user said they have.",
            "check_ingredients",
            {
                "available_ingredients": pantry or "(none stated)",
                "recipe_text": "(from short-term memory)",
            },
            observation,
        ))
        missing = check_public.get("missing") or []
    else:
        missing = []

    # 3. One substitute lookup per missing ingredient, up to three.
    if status == "completed" and not halted(3):
        if missing:
            for ingredient in missing[:3]:
                public, usage = tools.run_tool("find_substitute", {"ingredient": ingredient})
                _record_tool(stats, usage)
                substitute_results.append(public)
            observation = tools.to_text(substitute_results)
            display_args = {"ingredient": missing[:3]}
        else:
            observation = tools.to_text({
                "note": "No missing ingredients were found, so no substitute was looked up.",
            })
            display_args = {}
        memory.add("find_substitute", observation)
        steps.append(make_step(
            3,
            "Fixed workflow: look up a substitute for each missing ingredient.",
            "find_substitute",
            display_args,
            observation,
        ))

    # 4. Nutrition from the same saved recipe text.
    if status == "completed" and not halted(4):
        public, usage = tools.run_tool("calculate_nutrition", {"recipe_text": memory.recipe_text})
        _record_tool(stats, usage)
        nutrition_public = public
        observation = tools.to_text(public)
        memory.add("calculate_nutrition", observation)
        steps.append(make_step(
            4,
            "Fixed workflow: estimate calories from the ingredients recognized in the saved recipe.",
            "calculate_nutrition",
            {"recipe_text": "(from short-term memory)"},
            observation,
        ))

    # 5. Final answer is assembled from the tool results. Search is the only
    # model call, so this stage cannot invent a different ingredient list.
    if status == "completed" and not halted(5):
        answer = _compose_answer(
            search_public, check_public, substitute_results, nutrition_public
        )
        memory.add("final_answer", answer)
        thought = (
            "Search did not find the recipe, so the final answer is that refusal."
            if _search_refused(search_public)
            else "Fixed workflow: assemble the final answer from the tool results."
        )
        steps.append(make_step(5, thought, observation=answer, final=True))

    if not answer:
        answer = (
            f"The workflow stopped ({status}) before a final answer.\n\n"
            f"Short-term memory for this task:\n{memory.as_text()}"
        )

    return package_run("workflow", question, answer, status, steps, memory, stats, started)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip() or (
        "I have tuna, pasta, and cheese. Can I make the Tuna and Pasta Bake? "
        "Tell me what I am missing, suggest a substitute, and estimate nutrition."
    )
    result = run_workflow(question)
    print(f"status={result['status']} tools={result['tool_calls']} "
          f"tokens={result['total_tokens']} seconds={result['elapsed_seconds']}")
    print(result["answer"])
