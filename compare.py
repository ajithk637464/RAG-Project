"""
Run the same recipe questions through the agent and the fixed workflow.

A case counts as a success only when that system finished with status
"completed" and the final answer contains one of the expected phrases.
Stopping on MAX_STEPS, timeout, token budget, or an error is a failure.
An honest refusal counts as success on the "not in the cookbook" case.
"""
import json

import agent
import workflow

TEST_CASES = [
    {
        "name": "Known recipe, partial pantry",
        "question": (
            "I have tuna, pasta, and cheese. "
            "Can I make the Tuna and Pasta Bake? "
            "Tell me what I am missing, suggest a substitute, and estimate nutrition."
        ),
        "expect_any": ["tuna", "pasta bake", "conchiglie", "knorr herb"],
    },
    {
        "name": "Second known recipe",
        "question": (
            "I have pasta and tomatoes but no cream. "
            "How do I make Pasta Primavera, what am I missing, "
            "what can I substitute, and what is a rough nutrition estimate?"
        ),
        "expect_any": ["primavera"],
    },
    {
        "name": "Not in the cookbook",
        "question": (
            "I have chocolate and flour. "
            "How do I make a chocolate lava cake? "
            "Check ingredients, suggest a substitute, and estimate nutrition."
        ),
        "expect_any": [
            "couldn't find",
            "could not find",
            "not in the cookbook",
            "not in your cookbook",
        ],
    },
]


def succeeded(result: dict, case: dict) -> bool:
    if result.get("status") != "completed":
        return False
    answer = (result.get("answer") or "").lower()
    return any(phrase.lower() in answer for phrase in case["expect_any"])


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def summarize(rows: list[dict]) -> dict:
    count = len(rows)

    def side(system: str, field: str) -> list:
        return [row[system][field] for row in rows]

    def block(system: str, success_key: str) -> dict:
        successes = sum(1 for row in rows if row[success_key])
        return {
            "avg_seconds": _average(side(system, "elapsed_seconds")),
            "avg_tokens": _average(side(system, "total_tokens")),
            "avg_cost_usd": _average(side(system, "cost_usd")),
            "avg_tool_calls": _average(side(system, "tool_calls")),
            "successes": successes,
            "failures": count - successes,
            "success_rate": round(successes / count, 2) if count else 0.0,
        }

    return {
        "cases": count,
        "agent": block("agent", "agent_success"),
        "workflow": block("workflow", "workflow_success"),
    }


def run_comparison(cases: list[dict] = None) -> dict:
    """Execute every case on both systems. This calls the LLM."""
    cases = cases or TEST_CASES
    rows = []
    for case in cases:
        agent_result = agent.run_agent(case["question"])
        workflow_result = workflow.run_workflow(case["question"])
        rows.append({
            "name": case["name"],
            "question": case["question"],
            "expect_any": case["expect_any"],
            "agent": agent_result,
            "workflow": workflow_result,
            "agent_success": succeeded(agent_result, case),
            "workflow_success": succeeded(workflow_result, case),
        })
    return {"rows": rows, "summary": summarize(rows)}


def format_report(report: dict) -> str:
    lines = ["Recipe agent vs fixed workflow", ""]
    for row in report["rows"]:
        lines.append(f"Case: {row['name']}")
        lines.append(f"  Question: {row['question']}")
        for label, key, success_key in (
            ("Agent", "agent", "agent_success"),
            ("Workflow", "workflow", "workflow_success"),
        ):
            result = row[key]
            lines.append(
                f"  {label}: {'success' if row[success_key] else 'failure'} | "
                f"status={result['status']} | {result['elapsed_seconds']}s | "
                f"tokens={result['total_tokens']} | cost=${result['cost_usd']:.4f} | "
                f"tool_calls={result['tool_calls']} | llm_calls={result['llm_calls']}"
            )
            lines.append(f"    Answer: {result['answer'][:400].replace(chr(10), ' ')}")
        lines.append("")

    summary = report["summary"]
    lines.append("Averages")
    for label, key in (("Agent", "agent"), ("Workflow", "workflow")):
        block = summary[key]
        lines.append(
            f"  {label}: time={block['avg_seconds']}s | tokens={block['avg_tokens']} | "
            f"cost=${block['avg_cost_usd']:.4f} | tool_calls={block['avg_tool_calls']} | "
            f"success={block['successes']}/{summary['cases']} ({block['success_rate']:.0%}) | "
            f"failure={block['failures']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    report = run_comparison()
    print(format_report(report))
    print("\n--- summary json ---")
    print(json.dumps(report["summary"], indent=2))
