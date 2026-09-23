"""
Short-term memory and run accounting for one recipe task.

Memory lives only for the question currently being answered. A new question
creates a new TaskMemory. Nothing is written to disk.
"""
import time

import config


class TaskMemory:
    """Tool results remembered while the current task is in progress."""

    def __init__(self, goal: str):
        self.goal = goal
        self.notes: list[dict] = []
        # Full cookbook excerpt from the latest search. Later tools read this
        # so the model does not have to copy a long recipe back into JSON.
        self.recipe_text = ""

    def add(self, kind: str, content: str) -> None:
        self.notes.append({"kind": kind, "content": content})

    def as_text(self) -> str:
        """Text block pasted into the next Think prompt."""
        if not self.notes and not self.recipe_text:
            return "(nothing remembered yet for this task)"

        lines = [f"Goal: {self.goal}"]
        if self.recipe_text:
            preview = self.recipe_text[:400].replace("\n", " ")
            lines.append(
                "Saved recipe text is available to check_ingredients and "
                f"calculate_nutrition. Preview: {preview}"
            )
        for index, note in enumerate(self.notes, start=1):
            content = note["content"]
            if len(content) > 700:
                content = content[:700] + "..."
            lines.append(f"{index}. [{note['kind']}] {content}")
        return "\n".join(lines)

    def export(self) -> list[dict]:
        """What the UI shows after the task. The full recipe text stays truncated."""
        exported = list(self.notes)
        if self.recipe_text:
            preview = self.recipe_text[:500]
            if len(self.recipe_text) > 500:
                preview += "..."
            exported.append({"kind": "recipe_text", "content": preview})
        return exported


class RunStats:
    """Token, cost, and tool-call counters for one agent or workflow run."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_estimated = False
        self.tool_calls = 0
        self.llm_calls = 0

    def add_usage(self, prompt_tokens: int, completion_tokens: int, estimated: bool = False) -> None:
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            return
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.llm_calls += 1
        if estimated:
            self.usage_estimated = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self) -> float:
        if config.LLM_PROVIDER != "openai":
            return 0.0
        return (
            self.prompt_tokens * config.OPENAI_INPUT_COST_PER_1M
            + self.completion_tokens * config.OPENAI_OUTPUT_COST_PER_1M
        ) / 1_000_000


def limit_reason(started: float, total_tokens: int) -> str | None:
    """Return a safeguard name when the run should stop, else None."""
    if time.perf_counter() - started > config.AGENT_TIMEOUT_SECONDS:
        return "timeout"
    if total_tokens > config.AGENT_MAX_TOTAL_TOKENS:
        return "token_budget"
    return None


def stop_step(step_number: int, reason: str) -> dict:
    messages = {
        "timeout": f"Stopped: ran longer than {config.AGENT_TIMEOUT_SECONDS:g} seconds.",
        "token_budget": f"Stopped: used more than {config.AGENT_MAX_TOTAL_TOKENS} tokens.",
        "max_steps": f"Stopped: reached MAX_STEPS ({config.AGENT_MAX_STEPS}).",
    }
    return make_step(
        step_number,
        thought="A safeguard ended the run.",
        observation=messages[reason],
    )


def make_step(step: int, thought: str, tool: str = None, arguments: dict = None,
              observation: str = "", final: bool = False) -> dict:
    return {
        "step": step,
        "thought": thought,
        "tool": tool,
        "arguments": arguments or {},
        "observation": observation,
        "final": final,
    }


def package_run(mode: str, question: str, answer: str, status: str, steps: list,
                memory: TaskMemory, stats: RunStats, started: float) -> dict:
    return {
        "mode": mode,
        "question": question,
        "answer": answer,
        "status": status,
        "steps": steps,
        "memory": memory.export(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.total_tokens,
        "cost_usd": round(stats.cost_usd(), 6),
        "tool_calls": stats.tool_calls,
        "llm_calls": stats.llm_calls,
        "usage_estimated": stats.usage_estimated,
    }
