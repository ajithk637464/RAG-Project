"""
Streamlit UI for the cookbook chat, the manual recipe agent, the fixed
workflow, and the side-by-side comparison.

The cookbook tab still calls rag.answer_question(). The agent and workflow
tabs call agent.run_agent() and workflow.run_workflow() and show every step.
"""
import streamlit as st

import agent
import compare
import config
import eval_week8
import rag
import workflow

st.set_page_config(page_title="Ask My Cookbook", page_icon=None)
st.title("Ask My Cookbook")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "workflow_messages" not in st.session_state:
    st.session_state.workflow_messages = []
if "comparison" not in st.session_state:
    st.session_state.comparison = None
if "week8" not in st.session_state:
    st.session_state.week8 = None

mode = st.sidebar.radio(
    "Mode",
    ["Cookbook chat", "Recipe agent", "Fixed workflow", "Comparison", "Week 8 checks"],
)
st.sidebar.caption(
    f"Agent safeguards: MAX_STEPS={config.AGENT_MAX_STEPS}, "
    f"timeout={config.AGENT_TIMEOUT_SECONDS:g}s, "
    f"token budget={config.AGENT_MAX_TOTAL_TOKENS}."
)
st.sidebar.caption(f"LLM: {config.LLM_PROVIDER} / {config.LLM_MODEL}")


def format_sources(chunks: list[dict]) -> list[str]:
    """One citation line per source page, even if several chunks came from it."""
    seen = []
    for chunk in chunks:
        line = f"{chunk['source']} - Page {chunk['page']}"
        if chunk["recipe_name"]:
            line += f" (Recipe: {chunk['recipe_name']})"
        if line not in seen:
            seen.append(line)
    return seen


def show_chunks(chunks: list[dict]) -> None:
    with st.expander("Show retrieved chunks (debug)"):
        for chunk in chunks:
            distance = f"{chunk['distance']:.3f}" if chunk.get("distance") is not None else "n/a"
            st.markdown(
                f"**{chunk['source']} p.{chunk['page']}** | "
                f"RRF `{chunk.get('rrf_score', 0):.4f}` | "
                f"semantic rank `{chunk.get('semantic_rank', 'n/a')}` | "
                f"BM25 rank `{chunk.get('bm25_rank', 'n/a')}` | distance `{distance}`"
            )
            st.text(chunk["text"])


def render_steps(steps: list[dict]) -> None:
    """Show Think → Tool → Observe for every step the loop recorded."""
    for step in steps:
        if step.get("final"):
            title = f"Step {step['step']} — Final answer"
        elif step.get("tool"):
            title = f"Step {step['step']} — {step['tool']}"
        else:
            title = f"Step {step['step']} — Note"
        with st.expander(title, expanded=True):
            st.markdown("**Think**")
            st.write(step.get("thought") or "")
            if step.get("tool"):
                arguments = step.get("arguments") or {}
                argument_text = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
                st.markdown("**Tool**")
                st.code(f"{step['tool']}({argument_text})")
                st.markdown("**Observe**")
                st.write(step.get("observation") or "")
            elif step.get("final"):
                st.markdown("**Final answer**")
                st.write(step.get("observation") or "")
            else:
                st.markdown("**Observe**")
                st.write(step.get("observation") or "")


def render_run(result: dict) -> None:
    st.write(result["answer"])
    time_col, token_col, cost_col, tool_col, status_col = st.columns(5)
    time_col.metric("Time (s)", result["elapsed_seconds"])
    token_col.metric("Tokens", result["total_tokens"])
    cost_col.metric("Cost (USD)", f"{result['cost_usd']:.4f}")
    tool_col.metric("Tool calls", result["tool_calls"])
    status_col.metric("Status", result["status"])
    if result.get("usage_estimated"):
        st.caption("Some token counts were estimated because the model did not report usage.")
    elif config.LLM_PROVIDER != "openai":
        st.caption("Cost is $0 because the model is running locally. Token counts are still tracked.")
    st.markdown("**Steps**")
    render_steps(result["steps"])
    with st.expander("Short-term memory for this task"):
        if not result["memory"]:
            st.write("Nothing was remembered.")
        for note in result["memory"]:
            st.markdown(f"**{note['kind']}**")
            st.text(note["content"])


def render_cookbook() -> None:
    st.caption(
        "Answers come only from your cookbook PDF. "
        f"Chunk config '{config.ACTIVE_CHUNK_CONFIG}' "
        f"({config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}), TOP_K={config.TOP_K}."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("chunks"):
                st.markdown("**Sources:**")
                for line in format_sources(msg["chunks"]):
                    st.markdown(f"- {line}")
                show_chunks(msg["chunks"])

    question = st.chat_input("Ask a question about your cookbook...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the cookbook..."):
            try:
                result = rag.answer_question(question)
                answer, chunks = result["answer"], result["chunks"]
            except RuntimeError as exc:
                answer, chunks = f"Setup error: {exc}", []
            except Exception as exc:
                answer, chunks = f"Unexpected error while generating the answer: {exc}", []

        st.write(answer)
        if chunks:
            st.markdown("**Sources:**")
            for line in format_sources(chunks):
                st.markdown(f"- {line}")
            show_chunks(chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})


def render_loop_mode(caption: str, history_key: str, runner) -> None:
    st.caption(caption)
    if st.button("Clear chat"):
        st.session_state[history_key] = []
        st.rerun()

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("result"):
                render_run(msg["result"])
            else:
                st.write(msg["content"])

    question = st.chat_input("Ask the recipe assistant...")
    if not question:
        return

    st.session_state[history_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                result = runner(question)
            except Exception as exc:
                result = {
                    "answer": f"Unexpected error: {exc}",
                    "status": "error",
                    "steps": [],
                    "memory": [],
                    "elapsed_seconds": 0,
                    "total_tokens": 0,
                    "cost_usd": 0,
                    "tool_calls": 0,
                    "usage_estimated": False,
                }
        render_run(result)

    st.session_state[history_key].append({
        "role": "assistant",
        "content": result["answer"],
        "result": result,
    })


def render_comparison() -> None:
    st.caption(
        "Runs the same three questions through the agent and the fixed workflow. "
        "A run succeeds when it finishes and the answer contains the expected phrase. "
        "The lava-cake question succeeds only when the answer says it is not in the cookbook."
    )
    if st.button("Run comparison"):
        with st.spinner("Running both systems. This makes several model calls and can take a few minutes."):
            st.session_state.comparison = compare.run_comparison()

    report = st.session_state.comparison
    if not report:
        st.write("No comparison yet. The three questions are:")
        for case in compare.TEST_CASES:
            st.markdown(f"- **{case['name']}** — {case['question']}")
        return

    summary = report["summary"]
    agent_summary = summary["agent"]
    workflow_summary = summary["workflow"]
    left, right = st.columns(2)
    left.metric("Agent success rate", f"{agent_summary['success_rate']:.0%}")
    right.metric("Workflow success rate", f"{workflow_summary['success_rate']:.0%}")

    table = []
    for row in report["rows"]:
        table.append({
            "Case": row["name"],
            "Agent time (s)": row["agent"]["elapsed_seconds"],
            "Workflow time (s)": row["workflow"]["elapsed_seconds"],
            "Agent tokens": row["agent"]["total_tokens"],
            "Workflow tokens": row["workflow"]["total_tokens"],
            "Agent cost (USD)": row["agent"]["cost_usd"],
            "Workflow cost (USD)": row["workflow"]["cost_usd"],
            "Agent tool calls": row["agent"]["tool_calls"],
            "Workflow tool calls": row["workflow"]["tool_calls"],
            "Agent": "success" if row["agent_success"] else "failure",
            "Workflow": "success" if row["workflow_success"] else "failure",
        })
    st.dataframe(table, use_container_width=True)

    average_table = [
        {
            "System": "Agent",
            "Avg time (s)": agent_summary["avg_seconds"],
            "Avg tokens": agent_summary["avg_tokens"],
            "Avg cost (USD)": agent_summary["avg_cost_usd"],
            "Avg tool calls": agent_summary["avg_tool_calls"],
            "Successes": f"{agent_summary['successes']}/{summary['cases']}",
            "Failures": agent_summary["failures"],
        },
        {
            "System": "Fixed workflow",
            "Avg time (s)": workflow_summary["avg_seconds"],
            "Avg tokens": workflow_summary["avg_tokens"],
            "Avg cost (USD)": workflow_summary["avg_cost_usd"],
            "Avg tool calls": workflow_summary["avg_tool_calls"],
            "Successes": f"{workflow_summary['successes']}/{summary['cases']}",
            "Failures": workflow_summary["failures"],
        },
    ]
    st.markdown("**Averages**")
    st.dataframe(average_table, use_container_width=True)

    for row in report["rows"]:
        with st.expander(row["name"]):
            st.write(row["question"])
            agent_tab, workflow_tab = st.tabs(["Recipe agent", "Fixed workflow"])
            with agent_tab:
                st.markdown(f"**{'Success' if row['agent_success'] else 'Failure'}**")
                render_run(row["agent"])
            with workflow_tab:
                st.markdown(f"**{'Success' if row['workflow_success'] else 'Failure'}**")
                render_run(row["workflow"])


def render_week8() -> None:
    st.caption(
        "Checks the path, not only the final answer. "
        "The top failure is a finished answer that skipped the ingredient, substitute, "
        "or nutrition tool. A second check hides an instruction in a retrieved note "
        "and shows whether the agent copies it."
    )
    st.markdown(
        "Defenses on the recipe agent: document lines that look like instructions are "
        "removed before they enter memory, a final answer is rejected when its calorie "
        "number did not come from `calculate_nutrition`, and a skipped required tool "
        "is run by the controller. Each tool can still only do the one job listed in "
        "`safety.TOOL_RIGHTS`."
    )
    if st.button("Run before/after eval"):
        with st.spinner("Running the baseline and the defended agent. This takes several minutes."):
            eval_week8.self_check()
            st.session_state.week8 = eval_week8.run_eval()

    report = st.session_state.week8
    if not report:
        st.write("No eval yet. The same three recipe questions are run with defenses off, then on.")
        return

    before = report["before"]
    after = report["after"]
    st.markdown(
        f"**Quiet give-up:** {before['quiet_give_up_rate']:.0%} before, "
        f"{after['quiet_give_up_rate']:.0%} after."
    )
    st.markdown(
        f"**Outcome vs trajectory gap:** {before['gap_count']} before, {after['gap_count']} after "
        f"(out of {before['cases']})."
    )
    st.markdown(
        f"**Planted phrase during the full agent loop:** "
        f"{'yes' if report['injection_before'] else 'no'} before the filter, "
        f"{'yes' if report['injection_after'] else 'no'} after it."
    )
    if "probe_before" in report:
        st.markdown(
            f"**Same note, one read with the agent's instructions:** "
            f"{'yes' if report['probe_before'] else 'no'} before the filter, "
            f"{'yes' if report['probe_after'] else 'no'} after it."
        )
    table = [
        {
            "Run": "Before",
            "Outcome rate": before["outcome_rate"],
            "Trajectory rate": before["trajectory_rate"],
            "Quiet give-up": before["quiet_give_up_rate"],
            "Made-up calories": before["made_up_input_rate"],
            "Tool-choice accuracy": before["mean_tool_choice_accuracy"],
            "Tokens mean": before["mean_tokens"],
            "Tokens p99": before["p99_tokens"],
            "Cost mean (USD)": before["mean_cost_usd"],
            "Cost p99 (USD)": before["p99_cost_usd"],
        },
        {
            "Run": "After",
            "Outcome rate": after["outcome_rate"],
            "Trajectory rate": after["trajectory_rate"],
            "Quiet give-up": after["quiet_give_up_rate"],
            "Made-up calories": after["made_up_input_rate"],
            "Tool-choice accuracy": after["mean_tool_choice_accuracy"],
            "Tokens mean": after["mean_tokens"],
            "Tokens p99": after["p99_tokens"],
            "Cost mean (USD)": after["mean_cost_usd"],
            "Cost p99 (USD)": after["p99_cost_usd"],
        },
    ]
    st.dataframe(table, use_container_width=True)
    st.markdown("**What can still get through**")
    st.markdown(
        "- A hidden instruction phrased differently from the patterns the filter knows.\n"
        "- A calorie number printed on the cookbook page, if the model copies that instead of the tool.\n"
        "- A user message that overrides the agent with different wording than the direct-injection check.\n"
        "- The model's first tool choice can still be wrong. The trace shows when the controller repairs it."
    )


if mode == "Cookbook chat":
    render_cookbook()
elif mode == "Recipe agent":
    render_loop_mode(
        "Manual loop: Think, then a tool, then observe, then repeat, until a final answer or a safeguard.",
        "agent_messages",
        agent.run_agent,
    )
elif mode == "Fixed workflow":
    render_loop_mode(
        "Fixed sequence: RAG search, ingredient check, substitute, nutrition, then a final answer.",
        "workflow_messages",
        workflow.run_workflow,
    )
elif mode == "Comparison":
    render_comparison()
else:
    render_week8()
