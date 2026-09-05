"""
Phase 8-9: Streamlit chat UI, with source citations for every answer.

This file is intentionally thin - all the real RAG logic (retrieval, LLM
generation) lives in rag.py. The UI just calls rag.answer_question() and
displays what comes back.
"""
import streamlit as st

import config
import rag

st.set_page_config(page_title="Ask My Cookbook", page_icon=None)
st.title("Ask My Cookbook")
st.caption(
    f"Answers come only from your cookbook PDF - chunk config: "
    f"'{config.ACTIVE_CHUNK_CONFIG}' ({config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}), TOP_K={config.TOP_K}"
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role", "content", "chunks"?}

col1, col2 = st.columns([5, 1])
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def format_sources(chunks: list[dict]) -> str:
    """
    Dedupe chunks down to one citation line per (source, page), so a question
    answered from three overlapping chunks on the same page doesn't repeat
    the same citation three times.
    """
    seen = []
    for c in chunks:
        line = f"{c['source']} - Page {c['page']}"
        if c["recipe_name"]:
            line += f" (Recipe: {c['recipe_name']})"
        if line not in seen:
            seen.append(line)
    return seen


# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            st.markdown("**Sources:**")
            for line in format_sources(msg["chunks"]):
                st.markdown(f"- {line}")
            with st.expander("Show retrieved chunks (debug)"):
                for c in msg["chunks"]:
                    distance = f"{c['distance']:.3f}" if c.get("distance") is not None else "n/a"
                    st.markdown(
                        f"**{c['source']} p.{c['page']}** | "
                        f"RRF `{c.get('rrf_score', 0):.4f}` | "
                        f"semantic rank `{c.get('semantic_rank', 'n/a')}` | "
                        f"BM25 rank `{c.get('bm25_rank', 'n/a')}` | distance `{distance}`"
                    )
                    st.text(c["text"])

question = st.chat_input("Ask a question about your cookbook...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the cookbook..."):
            try:
                result = rag.answer_question(question)
                answer, chunks = result["answer"], result["chunks"]
            except RuntimeError as e:
                answer, chunks = f"Setup error: {e}", []
            except Exception as e:
                answer, chunks = f"Unexpected error while generating the answer: {e}", []

        st.write(answer)
        if chunks:
            st.markdown("**Sources:**")
            for line in format_sources(chunks):
                st.markdown(f"- {line}")
            with st.expander("Show retrieved chunks (debug)"):
                for c in chunks:
                    distance = f"{c['distance']:.3f}" if c.get("distance") is not None else "n/a"
                    st.markdown(
                        f"**{c['source']} p.{c['page']}** | "
                        f"RRF `{c.get('rrf_score', 0):.4f}` | "
                        f"semantic rank `{c.get('semantic_rank', 'n/a')}` | "
                        f"BM25 rank `{c.get('bm25_rank', 'n/a')}` | distance `{distance}`"
                    )
                    st.text(c["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
