"""Streamlit UI for the Agentic AI RAG chatbot.  Run: streamlit run streamlit_app.py"""
import streamlit as st

from src.graph import build_rag_graph, run_query

st.set_page_config(page_title="Agentic AI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("Agentic AI eBook Chatbot")
st.caption("Answers strictly from the Agentic AI eBook. Out-of-scope questions are refused.")


@st.cache_resource
def get_graph():
    return build_rag_graph()


if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something about the eBook...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"), st.spinner("Retrieving and verifying..."):
        result = run_query(get_graph(), query)
        st.markdown(result["final_answer"])
    st.session_state.history.append({"role": "assistant", "content": result["final_answer"]})

    with st.sidebar:
        st.metric("Confidence score", f"{result['confidence_score']:.2f}")
        st.subheader("Retrieved context")
        for chunk, src in zip(result["retrieved_context_chunks"], result["sources"]):
            used = "✅ used" if src["used_in_answer"] else "⬜ not used"
            with st.expander(f"Page {src['page']} · similarity {src['similarity']:.2f} · {used}"):
                st.write(chunk)