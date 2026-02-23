import streamlit as st
from engine import get_rag_chain

st.set_page_config(page_title="AI Intern Assistant", page_icon="🤖")
st.title("📚 RAG Knowledge Base")

if "rag_chain" not in st.session_state:
    with st.spinner("Initializing AI..."):
        st.session_state.rag_chain = get_rag_chain()

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Searching documents..."):
        response = st.session_state.rag_chain.invoke({"query": query})
        st.markdown("### Answer")
        st.write(response["result"])