from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Rulebook Assistant", page_icon="📘", layout="wide")
st.title("📘 Rulebook Assistant")
st.caption("Ask grounded questions about your PDFs. Answers include page references.")

with st.sidebar:
    st.header("Knowledge base")
    api_url = st.text_input("Local API", "http://127.0.0.1:5000")
    uploaded = st.file_uploader("Add a rulebook PDF", type=["pdf"])
    if uploaded and st.button("Index PDF", use_container_width=True):
        with st.spinner("Extracting text, summarising tables, and building the search index…"):
            try:
                response = requests.post(
                    f"{api_url}/ingest", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}, timeout=300
                )
                response.raise_for_status()
                info = response.json()
                st.session_state.document_id = info["document_id"]
                st.success(f"Indexed {info['chunks']} passages from {info['filename']}.")
            except requests.RequestException as exc:
                st.error(f"Could not index PDF: {exc}")
    st.divider()
    st.caption("Runs locally. Files and vector index are stored on this machine.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about the rulebook…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the rulebook…"):
            try:
                response = requests.post(
                    f"{api_url}/chat",
                    json={"question": question, "document_id": st.session_state.get("document_id"), "top_k": 5},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
                answer = result["answer"]
                if result["sources"]:
                    citations = ", ".join(f"{source['filename']} p. {source['page']}" for source in result["sources"])
                    answer += f"\n\n*Sources: {citations}*"
                st.markdown(answer)
                if result.get("evidence"):
                    with st.expander("Retrieved evidence"):
                        st.caption("These are the exact passages the answer was allowed to use.")
                        for item in result["evidence"]:
                            st.markdown(f"**Page {item['page']}** · {item['kind']} · match {item['similarity']}")
                            st.write(item["text"])
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except requests.RequestException as exc:
                message = f"I couldn't reach the local API: {exc}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
