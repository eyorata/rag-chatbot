import streamlit as st
import requests
import uuid

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("Chat with your documents")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("LLM settings")
    st.caption("Set via backend .env — shown here for visibility")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")

    st.subheader("Documents")
    files = st.file_uploader("Upload files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Index files") and files:
        with st.spinner("Indexing..."):
            response = requests.post(
                f"{BACKEND_URL}/upload",
                params={"session_id": st.session_state.session_id},
                files=[("files", (f.name, f.getvalue())) for f in files],
            )
        if response.ok:
            data = response.json()
            st.success(f"Indexed {data['chunks_indexed']} chunks from {len(data['files'])} file(s)")
        else:
            st.error("Upload failed")

    if st.button("Clear session documents"):
        requests.delete(f"{BACKEND_URL}/session/{st.session_state.session_id}")
        st.session_state.messages = []
        st.success("Session cleared")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source_filename']}** ({s.get('section_title')}) — similarity {s['similarity']:.2f}")
                    st.caption(s["content"][:300] + "...")

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"session_id": st.session_state.session_id, "query": prompt},
        )

    if response.ok:
        data = response.json()
        st.session_state.messages.append({"role": "assistant", "content": data["answer"], "sources": data["sources"]})
        with st.chat_message("assistant"):
            st.write(data["answer"])
            if data["sources"]:
                with st.expander("Sources"):
                    for s in data["sources"]:
                        st.markdown(f"**{s['source_filename']}** ({s.get('section_title')}) — similarity {s['similarity']:.2f}")
                        st.caption(s["content"][:300] + "...")
    else:
        st.error("Request failed")