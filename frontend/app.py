import streamlit as st
import requests
import uuid
import json

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("Chat with your documents")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch active models from backend
@st.cache_data(ttl=60)
def get_models():
    try:
        r = requests.get(f"{BACKEND_URL}/models", timeout=3)
        if r.ok:
            data = r.json()
            return data.get("llm_models", ["default"]), data.get("embed_models", [])
    except:
        pass
    return ["ollama-default"], []

available_llms, available_embeds = get_models()

# Fetch active documents for this session
def get_documents():
    try:
        r = requests.get(f"{BACKEND_URL}/documents/{st.session_state.session_id}", timeout=5)
        if r.ok:
            return r.json().get("documents", [])
    except:
        pass
    return []

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Setup & Document Management
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Settings")
    selected_model = st.selectbox("LLM Model", options=available_llms)
    selected_embed = st.selectbox("Embedding Model (Info Only)", options=available_embeds, disabled=True)
    st.caption("Changes are advisory unless backend LLM_PROVIDER=ollama")
    st.text(f"Session: {st.session_state.session_id[:8]}...")
    
    st.divider()
    
    st.subheader("📚 Knowledge Base")
    docs = get_documents()
    
    # Show existing documents
    if docs:
        for d in docs:
            col1, col2 = st.columns([4, 1])
            size_str = f"{(d.get('file_size_bytes', 0) / 1024 / 1024):.1f}MB" if d.get('file_size_bytes') else "Unknown size"
            col1.markdown(f"**{d['filename']}**\n<small>{d['chunk_count']} chunks • {size_str}</small>", unsafe_allow_html=True)
            if col2.button("🗑️", key=f"del_{d['filename']}", help=f"Delete {d['filename']}"):
                requests.delete(f"{BACKEND_URL}/documents/{st.session_state.session_id}/{d['filename']}")
                st.rerun()
                
        if st.button("🗑️ Clear all documents", use_container_width=True):
            requests.delete(f"{BACKEND_URL}/session/{st.session_state.session_id}")
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # Upload new documents
    st.subheader("📤 Upload")
    files = st.file_uploader("Add files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
    if st.button("Index files", type="primary", use_container_width=True) and files:
        progress_text = "Uploading and indexing chunks..."
        my_bar = st.progress(0, text=progress_text)
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/upload",
                params={"session_id": st.session_state.session_id},
                files=[("files", (f.name, f.getvalue())) for f in files],
                timeout=300 # Embedding can take a while
            )
            my_bar.progress(100, text="Complete!")
            
            if response.ok:
                data = response.json()
                if data.get('errors'):
                    for e in data['errors']:
                        st.error(e)
                if data['chunks_indexed'] > 0:
                    st.success(f"Successfully indexed {data['chunks_indexed']} chunks.")
                st.rerun()
            else:
                st.error(f"Upload failed: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Main Area: Chat & Search Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_search = st.tabs(["💬 Chat", "🔍 Document Search"])

with tab_chat:
    if not docs:
        st.info("""
        ### 👋 Welcome to RAG Chatbot!
        
        To get started, upload some documents in the sidebar.
        You can upload:
        - **PDFs** (Reports, papers, manuals)
        - **Word Documents** (.docx)
        - **Text / Markdown** (.txt, .md)
        
        *Example question once uploaded: "What is the main conclusion of the report?"*
        """)
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                stats = f"Retrieved {msg.get('chunks_retrieved', 0)} chunks • Top match: {msg.get('top_similarity', 0):.2f}"
                st.caption(f"_{stats}_")
                with st.expander("View Sources"):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {s['source_filename']}** (_{s.get('section_title', 'Unknown')}_) — Sim: `{s['similarity']:.3f}`")
                        st.caption(f"\"{s['content']}\"")
                        st.divider()

    if prompt := st.chat_input("Ask a question about your documents", disabled=len(docs) == 0):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking... ⏳"):
                    # SSE Streaming client
                    response = requests.post(
                        f"{BACKEND_URL}/chat/stream",
                        json={"session_id": st.session_state.session_id, "query": prompt},
                        stream=True,
                        timeout=120
                    )
                    response.raise_for_status()

                    # Block the spinner until TTFT (Time To First Token) resolves
                    lines_iter = response.iter_lines()
                    try:
                        first_line = next(lines_iter)
                    except StopIteration:
                        first_line = None

                # Generate streaming text parsing reasoning tags
                def generate():
                    in_think = False
                    
                    def process_line(line):
                        nonlocal in_think
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                try:
                                    data = json.loads(data_str)
                                    if "content" in data:
                                        content = data["content"]
                                        if "<think>" in content:
                                            in_think = True
                                            content = content.replace("<think>", "💭 **Thinking...**\n\n_")
                                        if "</think>" in content:
                                            in_think = False
                                            content = content.replace("</think>", "_\n\n")
                                        return content
                                    if "error" in data:
                                        return f"\n\n**Error:** {data['error']}"
                                    if "metadata" in data:
                                        st.session_state.last_metadata = data["metadata"]
                                except Exception:
                                    pass
                        return ""
                    
                    if first_line:
                        chunk = process_line(first_line)
                        if chunk:
                            yield chunk
                            
                    for line in lines_iter:
                        chunk = process_line(line)
                        if chunk:
                            yield chunk

                # Render stream as it arrives
                full_response = st.write_stream(generate())
                
                # Fetch metadata that was populated at the end of the stream
                metadata = st.session_state.get("last_metadata", {})
                sources = metadata.get("sources", [])
                chunks_retrieved = metadata.get("chunks_retrieved", 0)
                top_similarity = metadata.get("top_similarity", 0.0)

                # Show stats immediately
                stats = f"Retrieved {chunks_retrieved} chunks • Top match: {top_similarity:.2f}"
                st.caption(f"_{stats}_")
                
                if sources:
                    with st.expander("View Sources"):
                        for i, s in enumerate(sources, 1):
                            st.markdown(f"**[{i}] {s['source_filename']}** (_{s.get('section_title', 'Unknown')}_) — Sim: `{s['similarity']:.3f}`")
                            st.caption(f"\"{s['content']}\"")
                            st.divider()

                # Save turn to local frontend state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "sources": sources,
                    "chunks_retrieved": chunks_retrieved,
                    "top_similarity": top_similarity
                })
                
                # Clear temp metadata
                if "last_metadata" in st.session_state:
                    del st.session_state["last_metadata"]
                    
            except Exception as e:
                st.error(f"Failed to communicate with backend: {e}")

with tab_search:
    st.markdown("### 🔍 Raw Chunk Search")
    st.markdown("Use this tab to debug hybrid search. It queries the Vector DB and full-text index directly, bypassing the LLM.")
    
    search_query = st.text_input("Enter keywords to find matching chunks", disabled=len(docs) == 0)
    if search_query:
        with st.spinner("Searching..."):
            try:
                res = requests.get(
                    f"{BACKEND_URL}/search", 
                    params={"session_id": st.session_state.session_id, "query": search_query}
                )
                if res.ok:
                    data = res.json()
                    chunks = data.get("chunks", [])
                    if chunks:
                        st.success(f"Found {len(chunks)} relevant chunks")
                        for i, c in enumerate(chunks, 1):
                            st.markdown(f"#### {i}. {c['source_filename']} ")
                            st.markdown(f"**Section:** {c.get('section_title', 'N/A')} | **Similarity:** `{c['similarity']:.3f}`")
                            st.info(c["content"])
                    else:
                        st.warning("No matches found. Try different keywords.")
                else:
                    st.error("Search failed.")
            except Exception as e:
                 st.error(f"Connection error: {e}")