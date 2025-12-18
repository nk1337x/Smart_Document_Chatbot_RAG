"""
RAG Document Chatbot - ChatGPT Style Interface
Enhanced UI with proper sidebar and file upload
"""

import streamlit as st
import hashlib
import pickle
import time
from pathlib import Path

from utils.document_processor import DocumentProcessor
from utils.text_chunker import TextChunker
from utils.embedding_manager import EmbeddingManager
from utils.vector_store import VectorStore
from utils.llm_handler import LLMHandler

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════

GROQ_API_KEY = ""
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════

st.markdown("""
<style>
/* Hide Streamlit defaults */
#MainMenu, footer, header, .stDeployButton {display: none !important;}
[data-testid="stToolbar"] {display: none !important;}

/* ══════════ GLOBAL THEME ══════════ */
.stApp {
    background-color: #0f0f0f;
}

/* Remove any background from main content */
.main {
    background-color: #0f0f0f !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #0f0f0f !important;
}

/* ══════════ SIDEBAR STYLING ══════════ */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #1f1f1f;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* Sidebar text styling */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #e5e5e5;
}

.sidebar-title {
    color: #ffffff;
    font-size: 20px;
    font-weight: 600;
    padding: 0 1rem 1rem 1rem;
    border-bottom: 1px solid #1f1f1f;
    margin-bottom: 1.5rem;
}

.sidebar-section-title {
    color: #6b7280;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 0 1rem;
    margin: 1.5rem 0 0.5rem 0;
}

/* File uploader in sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background-color: transparent !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin: 0.5rem 1rem !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] label {
    color: #9ca3af !important;
    font-size: 13px !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #1f1f1f !important;
    color: #e5e5e5 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-size: 13px !important;
    margin-top: 0.5rem !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background-color: #2a2a2a !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] button[kind="secondary"] {
    background-color: #1f1f1f !important;
    color: #e5e5e5 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    width: calc(100% - 2rem) !important;
    margin: 0.5rem 1rem !important;
    padding: 0.75rem !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background-color: #2a2a2a !important;
    border-color: #3a3a3a !important;
}

/* Document status in sidebar */
.doc-status {
    background-color: rgba(45, 74, 62, 0.25);
    border: 1px solid rgba(110, 231, 183, 0.2);
    color: #6ee7b7;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: 13px;
    margin: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ══════════ MAIN CONTENT AREA ══════════ */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ══════════ CHAT WRAPPER - CENTERS EVERYTHING ══════════ */
/* Outer wrapper that centers the chat column */
.chat-wrapper {
    display: flex;
    justify-content: center;
    width: 100%;
    min-height: 60vh;
}

/* ══════════ CHAT CONTAINER - FIXED WIDTH COLUMN ══════════ */
/* Inner container with fixed max-width - prevents edge hugging */
.chat-container {
    width: 100%;
    max-width: 720px;
    padding: 2rem 2.5rem 12rem 2.5rem;
    display: flex;
    flex-direction: column;
}

/* ══════════ MESSAGE ROW - FLEX CONTAINER ══════════ */
/* Each message wrapped in a row for left/right alignment */
.message-row {
    width: 100%;
    display: flex;
    margin-bottom: 12px;
    animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* User messages - aligned to the right with gap from edge */
.message-row.user {
    justify-content: flex-end;
}

/* Assistant messages - aligned to the left with gap from edge */
.message-row.assistant {
    justify-content: flex-start;
}

/* ══════════ CHAT BUBBLES ══════════ */
.chat-bubble {
    padding: 12px 16px;
    border-radius: 14px;
    max-width: 60%;
    line-height: 1.6;
    font-size: 15px;
    word-wrap: break-word;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
    margin: 0 8px;
}

.chat-bubble.user {
    background-color: #2a2a2a;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}

.chat-bubble.assistant {
    background-color: #1f1f1f;
    color: #e5e5e5;
}

/* Markdown in bubbles */
.chat-bubble p {
    margin: 0 0 0.5rem 0;
}

.chat-bubble p:last-child {
    margin-bottom: 0;
}

.chat-bubble ul,
.chat-bubble ol {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
}

.chat-bubble li {
    margin: 0.25rem 0;
}

.chat-bubble strong {
    color: #ffffff;
    font-weight: 600;
}

.chat-bubble code {
    background-color: #0d0d0d;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-size: 14px;
    font-family: 'Consolas', 'Monaco', monospace;
}

.chat-bubble pre {
    background-color: #0d0d0d;
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
    margin: 0.75rem 0;
}

.chat-bubble pre code {
    background-color: transparent;
    padding: 0;
}

/* ══════════ THINKING INDICATOR ══════════ */
/* Thinking indicator appears like an assistant message */
.chat-bubble.thinking {
    color: #6b7280;
    font-style: italic;
}

.thinking-dots {
    display: inline-block;
    animation: blink 1.4s infinite both;
}

@keyframes blink {
    0% { opacity: 0.3; }
    20% { opacity: 1; }
    100% { opacity: 0.3; }
}

/* Typing cursor animation */
.typing-cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    background-color: #e5e5e5;
    margin-left: 2px;
    animation: cursor-blink 1s infinite;
}

@keyframes cursor-blink {
    0%, 49% { opacity: 1; }
    50%, 100% { opacity: 0; }
}

/* ══════════ INPUT BAR (BOTTOM FIXED) ══════════ */
/* Fixed at bottom but centered to match chat width */
.input-bar-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    background: linear-gradient(to top, #0f0f0f 75%, transparent);
    padding: 1.5rem;
    z-index: 1000;
    pointer-events: none;
}

.input-bar-inner {
    width: 100%;
    max-width: 720px;
    pointer-events: all;
}

/* Chat input styling - matches background */
[data-testid="stChatInputContainer"] {
    background-color: #0f0f0f !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 26px !important;
    padding: 0 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.6) !important;
}

[data-testid="stChatInput"] {
    background-color: transparent !important;
    border: none !important;
}

[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #ffffff !important;
    caret-color: #ffffff !important;
    border: none !important;
    padding: 1rem 1.25rem !important;
    font-size: 15px !important;
    resize: none !important;
    line-height: 1.5 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #6b7280 !important;
}

[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] button {
    background-color: transparent !important;
    color: #6b7280 !important;
    border: none !important;
    padding: 0.5rem !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: #9ca3af !important;
}

/* ══════════ WELCOME SCREEN ══════════ */
.welcome-wrapper {
    display: flex;
    justify-content: center;
    width: 100%;
    min-height: 65vh;
}

.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    max-width: 720px;
    text-align: center;
    padding: 2rem 1rem;
}

.welcome-icon {
    font-size: 56px;
    margin-bottom: 1.5rem;
    opacity: 0.7;
}

.welcome-title {
    color: #ffffff;
    font-size: 36px;
    font-weight: 600;
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
}

.welcome-subtitle {
    color: #6b7280;
    font-size: 16px;
    max-width: 480px;
    line-height: 1.5;
}

/* ══════════ MISC ══════════ */
.stSpinner > div {
    border-color: #4a4a4a transparent transparent transparent !important;
}

::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #0f0f0f;
}

::-webkit-scrollbar-thumb {
    background: #2a2a2a;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #3a3a3a;
}

.stToast {
    background-color: #1f1f1f !important;
    color: #e5e5e5 !important;
    border: 1px solid #2a2a2a !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ready" not in st.session_state:
    st.session_state.ready = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = None
if "llm_handler" not in st.session_state:
    st.session_state.llm_handler = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""

# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def get_hash(files):
    """Generate hash for file caching."""
    h = hashlib.md5()
    for f in files:
        f.seek(0)
        h.update(f.read())
        f.seek(0)
    return h.hexdigest()

def process_files(files):
    """Process uploaded documents."""
    processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    
    chunks = []
    for f in files:
        ext = Path(f.name).suffix.lower()
        if ext == '.pdf':
            text = processor.extract_from_pdf(f)
        elif ext == '.txt':
            text = processor.extract_from_txt(f)
        else:
            continue
        chunks.extend(chunker.chunk_text(text, metadata={"source": f.name}))
    
    if not chunks:
        return None, None
    
    texts = [c["text"] for c in chunks]
    embeddings = embedder.generate_embeddings(texts)
    
    store = VectorStore(embedding_dim=embeddings.shape[1])
    store.add_documents(chunks, embeddings)
    
    return store, embedder

def process_uploaded_files(files):
    """Process and cache uploaded files."""
    file_hash = get_hash(files)
    cache_file = CACHE_DIR / f"{file_hash}.pkl"
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        st.session_state.vector_store = data["store"]
        st.session_state.embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    else:
        store, embedder = process_files(files)
        if store is None:
            return False
        st.session_state.vector_store = store
        st.session_state.embedding_manager = embedder
        with open(cache_file, 'wb') as f:
            pickle.dump({"store": store}, f)
    
    # Initialize LLM
    st.session_state.llm_handler = LLMHandler(model_type="groq", api_key=GROQ_API_KEY)
    
    st.session_state.ready = True
    st.session_state.doc_name = ", ".join([f.name for f in files[:2]]) + ("..." if len(files) > 2 else "")
    return True

def ask(question):
    """Query the RAG system."""
    emb = st.session_state.embedding_manager.generate_embeddings([question])
    results = st.session_state.vector_store.search(emb[0], top_k=3)
    context = "\n\n".join([r['text'] for r in results])
    return st.session_state.llm_handler.generate_response(question, context)


with st.sidebar:
    # App title
    st.markdown('<div class="sidebar-title">💬 RAG Chatbot</div>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="sidebar-section-title">Upload Documents</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="file_upload",
        help="Upload documents to chat with"
    )
    
    # Process files if uploaded and not already processed
    if uploaded_files and not st.session_state.ready:
        with st.spinner("Processing documents..."):
            success = process_uploaded_files(uploaded_files)
            if success:
                st.rerun()
            else:
                st.error("Failed to process documents")
    
    # Document status indicator
    if st.session_state.ready and st.session_state.doc_name:
        st.markdown(f'<div class="doc-status">✅ {st.session_state.doc_name}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-title">Actions</div>', unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", key="clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Reset all button
    if st.button("🔄 New Session", key="reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ready = False
        st.session_state.vector_store = None
        st.session_state.embedding_manager = None
        st.session_state.llm_handler = None
        st.session_state.doc_name = ""
        st.rerun()

# ═══════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════

# Welcome screen or chat messages
if not st.session_state.messages:
    if st.session_state.ready:
        # Document loaded, ready to chat
        st.markdown("""
        <div class="welcome-wrapper">
            <div class="welcome-container">
                <div class="welcome-icon">📄</div>
                <div class="welcome-title">Document Loaded!</div>
                <div class="welcome-subtitle">Your document has been processed. Ask me anything about it.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # No document yet
        st.markdown("""
        <div class="welcome-wrapper">
            <div class="welcome-container">
                <div class="welcome-icon">💬</div>
                <div class="welcome-title">What can I help with?</div>
                <div class="welcome-subtitle">Upload a PDF or TXT document using the sidebar to get started.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Display chat messages in centered wrapper (prevents edge hugging)
    st.markdown('<div class="chat-wrapper"><div class="chat-container">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        role_class = "assistant" if msg["role"] == "assistant" else "user"
        content = msg["content"].replace('\n', '<br>')
        
        # Each message in a row: user right-aligned, assistant left-aligned
        st.markdown(f"""
        <div class="message-row {role_class}">
            <div class="chat-bubble {role_class}">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)


# Create the fixed input container (visual only)
st.markdown('<div class="input-bar-container"><div class="input-bar-inner"></div></div>', unsafe_allow_html=True)

# Chat input (Streamlit component)
if prompt := st.chat_input("Ask Anything"):
    if st.session_state.ready:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Create placeholder for dynamic updates
        response_placeholder = st.empty()
        
        # Show all messages + thinking indicator
        with response_placeholder.container():
            st.markdown('<div class="chat-wrapper"><div class="chat-container">', unsafe_allow_html=True)
            
            # Show existing messages
            for msg in st.session_state.messages:
                role_class = "assistant" if msg["role"] == "assistant" else "user"
                content = msg["content"].replace('\n', '<br>')
                st.markdown(f"""
                <div class="message-row {role_class}">
                    <div class="chat-bubble {role_class}">{content}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show thinking indicator
            st.markdown("""
            <div class="message-row assistant">
                <div class="chat-bubble assistant thinking">
                    Thinking<span class="thinking-dots">...</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Get AI response
        try:
            response = ask(prompt)
            
            # Replace thinking with typing animation
            typed_text = ""
            words = response.split()
            
            for i, word in enumerate(words):
                typed_text += word + " "
                
                # Update display with cursor
                with response_placeholder.container():
                    st.markdown('<div class="chat-wrapper"><div class="chat-container">', unsafe_allow_html=True)
                    
                    # Show existing messages
                    for msg in st.session_state.messages:
                        role_class = "assistant" if msg["role"] == "assistant" else "user"
                        content = msg["content"].replace('\n', '<br>')
                        st.markdown(f"""
                        <div class="message-row {role_class}">
                            <div class="chat-bubble {role_class}">{content}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show typing text with cursor
                    display_text = typed_text.replace('\n', '<br>')
                    st.markdown(f"""
                    <div class="message-row assistant">
                        <div class="chat-bubble assistant">{display_text}<span class="typing-cursor"></span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Typing speed (adjust for faster/slower)
                time.sleep(0.03)
            
            # Final display without cursor
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        response_placeholder.empty()
        st.rerun()
    else:
        st.toast("⚠️ Please upload a document first", icon="📄")
