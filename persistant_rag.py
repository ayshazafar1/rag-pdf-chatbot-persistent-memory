# ==========================
# Imports
# ==========================

import os
import tempfile
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==========================
# Page Setup
# ==========================

st.set_page_config(
    page_title="RAG PDF Assistant",
    layout="wide",
    page_icon="📚"
)

load_dotenv()

st.title("📚 RAG PDF Chat Assistant")
st.caption("Ask questions from your uploaded PDFs using AI")


# ==========================
# UI Styling
# ==========================

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

h1,h2,h3{
color:#e2e8f0;
}

[data-testid="stChatMessage"]{
border-radius:15px;
padding:12px;
margin-bottom:10px;
}

[data-testid="stChatMessage"]:nth-child(odd){
background-color:#1e293b;
}

[data-testid="stChatMessage"]:nth-child(even){
background-color:#334155;
}

.stTextInput input{
background-color:#1e293b;
color:white;
border-radius:8px;
}

</style>
""", unsafe_allow_html=True)


# ==========================
# Memory Folder
# ==========================

MEMORY_DIR = "chat_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)


# ==========================
# Sidebar
# ==========================

with st.sidebar:

    st.header("⚙️ Configuration")

    api_key_input = st.text_input("Groq API Key", type="password")

    session_id = st.text_input("Session ID", value="default_session")

    st.markdown("---")

    if st.button("Clear Chat Memory"):

        file_path = os.path.join(MEMORY_DIR, f"{session_id}.pkl")

        if os.path.exists(file_path):
            os.remove(file_path)

        st.success("Chat memory cleared")

    st.markdown("---")
    st.caption("Upload PDFs → Ask questions → Get answers")


api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("Please enter your Groq API key")
    st.stop()


# ==========================
# Embeddings + LLM
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)


# ==========================
# Upload PDFs
# ==========================

uploaded_files = st.file_uploader(
    "📂 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin")
    st.stop()


# ==========================
# Load Documents
# ==========================

all_docs = []
tmp_paths = []

for pdf in uploaded_files:

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.getvalue())
    tmp.close()

    tmp_paths.append(tmp.name)

    loader = PyPDFLoader(tmp.name)
    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = pdf.name

    all_docs.extend(docs)

st.success(f"Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")

for p in tmp_paths:
    try:
        os.unlink(p)
    except:
        pass


# ==========================
# Text Chunking
# ==========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

splits = text_splitter.split_documents(all_docs)


# ==========================
# Vector Database
# ==========================

INDEX_DIR = "chroma_index"

if "vectorstore" not in st.session_state:

    if os.path.exists(INDEX_DIR):

        vectorstore = Chroma(
            persist_directory=INDEX_DIR,
            embedding_function=embeddings
        )

    else:

        vectorstore = Chroma.from_documents(
            splits,
            embeddings,
            persist_directory=INDEX_DIR
        )

    st.session_state.vectorstore = vectorstore

vectorstore = st.session_state.vectorstore

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# ==========================
# Helper Function
# ==========================

def join_docs(docs, max_chars=7000):

    chunks = []
    total = 0

    for d in docs:

        text = d.page_content

        if total + len(text) > max_chars:
            break

        chunks.append(text)
        total += len(text)

    return "\n\n---\n\n".join(chunks)


# ==========================
# Prompts
# ==========================

rewrite_prompt = ChatPromptTemplate.from_messages([

    ("system",
     "Rewrite the user's latest question into a standalone search query using chat history. "
     "Return only the rewritten query."),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")
])


qa_prompt = ChatPromptTemplate.from_messages([

    ("system",
     "You are a STRICT RAG assistant.\n"
     "Answer ONLY using the provided context.\n"
     "If the answer is not present reply exactly:\n"
     "'Out of scope - not found in provided documents.'\n\n"
     "Context:\n{context}"
     ),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")
])


# ==========================
# Chat Memory Functions
# ==========================

def get_history(session_id):

    path = os.path.join(MEMORY_DIR, f"{session_id}.pkl")

    if os.path.exists(path):

        with open(path, "rb") as f:
            history = pickle.load(f)

    else:
        history = ChatMessageHistory()

    return history


def save_history(session_id, history):

    path = os.path.join(MEMORY_DIR, f"{session_id}.pkl")

    with open(path, "wb") as f:
        pickle.dump(history, f)


history = get_history(session_id)


# ==========================
# Display Previous Messages
# ==========================

for msg in history.messages:

    if msg.type == "human":
        st.chat_message("user").write(msg.content)

    else:
        st.chat_message("assistant").write(msg.content)


# ==========================
# Chat Input
# ==========================

user_q = st.chat_input("Ask a question about the PDFs...")


# ==========================
# Chat Logic
# ==========================

if user_q:

    rewrite_msgs = rewrite_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )

    standalone_q = llm.invoke(rewrite_msgs).content.strip()

    docs = retriever.invoke(standalone_q)

    if not docs:

        answer = "Out of scope - not found in provided documents."

    else:

        context = join_docs(docs)

        qa_msgs = qa_prompt.format_messages(
            chat_history=history.messages,
            input=user_q,
            context=context
        )

        answer = llm.invoke(qa_msgs).content


    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)

    save_history(session_id, history)


    # Debug section

    with st.expander("Debug Information"):

        st.write("Standalone Query")
        st.code(standalone_q)

        st.write(f"Retrieved {len(docs)} chunks")

        for i, doc in enumerate(docs, 1):

            st.markdown(
                f"**{i}. {doc.metadata.get('source_file','Unknown')} "
                f"(p{doc.metadata.get('page','?')})**"
            )

            st.write(doc.page_content[:400])
