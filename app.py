import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="SystemVerilog RAG Chatbot (Chroma, optimized)", layout="wide")
st.title("🤖 SystemVerilog Documentation Chatbot")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

@st.cache_data
def load_text_file(path="pdf_extracted_text.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

text_data = load_text_file()

@st.cache_resource
def get_text_chunks():
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_text(text_data)

docs = get_text_chunks()

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="intfloat/e5-base")

emb = get_embedding_model()

@st.cache_resource
def get_chroma():
    persist_dir = "chroma.db"
    db = Chroma(collection_name="verilog", embedding_function=emb, persist_directory=persist_dir)

    count = 0
    try:
        count = db._collection.count()
    except Exception:
        try:
            info = db.get()
            count = len(info.get("documents", []))
        except Exception:
            count = 0

    if count == 0:
        db.add_texts(docs)
        try:
            db.persist()
        except Exception:
            pass
    return db

db = get_chroma()

chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.25)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "max_history" not in st.session_state:
    st.session_state.max_history = 10

col1, col2 = st.columns([3, 1])

with col2:
    if st.button("Rebuild Chroma DB (force)"):
        try:
            db._collection.reset()
        except Exception:
            try:
                db.delete_collection()
            except Exception:
                pass
        db.add_texts(docs)
        try:
            db.persist()
            st.success("Chroma DB rebuilt and persisted.")
        except Exception:
            st.warning("Chroma rebuilt but persist failed (check permissions).")

with col1:
    user_query = st.chat_input("Ask a SystemVerilog question (answers strictly from the document)")

if user_query:
    retriever_docs = db.similarity_search(user_query, k=5)
    context = "\n\n".join(getattr(d, "page_content", str(d)) for d in retriever_docs)

    prompt = f"""
You are an expert SystemVerilog Verification Engineer, technical author, and educator.

You MUST use ONLY the retrieved context for all technical content.

-----------------------------------------------------
Retrieved Context:
{context}
-----------------------------------------------------

User Query:
{user_query}

If the answer cannot be found in the retrieved context, respond exactly:
"Insufficient document context to answer."
"""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_query),
    ]

    try:
        response = chat.invoke(messages)
        assistant_text = response.content
    except Exception as e:
        assistant_text = "Error invoking chat model."

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

    if len(st.session_state.chat_history) > st.session_state.max_history * 2:
        st.session_state.chat_history = st.session_state.chat_history[-st.session_state.max_history * 2 :]

for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role):
        st.markdown(content)
