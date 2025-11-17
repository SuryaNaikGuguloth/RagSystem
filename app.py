import streamlit as st
import os
import torch
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="SystemVerilog RAG Chatbot", layout="wide")
st.title("🤖 SystemVerilog Documentation Chatbot")

for var in ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"]:
    if var in os.environ:
        del os.environ[var]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

torch.set_default_device("cpu")

@st.cache_resource
def load_text():
    with open("pdf_extracted_text.txt", "r", encoding="utf-8") as f:
        return f.read()

text_data = load_text()

@st.cache_resource
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

docs = split_text(text_data)

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")

emb = load_embedding_model()

@st.cache_resource
def load_chroma(embeddings):
    db = Chroma(
        collection_name="verilog",
        embedding_function=embeddings,
        persist_directory="chroma.db",
    )

    # Only add if database is empty
    try:
        existing = db.get()
        if len(existing["documents"]) == 0:
            db.add_texts(docs)
    except:
        db.add_texts(docs)

    return db

data_base = load_chroma(emb)

chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_query = st.chat_input("Ask a SystemVerilog question (answers only from your document!)")

if user_query:

    retriever = data_base.similarity_search(user_query, k=5)
    context = "".join(doc.page_content for doc in retriever[:5])

    # STRICT RAG PROMPT
    prompt = f"""
You are an expert **SystemVerilog Verification Engineer**, **technical author**, and **educator** with over 20 years of experience in digital design and verification.

Your output MUST be strictly grounded in the retrieved document context.

-----------------------------------------------------
📘 Retrieved Context (from RAG system):
{context}
-----------------------------------------------------

❓ User Query:
{user_query}

-----------------------------------------------------
🧠 STRICT TASK RULES (READ CAREFULLY)

1. PRIMARY REQUIREMENT — RAG-ONLY ANSWERING
   - **Use ONLY the retrieved context** for all technical content.
   - If the answer requires information not present in the context:
     → Respond with: **"Insufficient document context to answer."**
   - DO NOT use your own knowledge, memory, or assumptions.

2. CONTENT HANDLING
   - If context chunks break sentences or code, merge them logically.
   - Only reorganize, reformat, or clarify what already exists.
   - NO new concepts, NO new definitions, NO external SystemVerilog knowledge.

3. ALLOWED USE OF EXPERTISE
   (Only for improving existing context — NOT adding new content)
   - Reformat SystemVerilog code for readability.
   - Fix syntax errors in context code.
   - Complete truncated code **only when the missing part is implied**.
   - Reconstruct minimal code to match given output (if output appears in context).
   *Never add features or logic not traceable to the context.*

4. CODE & OUTPUT RULES
   - If code + output exist in context → present both exactly.
   - If only output exists → reconstruct minimal SystemVerilog code.
   - Label reconstructed code as: **(Reconstructed Based on Output Behavior)**
   - Show outputs in fenced 
Output
 blocks.

5. SECTION / SAMPLE IDENTIFICATION
   - If context includes Section, Example, Sample labels → include them.
   - If multiple → list all.
   - If none → skip.

6. FORMAT OF FINAL ANSWER (REQUIRED)
   Your response MUST follow this structure:

   **📄 Section Reference:** Section <no> / Sample <no> / Derived from Chunk <no(s)>

   **Document-Based Explanation**
   <Strictly derived from context>

   **Code Examples**
   
systemverilog
   <Code from context or reconstructed>
"""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_query),
    ]

    response = chat.invoke(messages)

    st.session_state.chat_history.append(
        {"role": "user", "content": user_query}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response.content}
    )

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
