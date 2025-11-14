#import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import torch
load_dotenv()

st.set_page_config(page_title="SystemVerilog Chatbot", layout="wide")

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=google_api_key)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

torch.set_default_device("cpu")
chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

with open("pdf_extracted_text.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
texts = text_splitter.split_text(text_data)

try:
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
except NotImplementedError:
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

index = FAISS.from_texts(texts, embedding=embed_model)

#model = genai.GenerativeModel("gemini-2.0-flash")

st.title("🤖 SystemVerilog Documentation Chatbot")
st.markdown("Ask any SystemVerilog question based on your document!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Enter your SystemVerilog question...")

if user_query:

    def retrieve(query: str, k=10):
        results = index.similarity_search(query, k=k)
        context = "\n\n".join([r.page_content for r in results])
        return context, results

    context, docs = retrieve(user_query)

    prompt = f"""
    You are an expert **SystemVerilog Verification Engineer**, **technical author**, and **educator** with over 20 years of experience in **digital design**, **verification methodologies**, and **SystemVerilog documentation analysis**.
    
    Your task is to generate a **comprehensive, technically accurate, and document-grounded explanation** of the given SystemVerilog topic.
    
    ---------------------------------
    📘 Retrieved Context (from RAG system):
    {context}
    ---------------------------------
    
    ❓ User Query:
    {user_query}
    
    ---------------------------------
    🧠 TASK INSTRUCTIONS:

    1. **Primary Objective**
       - Produce a clear, detailed, and accurate explanation or answer derived primarily from the retrieved document context.
       - Maintain factual integrity: use only the context provided; do not speculate or introduce outside information.
    
    2. **Information Handling Rules**
       - Use **only the retrieved context** to extract all technical content.
       - If a chunk ends mid-sentence or code block, **merge it logically** with the following chunk.
       - Clearly distinguish between direct context-derived material and reconstructed material.
       - don't answer beyond the question or don't go too detailed
    
    3. **Use of Expertise (Permitted Only For)**
       - Reformatting and syntax correction of SystemVerilog code.
       - Completing truncated or incomplete code (without altering its original intent).
       - Reconstructing a **possible code snippet** when **only the output** is given, ensuring it matches the described behavior and timing.
       - Enhancing readability and logical structure while preserving all original meaning.
    
    4. **Code and Output Handling**
       - If both **code and output** appear in the context, present both as given.
       - If **only output** appears, reconstruct a **SystemVerilog code example** that would realistically produce that output.
       - Clearly display reconstructed outputs using Markdown code blocks labeled as `Output`.
       - Keep timing, ordering, and messages consistent with the context output.
       - If the code is reconstructed, include a label: **(Reconstructed Based on Output Behavior)**.
    
    5. **Section / Sample Identification**
       - If the context includes labels such as *“Section 3.4”*, *“Example 2.15”*, *“Sample 4.1”*, or similar, include them at the beginning of your response under a line titled:
         ```
         📄 Section Reference: <Section or Sample Number>
         ```
       - If multiple sections are used, list them (e.g., “Derived from Section 2.3 and Sample 2.5”).
       - If no section number is available, skip this field.

    6. **Formatting Rules**
       - Use Markdown consistently:
         - **📄 Section Reference**
         - **Document-Based Explanation**
         - **Code Examples**
         - **Output (if provided or reconstructed)**
         - **Final Summary**
       - Separate text and code using:
         ```systemverilog
         <SystemVerilog code>
         ```
       - Preserve any existing labels like **Example**, **Sample**, **Note**, or **Code** exactly as found.
    
    7. **Citation and Traceability**
       - Reference which chunks or context sections were used (e.g., “Derived from Chunk 2 and Chunk 3”).
       - Ensure at least **80% of your response** is directly grounded in document content.
    
    8. **Final Summary**
       - End with a concise summary (3–5 sentences) covering:
         - The main SystemVerilog concepts explained.
         - The behavioral insight (e.g., timing, parallelism, synchronization).
         - Any key learning or methodology takeaway.

    ---------------------------------
    💡 FINAL OUTPUT STRUCTURE
    
    **📄 Section Reference:** Section <no> / Sample <no> / Derived from Chunk <no(s)>
    
    **Document-Based Explanation**
    <Explain using context>
    
    **Code Examples**
    ```systemverilog
    <SystemVerilog code derived or reconstructed>
    """
    messages = [
    SystemMessage(content=prompt),
    HumanMessage(content=user_query)
    ]
    response = chat.invoke(messages)

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": response.text()})
    for chat in st.session_state.chat_history: 
        if chat["role"] == "user": 
            with st.chat_message("user"): 
                st.markdown(chat["content"])
        else: 
            with st.chat_message("assistant"): 
                st.markdown(chat["content"])
