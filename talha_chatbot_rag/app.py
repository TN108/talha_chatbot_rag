import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# -------------------------------
# 1. Load environment variables
# -------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Please set it in Streamlit → App settings → Secrets.")
    st.stop()

# -------------------------------
# 2. Define paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "faiss_index"

# -------------------------------
# 3. Load FAISS vectorstore
# -------------------------------
def load_vectorstore():
    if not INDEX_DIR.exists():
        st.error("⚠️ Vector index not found. Please run `ingest.py` locally and push the `faiss_index/` folder to GitHub.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Failed to load FAISS index: {e}")
        st.stop()

vectorstore = load_vectorstore()

# -------------------------------
# 4. Initialize Groq LLM
# -------------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",   # ✅ Use the latest recommended Groq model
    temperature=0
)

# -------------------------------
# 5. Setup Conversational QA Chain
# -------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    verbose=True,
)

# -------------------------------
# 6. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Talha Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Ask about Muhammad Talha Nasir")
st.caption("Answers grounded in my CV & projects. If it's not in my docs, I'll say I don't know.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything about Talha..."):
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from QA chain
    with st.chat_message("assistant"):
        try:
            response = qa_chain({"question": prompt})
            answer = response["answer"]
        except Exception as e:
            answer = f"❌ Error while generating response: {e}"

        st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
