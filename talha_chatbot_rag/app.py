import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file (local) or Streamlit Secrets (cloud).")
    st.stop()

# ---------------------------
# Load FAISS Vectorstore
# ---------------------------
def load_vectorstore():
    if not os.path.exists("faiss_index"):
        st.error("⚠️ Vector index not found. Please run `ingest.py` first to build it.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# ---------------------------
# Setup Groq LLM + QA chain
# ---------------------------
llm = ChatGroq(
    temperature=0,
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"   # ✅ locked to 70B model
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🤖 Ask about Muhammad Talha Nasir")
st.caption("Answers grounded in my CV & projects. If it's not in my docs, I'll say I don't know.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
prompt = st.text_input("Ask a question:")

if prompt:
    try:
        response = qa_chain({"question": prompt})
        answer = response["answer"]

        # Display user + bot messages
        st.session_state.chat_history.append(("You", prompt))
        st.session_state.chat_history.append(("Bot", answer))

        for role, msg in st.session_state.chat_history:
            st.write(f"**{role}:** {msg}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
