import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Talha Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Ask about Muhammad Talha Nasir")
st.caption("Answers grounded in my CV & projects. If it's not in my docs, I'll say I don't know.")

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Missing GROQ_API_KEY. Set it in environment or Streamlit Secrets.")
    st.stop()

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1, groq_api_key=api_key)

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
except Exception:
    st.error("Vector index not found. Run `python ingest.py` first.")
    st.stop()

retriever = db.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "doc")
        lines.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n".join(lines)

SYSTEM_PROMPT = """
You are an assistant for answering questions about Muhammad Talha Nasir.
Use ONLY the provided context to answer.
If the answer is not explicitly in the context, say: "I don't know based on my documents."
Be concise and professional. Include short sources like [1], [2].
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nContext:\n{context}"),
    ("human", "{question}")
])

chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
} | prompt | llm | StrOutputParser()

if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.text_input("Ask me about my skills, projects, or education:")
ask = st.button("Ask", type="primary")

if ask and user_q.strip():
    with st.spinner("Thinking..."):
        answer = chain.invoke(user_q.strip())
    st.session_state.history.append((user_q.strip(), answer))

for u, a in reversed(st.session_state.history[-8:]):
    with st.chat_message("user"):
        st.write(u)
    with st.chat_message("assistant"):
        st.write(a)

st.divider()
st.caption("Tip: Ask things like 'What is the Job Application Assistant project?' or 'What is Talha's education?'")
