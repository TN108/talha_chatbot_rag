import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Folders
DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

def load_docs():
    docs = []
    
    # ✅ Updated CV path (with spaces in filename)
    cv = DATA_DIR / "MUHAMMAD TALHA NASIR  CV.pdf"
    if cv.exists():
        docs.extend(PyPDFLoader(str(cv)).load())
    else:
        print(f"⚠️ CV not found at {cv}. Please check the file name.")

    # Load .md and .txt project files if available
    for p in DATA_DIR.glob("*.md"):
        docs.extend(TextLoader(str(p), encoding="utf-8").load())
    for p in DATA_DIR.glob("*.txt"):
        docs.extend(TextLoader(str(p), encoding="utf-8").load())

    if not docs:
        raise FileNotFoundError("No documents found in data/. Add your CV and projects.md")

    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs)

def build_index():
    print(" Loading documents...")
    docs = load_docs()
    print(f"Loaded {len(docs)} documents.")
    
    chunks = chunk_docs(docs)
    print(f" Created {len(chunks)} chunks.")

    print("⚙️ Embedding & building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(INDEX_DIR))
    print(" Index saved to 'index/' ")

if __name__ == "__main__":
    build_index()
