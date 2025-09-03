import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_data():
    # Path to your CV
    pdf_path = "data/MUHAMMAD TALHA NASIR  CV.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ Could not find file: {pdf_path}")

    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    print("✂️ Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    print("🧠 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index
    print("📦 Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index in folder "faiss_index"
    vectorstore.save_local("faiss_index")

    print("✅ Ingestion complete! Vector index saved to 'faiss_index/'")

if __name__ == "__main__":
    ingest_data()
