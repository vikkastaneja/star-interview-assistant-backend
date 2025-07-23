import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from pathlib import Path

# Config
PDF_DIR = "pdfs"
INDEX_DIR = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"  # or llama2, gemma, etc.

def load_and_chunk_pdfs(pdf_dir):
    print("📥 Loading and chunking PDFs...")
    documents = []
    for file in Path(pdf_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"✅ Loaded and chunked {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def build_vector_store(chunks, index_dir):
    print("🧠 Creating FAISS vector store with embeddings...")
    embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(index_dir)
    print("✅ Vector store saved.")
    return vector_store

def load_vector_store(index_dir):
    print("📂 Loading existing FAISS vector store...")
    embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(index_dir, embedding_model)

def start_qa_loop(vector_store):
    llm = Ollama(model=LLM_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

    print("\n💬 QA system is ready! Ask your questions (type 'exit' to quit):")
    while True:
        question = input("\n❓ Question: ")
        if question.strip().lower() in ("exit", "quit"):
            print("👋 Exiting.")
            break
        answer = qa_chain.run(question)
        print(f"📘 Answer: {answer}")

def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    if not os.listdir(PDF_DIR):
        print(f"⚠️ Please add PDF files to the '{PDF_DIR}' folder before running this script.")
        return

    if not os.path.exists(f"{INDEX_DIR}/index.faiss"):
        chunks = load_and_chunk_pdfs(PDF_DIR)
        vector_store = build_vector_store(chunks, INDEX_DIR)
    else:
        vector_store = load_vector_store(INDEX_DIR)

    start_qa_loop(vector_store)

if __name__ == "__main__":
    main()
