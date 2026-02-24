import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- CHANGED: Use Ollama instead of OpenAI ---
from langchain_ollama import OllamaEmbeddings 
from langchain_community.vectorstores import Chroma

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def build_vector_db():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    db_path = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
    
    loader = PyPDFDirectoryLoader(data_path)
    docs = loader.load()
    
    if not docs:
        print("❌ No PDFs found in the /data folder!")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # --- CHANGED: Use local embeddings ---
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="llama3.2"),
        persist_directory=db_path
    )
    print(f"✅ Vector DB Created at {db_path}")

if __name__ == "__main__":
    build_vector_db()