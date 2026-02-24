import os
from dotenv import load_dotenv
# --- CHANGED: Use Ollama ---
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_rag_chain():
    db_path = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
    
    # --- CHANGED: Use local embeddings function ---
    vector_db = Chroma(
        persist_directory=db_path, 
        embedding_function=OllamaEmbeddings(model="llama3.2")
    )
    
    # --- CHANGED: Use local LLM ---
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return chain