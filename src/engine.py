import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# --- CHANGED: Using modern LangChain LCEL imports ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_rag_chain():
    db_path = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
    
    vector_db = Chroma(
        persist_directory=db_path, 
        embedding_function=OllamaEmbeddings(model="llama3.1:8b")
    )
    
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    
    
    # --- CHANGED: Bumped k to 5 for better context gathering ---
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    # --- CHANGED: Added a clear system prompt for the AI ---
    system_prompt = (
        "You are an AI assistant for answering questions based on the provided context. "
        "Use only the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer concise and grounded in the text.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # --- CHANGED: Building the chain the modern way ---
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return chain