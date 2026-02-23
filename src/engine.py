import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_rag_chain():
    db_path = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
    
    vector_db = Chroma(
        persist_directory=db_path, 
        embedding_function=OpenAIEmbeddings()
    )
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return chain