# RAG_System
This project will build a system that allows a user to upload 100+ PDFs (like textbooks or legal docs) and "chat" with them.


# 📚 Local RAG Knowledge Base (RAG_System)

This project is a high-performance **Retrieval-Augmented Generation (RAG)** system designed to handle large-scale document analysis. It allows users to upload **100+ PDFs** (such as textbooks, legal docs, or research papers) and "chat" with them using local AI.

By leveraging **Ollama**, this system ensures 100% data privacy and zero API costs, running entirely on your local machine's hardware.



## 🚀 Key Features
- **Scalable Document Handling:** Engineered to process and index massive PDF libraries (100+ files).
- **100% Local Execution:** No data ever leaves your machine. Powered by Ollama.
- **Smart Retrieval:** Uses **ChromaDB** for high-speed semantic search and document retrieval.
- **Interactive UI:** Clean, responsive chat interface built with **Streamlit**.
- **Context-Aware Answers:** Strictly grounds AI responses in your uploaded data to prevent hallucinations.

## 🛠️ Tech Stack
- **LLM:** [Llama 3.2](https://ollama.com/library/llama3.2) (via Ollama)
- **Embeddings:** Ollama Embeddings (`llama3.2`)
- **Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Frontend:** Streamlit

## 📁 Project Structure
```text
RAG_System/
├── src/
│   ├── ingest.py     # Processes PDFs & creates the Vector DB
│   ├── engine.py     # RAG logic and LLM configuration
│   └── app.py        # Streamlit User Interface
├── data/             # Drop your PDFs here (supports 100+ files)
├── chroma_db/        # Local vector storage (Auto-generated)
├── requirements.txt  # Project dependencies
└── .env              # (Optional) Environment variables

```
## ⚙️ Setup & Installation
### 1. Prerequisites
Python 3.12+

Ollama: Download and install Ollama

Model Setup:

Bash
ollama pull llama3.2
### 2. Installation
Bash
# Clone the repository
git clone <your-repo-url>
cd RAG_System

Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies
pip install -r requirements.txt
### 3. Usage
Prepare Data: Place your PDF files into the /data folder.

Ingest Documents: Run the following to build your local vector database:

Bash
python3 src/ingest.py
Launch App: Start the chat interface:

Bash
python3 -m streamlit run src/app.py
## 🛡️ Privacy & Security
This system is strictly offline. It does not require OpenAI keys or external cloud processing. It is the ideal solution for analyzing sensitive resumes, private research, or confidential corporate documents.

## 🤝 Acknowledgments
Inspired by the need for low-cost, high-privacy AI development.

Built using the LangChain and Ollama communities.


---

### 📝 Final Step for your GitHub
To make sure the **Installation** step works for anyone who downloads your project, you need to create the `requirements.txt` file.

Type this in your VS Code terminal:
```bash
pip freeze > requirements.txt