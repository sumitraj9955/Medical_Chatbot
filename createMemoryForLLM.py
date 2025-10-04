# createMemoryForLLM.py (Updated)

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# -------------------------------
# Step 0 -> Load .env
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# -------------------------------
# Step 1 -> Load raw PDF(s)
# -------------------------------
DATA_PATH = "data/"  # Folder containing all PDFs

def load_pdf_files(data):
    loader = DirectoryLoader(
        path=data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print("Number of PDF pages loaded:", len(documents))

# -------------------------------
# Step 2 -> Create text chunks
# -------------------------------
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print("Number of text chunks created:", len(text_chunks))

# -------------------------------
# Step 3 -> Create OpenAI embeddings
# -------------------------------
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# -------------------------------
# Step 4 -> Store embeddings in FAISS
# -------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print(f"FAISS database saved at: {DB_FAISS_PATH}")
