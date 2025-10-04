# connectMemoryWithLLM.py (Updated)

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# -------------------------------
# Step 0 -> Load .env
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# -------------------------------
# Step 1 -> Setup OpenAI LLM
# -------------------------------
def load_llm():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        openai_api_key=OPENAI_API_KEY
    )
    return llm

# -------------------------------
# Step 2 -> Connect LLM with FAISS
# -------------------------------
CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. 
Do not provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please."""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Load FAISS database with OpenAI embeddings
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# -------------------------------
# Step 3 -> Create QA chain
# -------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# -------------------------------
# Step 4 -> Run query
# -------------------------------
user_query = input("Write your query here: ")
response = qa_chain.invoke({'query': user_query})

print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
