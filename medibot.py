import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# -------------------------------
# Load FAISS database
# -------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# -------------------------------
# Create prompt template
# -------------------------------
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# -------------------------------
# Load LLM
# -------------------------------
def load_llm():
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        openai_api_key=OPENAI_API_KEY
    )

# -------------------------------
# Main Streamlit app
# -------------------------------
def main():
    st.title("Ask MediBot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input("Give your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. 
Do not provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please."""

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Format source documents nicely
            source_text = "\n\n".join([
                f"File: {doc.metadata.get('source','Unknown')}\n{doc.page_content.strip()[:500]}..."
                for doc in source_documents
            ])

            result_to_show = f"{result}\n\nSource Docs:\n{source_text}"

        except Exception as e:
            st.error(f"Error: {str(e)}")
            result_to_show = "Something went wrong!"

        # Display bot response
        st.chat_message('assistant').markdown(result_to_show)
        st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

if __name__ == "__main__":
    main()
