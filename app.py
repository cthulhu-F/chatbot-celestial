import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

# Configura API Key desde secrets
os.environ["OPENAI_API_KEY"] = st.secrets["fw_3ZbpGN7h9xfqfRCox1E1i8V2"]
os.environ["OPENAI_API_BASE"] = "https://api.fireworks.ai/inference/v1"

# Cargar documentos PDF desde carpeta docs/
loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Crear vector store
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="db")
vectordb.persist()

# Modelo LLM
llm = ChatOpenAI(model="accounts/fireworks/models/mixtral-8x7b-instruct")

# Crear cadena QA con recuperación
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())

# Interfaz Streamlit
st.title("🤖 Chatbot con tus archivos")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Haz una pregunta:")

if query:
    result = qa_chain({
        "question": query,
        "chat_history": st.session_state.chat_history
    })
    st.session_state.chat_history.append((query, result["answer"]))
    st.markdown(f"**Respuesta:** {result['answer']}")