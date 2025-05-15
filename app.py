import streamlit as st
import os
import json

from langchain.schema import Document 
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Configurar API Fireworks desde Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["FIREWORKS_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.fireworks.ai/inference/v1"

# --- Cargar documentos JSONL ---
docs = []
for filename in os.listdir("docs"):
    if filename.endswith(".jsonl"):
        with open(os.path.join("docs", filename), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    content = obj.get("content", "")
                    if content:
                        docs.append(Document(page_content=content))
                except json.JSONDecodeError:
                    continue

# --- Crear vector store ---
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="db")
vectordb.persist()

# --- Preparar modelo LLM ---
llm = ChatOpenAI(model="accounts/fireworks/models/mixtral-8x7b-instruct")

# --- Cadena QA con recuperación ---
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())

# --- Interfaz Streamlit ---
st.set_page_config(page_title="Chatbot con JSONL", page_icon="🤖")
st.title("🤖 Chatbot con tus archivos .jsonl")

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