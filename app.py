import streamlit as st
import os
import json
from typing import List

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import HuggingFaceChat
from langchain.chains import ConversationalRetrievalChain

# Token de HuggingFace para API
HF_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN

# Carga los documentos JSONL
def load_docs(path="docs"):
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".jsonl"):
            with open(os.path.join(path, filename), encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        docs.append(Document(page_content=text))
    return docs

docs = load_docs()

# Crea embeddings usando un modelo de sentence-transformers compatible
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Extrae textos
texts = [doc.page_content for doc in docs]

# Crea vectorstore FAISS
vectordb = FAISS.from_texts(texts, embeddings)

# Define el modelo LLM HuggingFace Chat (usa un modelo instruct)
llm = HuggingFaceChat(model_name="tiiuae/falcon-7b-instruct", huggingfacehub_api_token=HF_API_TOKEN)

# Cadena de consulta conversacional con recuperación
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())

# Streamlit UI
st.title("🤖 Chatbot con HuggingFace API + FAISS")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Haz una pregunta:")

if query:
    result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))
    st.markdown(f"**Respuesta:** {result['answer']}")
