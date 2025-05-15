import streamlit as st
import os
import json
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.base import Embeddings
from typing import List
from sentence_transformers import SentenceTransformer

# Clase embeddings compatible con LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# Configurar API key Fireworks desde secrets
os.environ["OPENAI_API_KEY"] = st.secrets["FIREWORKS_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.fireworks.ai/inference/v1"

# Cargar documentos JSONL desde carpeta docs/
docs = []
for filename in os.listdir("docs"):
    if filename.endswith(".jsonl"):
        with open(os.path.join("docs", filename), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        docs.append(Document(page_content=text))
                except json.JSONDecodeError:
                    continue

texts = [doc.page_content for doc in docs]

# Crear embeddings y vectorstore
embeddings = SentenceTransformerEmbeddings()
vectordb = Chroma.from_texts(texts, embeddings, persist_directory="db")
vectordb.persist()

# Crear modelo Fireworks y cadena de QA
llm = ChatOpenAI(model="accounts/fireworks/models/mixtral-8x7b-instruct")
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())

# Interfaz Streamlit
st.set_page_config(page_title="Chatbot JSONL", page_icon="🤖")
st.title("🤖 Chatbot con archivos .jsonl")

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