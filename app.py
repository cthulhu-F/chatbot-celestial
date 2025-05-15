import streamlit as st
import os
import json
from typing import List
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer

# Wrapper simple para usar SentenceTransformer como embeddings en LangChain
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# Configurar API de Fireworks como si fuera OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["FIREWORKS_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.fireworks.ai/inference/v1"

# Cargar documentos desde JSONL
docs = []
for filename in os.listdir("docs"):
    if filename.endswith(".jsonl"):
        with open(os.path.join("docs", filename), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    content = obj.get("text", "")
                    if content:
                        docs.append(Document(page_content=content))
                except json.JSONDecodeError:
                    continue

# Crear embeddings y base vectorial con FAISS
embeddings = SentenceTransformerEmbeddings()
texts = [doc.page_content for doc in docs]
vectordb = FAISS.from_texts(texts, embeddings)

# LLM usando Fireworks
llm = ChatOpenAI(model="accounts/fireworks/models/mixtral-8x7b-instruct")

# Cadena de preguntas y respuestas con historial
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())

# Interfaz en Streamlit
st.set_page_config(page_title="Chatbot Celestial", page_icon="💬")
st.title("💬 Chatbot Espiritual con tus archivos JSONL")

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