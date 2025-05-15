import streamlit as st
import os
import json
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer

# Embedder personalizado
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# Variables de entorno
os.environ["OPENAI_API_KEY"] = st.secrets["FIREWORKS_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.fireworks.ai/inference/v1"

# Cargar datos
docs = []
for filename in os.listdir("docs"):
    if filename.endswith(".jsonl"):
        with open(os.path.join("docs", filename), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    docs.append(Document(page_content=text))

texts = [doc.page_content for doc in docs]
embeddings = SentenceTransformerEmbeddings()

# Persistir DB
vectordb = Chroma.from_texts(texts, embeddings, persist_directory="db")
vectordb.persist()

# Chat
llm = ChatOpenAI(model="accounts/fireworks/models/mixtral-8x7b-instruct")
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())

# Interfaz
st.set_page_config(page_title="Chat con Salmos", page_icon="📖")
st.title("📖 Chat Espiritual con los Salmos")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("¿Qué te inquieta hoy?")

if query:
    result = qa_chain({
        "question": query,
        "chat_history": st.session_state.chat_history
    })
    st.session_state.chat_history.append((query, result["answer"]))
    st.markdown(f"**Respuesta:** {result['answer']}")