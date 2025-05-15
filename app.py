import streamlit as st
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.docstore.document import Document

# Configura tu token de Hugging Face
HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("Por favor configura HUGGINGFACEHUB_API_TOKEN en Streamlit secrets.")
    st.stop()

@st.cache_resource(show_spinner=True)
def load_embeddings():
    # Modelo para embeddings (puedes cambiar a otro)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_faiss(texts, embeddings):
    # Convierte textos en Documentos
    docs = [Document(page_content=t) for t in texts]
    # Crea índice FAISS
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource(show_spinner=True)
def load_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature":0, "max_length":256},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

def main():
    st.title("Chatbot con HuggingFace + FAISS")

    uploaded_file = st.file_uploader("Sube tu archivo JSONL con conversaciones", type=["jsonl"])
    if not uploaded_file:
        st.info("Sube un archivo JSONL para empezar.")
        return

    # Carga textos del JSONL
    texts = []
    for line in uploaded_file:
        data = json.loads(line)
        texts.append(data["text"])

    embeddings = load_embeddings()
    vectordb = load_faiss(texts, embeddings)
    llm = load_llm()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = st.text_input("Haz una pregunta:")
    if query:
        with st.spinner("Buscando respuesta..."):
            result = qa_chain.run(query)
        st.markdown(f"**Respuesta:** {result}")

if __name__ == "__main__":
    main()