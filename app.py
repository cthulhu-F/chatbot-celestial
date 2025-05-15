import streamlit as st
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from transformers import pipeline

@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_faiss(texts, _embeddings):
    docs = [Document(page_content=t) for t in texts]
    return FAISS.from_documents(docs, _embeddings)

@st.cache_resource(show_spinner=True)
def load_pipeline():
    return pipeline("text2text-generation", model="mrm8488/bert2bert_shared-spanish-finetuned-squad2-es")

def main():
    st.title("🤖 Chatbot bíblico con HuggingFace y FAISS")

    uploaded_file = st.file_uploader("Sube tu archivo JSONL con conversaciones", type=["jsonl"])
    if not uploaded_file:
        st.info("Sube un archivo para comenzar.")
        return

    texts = []
    for line in uploaded_file:
        try:
            data = json.loads(line)
            texts.append(data["text"])
        except Exception as e:
            st.error(f"Error al leer línea: {e}")
            return

    embeddings = load_embeddings()
    vectordb = load_faiss(texts, embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    generator = load_pipeline()

    query = st.text_input("Haz una pregunta:")
    if query:
        with st.spinner("Buscando respuesta..."):
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"
            result = generator(prompt, max_length=256, do_sample=False)
            answer = result[0]["generated_text"]

        st.markdown(f"**Respuesta:** {answer}")

if __name__ == "__main__":
    main()