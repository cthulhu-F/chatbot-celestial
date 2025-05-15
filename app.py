import streamlit as st
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from huggingface_hub import InferenceApi

HUGGINGFACEHUB_API_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("Por favor configura HUGGINGFACEHUB_API_TOKEN en Streamlit secrets.")
    st.stop()

@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_faiss(texts, _embeddings):
    docs = [Document(page_content=t) for t in texts]
    return FAISS.from_documents(docs, _embeddings)

@st.cache_resource(show_spinner=True)
def load_inference_api():
    return InferenceApi(repo_id="bigscience/bloomz-560m", token=HUGGINGFACEHUB_API_TOKEN)

def main():
    st.title("Chatbot con HuggingFace + FAISS")

    uploaded_file = st.file_uploader("Sube tu archivo JSONL con conversaciones", type=["jsonl"])
    if not uploaded_file:
        st.info("Sube un archivo JSONL para empezar.")
        return

    texts = []
    for line in uploaded_file:
        data = json.loads(line)
        texts.append(data["text"])

    embeddings = load_embeddings()
    vectordb = load_faiss(texts, embeddings)
    inference_api = load_inference_api()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":3})

    query = st.text_input("Haz una pregunta:")
    if query:
        with st.spinner("Buscando respuesta..."):
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"

            response = inference_api(inputs=prompt, raw_response=True)
            response_json = response.json()

            # Mostrar la respuesta completa para debug
            st.write("Respuesta raw API:", response_json)

            # Intentamos extraer la respuesta con varios métodos:
            answer = None

            # 1. Intenta clave 'generated_text'
            if isinstance(response_json, dict):
                answer = response_json.get('generated_text', None)

            # 2. Si es lista, toma el primer elemento con 'generated_text'
            if not answer and isinstance(response_json, list) and len(response_json) > 0:
                first_item = response_json[0]
                if isinstance(first_item, dict):
                    answer = first_item.get('generated_text', None)
                else:
                    answer = str(first_item)

            # 3. Si no tenemos respuesta, conviértelo a string por si es texto plano
            if not answer:
                answer = str(response_json) or "No se obtuvo respuesta"

        st.markdown(f"**Respuesta:** {answer}")

if __name__ == "__main__":
    main()