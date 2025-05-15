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
    return pipeline("text2text-generation", model="mrm8488/spanish-t5-small-sqac-for-qa")

def main():
    st.title("🤖 Chatbot Bíblico (Español) con HuggingFace + FAISS")

    
    #uploaded_file = st.file_uploader("Sube tu archivo JSONL con versículos/consejos", type=["jsonl"])
    #if not uploaded_file:
    #    st.info("Sube un archivo JSONL para comenzar.")
    #    return
    #texts = []
    #for line in uploaded_file:
    #    try:
    #        data = json.loads(line)
    #        texts.append(data["text"])
    #    except Exception as e:
    #        st.error(f"Error en una línea del archivo: {e}")
    #        return

    jsonl_file_path = "conversaciones_autoayuda_cristiana_naturales.jsonl"

    # Verificar si el archivo existe
    if not os.path.exists(jsonl_file_path):
        st.error(f"No se encontró el archivo JSONL en la ruta: {jsonl_file_path}")
        return

    # Leer el archivo JSONL
    texts = []
    try:
        with open(jsonl_file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    if "text" in data:
                        texts.append(data["text"])
                    else:
                        st.warning(f"Línea omitida, no contiene campo 'text': {line}")
                except json.JSONDecodeError as e:
                    st.error(f"Error al parsear una línea del archivo JSONL: {e} - Línea: {line}")
                    continue  # Continuar con la siguiente línea
                except Exception as e:
                    st.error(f"Error inesperado en una línea: {e} - Línea: {line}")
                    continue
        if not texts:
            st.error("No se pudieron cargar diálogos del archivo JSONL.")
            return
        st.success(f"Archivo JSONL cargado correctamente. Se cargaron {len(texts)} diálogos.")
    except Exception as e:
        st.error(f"Error al leer el archivo JSONL: {e}")
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

            # Prompt en español, claro para el modelo
            prompt = (
                f"Contexto:\n{context}\n\n"
                f"Pregunta: {query}\n\n"
                f"Respuesta en español:"
            )

            result = generator(prompt, max_length=256, do_sample=False)
            answer = result[0]["generated_text"].strip()

        st.markdown("**Respuesta:**")
        st.write(answer if answer else "No se pudo generar una respuesta.")

if __name__ == "__main__":
    main()