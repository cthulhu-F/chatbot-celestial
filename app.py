import streamlit as st
import json
import os
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
    # Usar un modelo más robusto si es posible; mantener el actual si hay limitaciones
    return pipeline("text2text-generation", model="mrm8488/spanish-t5-small-sqac-for-qa")

# Función para detectar emociones o temas en la consulta
def detectar_emocion(query):
    query = query.lower()
    emociones = {
        "ansiedad": ["ansioso", "nervioso", "preocupado", "miedo"],
        "tristeza": ["triste", "deprimido", "desanimado", "llorar"],
        "soledad": ["solo", "abandonado", "nadie"],
        "estrés": ["estresado", "abrumado", "cansado"],
        "duda": ["dudo", "fe", "confundido"]
    }
    for emocion, palabras in emociones.items():
        if any(palabra in query for palabra in palabras):
            return emocion
    return "general"

def main():
    st.title("🤖 Chatbot Bíblico (Español) con HuggingFace + FAISS")

    jsonl_file_path = "docs/conversaciones_autoayuda_cristiana_unido.jsonl"

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
                    continue
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
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Aumentar k para más contexto

    generator = load_pipeline()

    # Añadir campo para personalización (opcional)
    st.markdown("**Personaliza tu experiencia**")
    user_name = st.text_input("¿Cuál es tu nombre? (Opcional)", "")
    user_context = st.text_area("Describe brevemente tu situación (Opcional)", "")

    # Mantener historial de conversación en la sesión
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Haz una pregunta o comparte cómo te sientes:")
    if query:
        with st.spinner("Buscando respuesta..."):
            # Detectar emoción para personalizar
            emocion = detectar_emocion(query)
            
            # Obtener documentos relevantes
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])

            # Prompt mejorado con instrucciones claras y few-shot
            prompt = (
                "Eres un chatbot bíblico con un enfoque cristiano protestante. Tu objetivo es ofrecer respuestas empáticas, personalizadas y basadas en la Biblia para ayudar al usuario con sus problemas o emociones. "
                "Usa un tono cálido, pastoral y alentador. Reconoce la emoción del usuario, incluye una cita bíblica relevante, y ofrece un consejo práctico. "
                "Si el usuario proporciona su nombre o contexto, incorpóralo en la respuesta de manera natural. "
                "Evita respuestas genéricas; adapta tu respuesta a la consulta específica.\n\n"
                "Ejemplo:\n"
                "Consulta: Me siento muy ansioso por mi trabajo.\n"
                "Respuesta: Entiendo lo abrumador que puede ser sentirse ansioso por el trabajo. La Biblia nos recuerda en Filipenses 4:6-7: 'Por nada estéis afanosos, sino sean conocidas vuestras peticiones delante de Dios en toda oración y ruego'. Te animo a tomar un momento para orar y entregar tus preocupaciones a Dios. Él promete darte paz. Quizás puedas escribir lo que te preocupa y orar específicamente por ello.\n\n"
                f"Contexto de diálogos relevantes:\n{context}\n\n"
                f"Nombre del usuario: {user_name if user_name else 'No proporcionado'}\n"
                f"Contexto adicional: {user_context if user_context else 'No proporcionado'}\n"
                f"Emoción detectada: {emocion}\n"
                f"Pregunta: {query}\n\n"
                "Respuesta en español:"
            )

            # Generar respuesta con parámetros ajustados
            result = generator(prompt, max_length=300, do_sample=True, temperature=0.7, top_p=0.9)
            answer = result[0]["generated_text"].strip()

            # Limpiar la respuesta (eliminar el prefijo "Respuesta en español:" si aparece)
            if answer.startswith("Respuesta en español:"):
                answer = answer[len("Respuesta en español:"):].strip()

            # Guardar en historial
            st.session_state.chat_history.append({"query": query, "answer": answer})

        st.markdown("**Respuesta:**")
        st.write(answer if answer else "No se pudo generar una respuesta.")

        # Mostrar historial (opcional)
        if st.session_state.chat_history:
            st.markdown("**Historial de conversación:**")
            for i, entry in enumerate(st.session_state.chat_history[-3:], 1):  # Últimas 3 interacciones
                st.write(f"{i}. **Tú**: {entry['query']} | **Chatbot**: {entry['answer']}")

if __name__ == "__main__":
    main()