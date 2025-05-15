import streamlit as st
import json
import os
import re
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

# Función para detectar emociones o temas
def detectar_emocion(query):
    query = query.lower()
    emociones = {
        "ansiedad": ["ansioso", "nervioso", "preocupado", "miedo", "angustiado"],
        "tristeza": ["triste", "deprimido", "desanimado", "llorar", "perdida"],
        "soledad": ["solo", "abandonado", "nadie", "solitario"],
        "estrés": ["estresado", "abrumado", "cansado", "presion"],
        "duda": ["dudo", "fe", "confundido", "inseguro"],
        "enojo": ["enojado", "frustrado", "molesto"],
        "culpa": ["culpa", "arrepentido", "avergonzado"]
    }
    for emocion, palabras in emociones.items():
        if any(palabra in query for palabra in palabras):
            return emocion
    return "general"

# Función para limpiar la respuesta
def limpiar_respuesta(answer, prompt):
    # Eliminar el prompt y cualquier fragmento no deseado
    answer = answer.replace(prompt, "").strip()
    # Eliminar prefijos comunes
    for prefix in ["Respuesta en español:", "Contexto adicional:", "Nombre del usuario:"]:
        answer = answer.replace(prefix, "").strip()
    # Eliminar nombres propios irrelevantes (como "Axel Franco")
    answer = re.sub(r'\b(?:Axel|Franco)\b', '', answer, flags=re.IGNORECASE).strip()
    # Eliminar cualquier texto entre delimitadores no deseados
    answer = re.sub(r'###.*?###', '', answer).strip()
    # Eliminar líneas vacías o espacios múltiples
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer

def main():
    st.title("🤖 Chatbot Bíblico (Español) con HuggingFace + FAISS")

    jsonl_file_path = "docs/conversaciones_autoayuda_cristiana_unido.jsonl"

    # Verificar si el archivo existe
    if not os.path.exists(jsonl_file_path):
        st.error(f"No se encontró el archivo JSONL en la ruta: {jsonl_file_path}")
        return

    # Leer y validar el archivo JSONL
    texts = []
    try:
        with open(jsonl_file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    if "text" in data:
                        # Filtrar entradas con nombres propios no deseados
                        if not re.search(r'\b(?:Axel|Franco)\b', data["text"], re.IGNORECASE):
                            texts.append(data["text"])
                        else:
                            st.warning(f"Línea omitida, contiene nombres no deseados: {line}")
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
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    generator = load_pipeline()

    # Personalización del usuario
    st.markdown("**Personaliza tu experiencia**")
    user_name = st.text_input("¿Cuál es tu nombre? (Opcional)", "")
    user_context = st.text_area("Describe brevemente tu situación (Opcional)", "")

    # Historial de conversación
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Haz una pregunta o comparte cómo te sientes:")
    if query:
        with st.spinner("Buscando respuesta..."):
            # Detectar emoción
            emocion = detectar_emocion(query)

            # Obtener documentos relevantes
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])

            # Incluir historial reciente (últimas 2 interacciones)
            historial = "\n".join(
                [f"Usuario: {entry['query']} | Respuesta: {entry['answer']}" 
                 for entry in st.session_state.chat_history[-2:]]
            ) if st.session_state.chat_history else "No hay historial."

            # Prompt optimizado
            prompt = (
                "Eres un chatbot bíblico con un enfoque cristiano protestante. Tu objetivo es ofrecer respuestas empáticas, personalizadas y basadas en la Biblia para ayudar al usuario con sus problemas o emociones. "
                "Usa un tono cálido, pastoral y alentador. Reconoce la emoción del usuario, incluye una cita bíblica relevante, y ofrece un consejo práctico. "
                "Si el usuario proporciona su nombre o contexto, incorpóralo de manera natural. Evita incluir nombres irrelevantes o partes del prompt en la respuesta. "
                "Basándote en el historial y el contexto, asegura que la respuesta sea coherente y relevante.\n\n"
                "Ejemplo:\n"
                "Consulta: Me siento muy ansioso por mi trabajo.\n"
                "Respuesta: Entiendo lo abrumador que puede ser sentirse ansioso por el trabajo. La Biblia nos recuerda en Filipenses 4:6-7: 'Por nada estéis afanosos, sino sean conocidas vuestras peticiones delante de Dios en toda oración y ruego'. Te animo a tomar un momento para orar y entregar tus preocupaciones a Dios. Quizás puedas escribir lo que te preocupa y orar específicamente por ello.\n\n"
                f"Historial reciente:\n{historial}\n\n"
                f"Contexto de diálogos relevantes:\n{context}\n\n"
                f"Nombre del usuario: {user_name if user_name else 'No proporcionado'}\n"
                f"Contexto adicional: {user_context if user_context else 'No proporcionado'}\n"
                f"Emoción detectada: {emocion}\n"
                f"Consulta: {query}\n\n"
                "### Respuesta ###"
            )

            # Generar respuesta
            result = generator(prompt, max_length=300, do_sample=True, temperature=0.7, top_p=0.9)
            answer = result[0]["generated_text"].strip()

            # Limpiar respuesta
            answer = limpiar_respuesta(answer, prompt)

            # Guardar en historial
            st.session_state.chat_history.append({"query": query, "answer": answer})

        st.markdown("**Respuesta:**")
        st.write(answer if answer else "No se pudo generar una respuesta.")

        # Mostrar historial
        if st.session_state.chat_history:
            st.markdown("**Historial de conversación:**")
            for i, entry in enumerate(st.session_state.chat_history[-3:], 1):
                st.write(f"{i}. **Tú**: {entry['query']} | **Chatbot**: {entry['answer']}")

if __name__ == "__main__":
    main()