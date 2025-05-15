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


def update_faiss_with_feedback(vectordb, embeddings, feedback_file):
    high_quality_texts = []
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    feedback_data = json.loads(line.strip())
                    if feedback_data["rating"] >= 4:  # Solo respuestas con calificación alta
                        # Combinar consulta y respuesta para mayor contexto
                        text = f"Consulta: {feedback_data['query']} | Respuesta: {feedback_data['answer']}"
                        high_quality_texts.append(text)
                except json.JSONDecodeError:
                    continue
        
        if high_quality_texts:
            # Crear documentos para FAISS
            new_docs = [Document(page_content=t) for t in high_quality_texts]
            # Actualizar el índice FAISS
            vectordb.add_documents(new_docs)
            st.success(f"Índice FAISS actualizado con {len(high_quality_texts)} nuevas entradas.")
    return vectordb

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
                    if "text" in data and not re.search(r'\b(?:Axel|Franco)\b', data["text"], re.IGNORECASE):
                        texts.append(data["text"])
                    else:
                        st.warning(f"Línea omitida: {line}")
                except json.JSONDecodeError as e:
                    st.error(f"Error al parsear línea: {e} - Línea: {line}")
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
    feedback_file = "feedback_log.jsonl"
    vectordb = update_faiss_with_feedback(vectordb, embeddings, feedback_file)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    generator = load_pipeline()

    # Personalización del usuario
    st.markdown("**Personaliza tu experiencia**")
    user_name = st.text_input("¿Cuál es tu nombre? (Opcional)", "")
    user_context = st.text_area("Describe brevemente tu situación (Opcional)", "")

    # Historial de conversación
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Modo de revisión
    if st.checkbox("Modo de revisión"):
        review_feedback(feedback_file)
    else:
        query = st.text_input("Haz una pregunta o comparte cómo te sientes:")
        if query:
            with st.spinner("Buscando respuesta..."):
                emocion = detectar_emocion(query)
                docs = retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in docs])
                historial = "\n".join(
                    [f"Usuario: {entry['query']} | Respuesta: {entry['answer']}" 
                     for entry in st.session_state.chat_history[-2:]]
                ) if st.session_state.chat_history else "No hay historial."

                prompt = (
                    "Eres un chatbot bíblico con un enfoque cristiano protestante. Tu objetivo es ofrecer respuestas empáticas, personalizadas y basadas en la Biblia. "
                    "Usa un tono cálido, pastoral y alentador. Reconoce la emoción del usuario, incluye una cita bíblica relevante, y ofrece un consejo práctico. "
                    f"Historial reciente:\n{historial}\n\n"
                    f"Contexto de diálogos relevantes:\n{context}\n\n"
                    f"Nombre del usuario: {user_name if user_name else 'No proporcionado'}\n"
                    f"Contexto adicional: {user_context if user_context else 'No proporcionado'}\n"
                    f"Emoción detectada: {emocion}\n"
                    f"Consulta: {query}\n\n"
                    "### Respuesta ###"
                )

                result = generator(prompt, max_length=300, do_sample=True, temperature=0.7, top_p=0.9)
                answer = result[0]["generated_text"].strip()
                answer = limpiar_respuesta(answer, prompt)

                st.session_state.chat_history.append({"query": query, "answer": answer})

            st.markdown("**Respuesta:**")
            st.write(answer if answer else "No se pudo generar una respuesta.")

            # Retroalimentación
            st.markdown("**¿Qué te pareció la respuesta?**")
            rating = st.slider("Califica la respuesta (1 = Mala, 5 = Excelente)", 1, 5, 3, key=f"rating_{len(st.session_state.chat_history)}")
            feedback = st.text_area("Comentarios sobre la respuesta (opcional)", key=f"feedback_{len(st.session_state.chat_history)}")

            if st.button("Enviar retroalimentación"):
                feedback_data = {
                    "query": query,
                    "answer": answer,
                    "rating": rating,
                    "feedback": feedback,
                    "timestamp": "2025-05-15 00:00:00"
                }
                with open(feedback_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
                st.success("¡Gracias por tu retroalimentación!")
                
                if rating <= 2 and feedback:
                    st.markdown("**Lo sentimos, parece que la respuesta no fue satisfactoria.**")
                    alternative = st.text_area("¿Cómo te gustaría que fuera la respuesta? (Opcional)", key=f"alternative_{len(st.session_state.chat_history)}")
                    if st.button("Enviar sugerencia"):
                        suggestion_data = {
                            "query": query,
                            "original_answer": answer,
                            "rating": rating,
                            "feedback": feedback,
                            "suggested_answer": alternative,
                            "timestamp": "2025-05-15 00:00:00"
                        }
                        with open(feedback_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(suggestion_data, ensure_ascii=False) + "\n")
                        st.success("Gracias por tu sugerencia. ¡La usaremos para mejorar!")

            # Mostrar historial
            if st.session_state.chat_history:
                st.markdown("**Historial de conversación:**")
                for i, entry in enumerate(st.session_state.chat_history[-3:], 1):
                    st.write(f"{i}. **Tú**: {entry['query']} | **Chatbot**: {entry['answer']}")

        # Generar dataset para ajuste fino
        if st.button("Generar dataset para ajuste fino"):
            finetune_data = prepare_finetune_dataset(feedback_file)
            if finetune_data:
                with open("finetune_dataset.json", "w", encoding="utf-8") as f:
                    json.dump(finetune_data, f, ensure_ascii=False, indent=2)
                st.success("Dataset para ajuste fino creado.")
if __name__ == "__main__":
    main()