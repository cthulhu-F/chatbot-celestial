import json
import random

# Listas para generar variedad en los diálogos
problemas = [
    "Me siento tan ansioso, no sé cómo manejar esto.",
    "Estoy muy triste, siento que no puedo seguir adelante.",
    "Me siento solo, como si nadie me entendiera.",
    "Tengo miedo de lo que viene en el futuro.",
    "Estoy agotado, no tengo fuerzas para continuar.",
    "Tuve una discusión con mi familia y me siento mal.",
    "Siento que he fallado en todo lo que hago.",
    "He perdido a alguien importante y no sé cómo sobrellevarlo.",
    "Estoy tan estresado con el trabajo, no puedo más.",
    "A veces dudo de mi fe, ¿es normal sentir esto?"
]

respuestas_base = [
    "Entiendo lo abrumador que puede ser {problema}. Dios nos recuerda en {versículo} que {mensaje}. ¿Por qué no tomas un momento para orar y confiarle tus preocupaciones? Él promete estar contigo siempre.",
    "Siento mucho que estés pasando por esto. En {versículo}, la Biblia dice {mensaje}. Lleva tus cargas a Dios en oración, Él quiere darte consuelo y fuerza.",
    "Es normal sentirse así a veces, pero no estás solo. {versículo} nos asegura que {mensaje}. Intenta hablar con Dios y pedirle su paz, Él te escucha.",
    "Tu dolor es real, y Dios lo ve. Según {versículo}, {mensaje}. ¿Te gustaría escribir lo que sientes y ofrecérselo a Dios en oración? Él está cerca de los quebrantados de corazón.",
    "Lo que sientes es difícil, pero Dios es tu refugio. En {versículo} leemos que {mensaje}. Confía en Él y busca un pequeño paso para avanzar, como leer un salmo o hablar con un amigo de fe."
]

versiculos = [
    ("Salmo 46:1", "Dios es nuestro amparo y fortaleza, nuestro pronto auxilio en las tribulaciones"),
    ("Filipenses 4:6-7", "no debemos angustiarnos, sino presentar nuestras peticiones a Dios, y su paz guardará nuestros corazones"),
    ("Mateo 11:28", "Jesús invita a los cansados y cargados a venir a Él, y les dará descanso"),
    ("Salmo 23:4", "aunque pasemos por valle de sombra, no temeremos, porque Dios está con nosotros"),
    ("Isaías 41:10", "Dios nos dice que no temamos, porque Él está con nosotros y nos sostiene"),
    ("Juan 16:33", "Jesús nos asegura que en Él tenemos paz, aunque en el mundo enfrentemos aflicciones"),
    ("Salmo 34:18", "el Señor está cerca de los quebrantados de corazón y salva a los de espíritu abatido"),
    ("2 Corintios 12:9", "la gracia de Dios es suficiente, y su poder se perfecciona en nuestra debilidad"),
    ("Proverbios 3:5-6", "debemos confiar en el Señor con todo nuestro corazón y Él enderezará nuestras sendas"),
    ("1 Pedro 5:7", "podemos echar toda nuestra ansiedad sobre Dios, porque Él cuida de nosotros")
]

# Función para generar un diálogo
def generar_dialogo(id):
    problema = random.choice(problemas)
    respuesta_base = random.choice(respuestas_base)
    versiculo, mensaje = random.choice(versiculos)
    
    # Sustituir {problema}, {versículo}, {mensaje} en la respuesta
    respuesta = respuesta_base.format(
        problema=problema.lower().split(",")[0],  # Simplifica el problema para la respuesta
        versículo=versiculo,
        mensaje=mensaje
    )
    
    # Combinar usuario y respuesta en un solo texto
    texto = f"Usuario: {problema} | Respuesta: {respuesta}"
    return {"id": id, "text": texto}

# Generar el archivo JSONL
with open("dialogos_autoayuda.jsonl", "w", encoding="utf-8") as file:
    for i in range(1, 10001):  # 10,000 líneas
        dialogo = generar_dialogo(i)
        json.dump(dialogo, file, ensure_ascii=False)
        file.write("\n")

print("Archivo 'dialogos_autoayuda.jsonl' generado con 10,000 líneas.")