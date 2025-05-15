import os
import json
from langchain.schema import Document
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()

# Cargar docs
docs = []
for filename in os.listdir("docs"):
    if filename.endswith(".jsonl"):
        with open(os.path.join("docs", filename), encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    docs.append(Document(page_content=text))

texts = [doc.page_content for doc in docs]

embeddings = SentenceTransformerEmbeddings()

# Sin persistencia
vectordb = Chroma.from_texts(texts, embeddings)

# Probar búsqueda simple
query = "¿Qué dice el salmo sobre la ansiedad?"
results = vectordb.similarity_search(query, k=3)
for r in results:
    print(r.page_content)