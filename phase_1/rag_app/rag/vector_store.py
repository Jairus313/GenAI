import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings: list[list[float]], texts: list[str]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding: list[float], k: int = 3):
        vector = np.array([query_embedding]).astype("float32")
        _, indices = self.index.search(vector, k)
        return [self.texts[i] for i in indices[0]]
