# retrieval/vector_store.py
import faiss
import numpy as np
from typing import List, Tuple

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        self.texts.extend(texts)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(distance)))
        return results