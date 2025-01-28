# embeddings/embedding_manager.py
from sentence_transformers import SentenceTransformer
from config import Config

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)

    def get_embedding(self, text):
        return self.model.encode(text)

    def get_embeddings(self, texts):
        return self.model.encode(texts)