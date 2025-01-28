# tests/test_retrieval.py
import pytest
import numpy as np
from retrieval.vector_store import VectorStore

def test_vector_store():
    dimension = 384  # Matches the embedding dimension
    store = VectorStore(dimension)
    
    # Test adding texts
    texts = ["test document 1", "test document 2"]
    embeddings = np.random.rand(2, dimension).astype('float32')
    store.add_texts(texts, embeddings)
    
    # Test searching
    query_embedding = np.random.rand(dimension).astype('float32')
    results = store.search(query_embedding, k=1)
    
    assert len(results) == 1
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][0], str)
    assert isinstance(results[0][1], float)