# tests/test_api.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_chat_endpoint(client):
    response = client.post('/chat', json={'query': 'What is artificial intelligence?'})
    assert response.status_code == 200
    assert 'answer' in response.json
    assert 'retrieved_chunks' in response.json

def test_history_endpoint(client):
    response = client.get('/history')
    assert response.status_code == 200
    assert isinstance(response.json, list)