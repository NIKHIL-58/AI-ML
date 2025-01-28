from flask import Flask, request, jsonify
from database.db_manager import DatabaseManager
from embeddings.embedding_manager import EmbeddingManager
from retrieval.vector_store import VectorStore
from transformers import pipeline
import numpy as np
from config import Config
import requests
import wikipedia
import textwrap

app = Flask(__name__)
db_manager = DatabaseManager()
embedding_manager = EmbeddingManager()
vector_store = None

def get_wikipedia_content():
    # Get content from multiple Wikipedia pages about AI
    try:
        # List of AI-related topics
        topics = [
            'Artificial intelligence',
            'Machine learning',
            'Deep learning',
            'Natural language processing',
            'Computer vision'
        ]
        
        all_content = []
        for topic in topics:
            try:
                # Get the Wikipedia page content
                page = wikipedia.page(topic)
                # Add the content
                all_content.append(page.content[:2000])  # Get first 2000 chars of each topic
            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation page, take the first suggestion
                try:
                    page = wikipedia.page(e.options[0])
                    all_content.append(page.content[:2000])
                except:
                    continue
            except:
                continue
                
        return '\n\n'.join(all_content)
    except Exception as e:
        # Fallback content in case of any issues
        return """
        Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. 
        These processes include learning (the acquisition of information and rules for using the information), 
        reasoning (using rules to reach approximate or definite conclusions) and self-correction.

        Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn 
        and improve from experience without being explicitly programmed. Machine learning focuses on the development 
        of computer programs that can access data and use it to learn for themselves.

        Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with 
        representation learning. Learning can be supervised, semi-supervised or unsupervised.

        Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret 
        and manipulate human language. NLP draws from many disciplines, including computer science and computational 
        linguistics, in its pursuit to fill the gap between human communication and computer understanding.
        """

def chunk_text(text, chunk_size=300):
    """Split text into chunks of approximately equal size."""
    # Split into sentences first (crude approach)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Join the current chunk and add to chunks
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

def init_vector_store():
    global vector_store
    
    # Get content from Wikipedia
    text = get_wikipedia_content()
    
    # Chunk the text
    chunks = chunk_text(text, Config.CHUNK_SIZE)
    
    # Get embeddings for all chunks
    embeddings = embedding_manager.get_embeddings(chunks)
    
    # Initialize and populate vector store
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_texts(chunks, embeddings)

    return vector_store

def generate_answer(query: str, context: str) -> str:
    try:
        # Using a small model for generation
        generator = pipeline('text-generation', model='gpt2')
        
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = generator(prompt, max_length=150, num_return_sequences=1)
        
        return response[0]['generated_text']
    except Exception as e:
        # Fallback to a simple extraction-based approach
        relevant_sentences = [s for s in context.split('.') if query.lower() in s.lower()]
        if relevant_sentences:
            return relevant_sentences[0] + '.'
        return "I apologize, but I couldn't generate a specific answer based on the available information."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Save user message
    db_manager.save_message('user', query)
    
    # Get query embedding
    query_embedding = embedding_manager.get_embedding(query)
    
    # Retrieve relevant chunks
    results = vector_store.search(query_embedding, Config.TOP_K_RESULTS)
    context = ' '.join([text for text, _ in results])
    
    # Generate answer
    answer = generate_answer(query, context)
    
    # Save system response
    db_manager.save_message('system', answer)
    
    return jsonify({
        'answer': answer,
        'retrieved_chunks': [{'text': text, 'score': score} for text, score in results]
    })

@app.route('/history', methods=['GET'])
def history():
    chat_history = db_manager.get_chat_history()
    return jsonify(chat_history)

if __name__ == '__main__':
    print("Creating database tables...")
    from database.models import create_tables
    create_tables()
    print("Initializing vector store with Wikipedia content...")
    init_vector_store()
    print("Vector store initialized. Starting server...")
    app.run(debug=True)
