# Assignment_1 IMDB Movie Review Sentiment Analysis

This project implements an end-to-end sentiment analysis pipeline for IMDB movie reviews using FastAPI. The system downloads the IMDB dataset, stores reviews in a SQLite database, trains a sentiment classification model, and serves predictions through a REST API.

## Project Structure

```
.
├── app/
│   ├── main.py           # FastAPI application and endpoints
│   ├── database.py       # Database configuration and models
│   ├── models.py         # Pydantic models for API
│   └── sentiment_model.py # Sentiment analysis model implementation
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Project Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone <repository-https://github.com/NIKHIL-58/AI-ML.git>
cd sentiment-analysis
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Acquisition

The IMDB dataset is automatically downloaded using the Hugging Face datasets library when you first run the application. The dataset includes:
- 25,000 training reviews
- 25,000 test reviews
- Binary sentiment labels (positive/negative)

The data loading process is handled in `app/main.py` using:
```python
from datasets import load_dataset
dataset = load_dataset("imdb", split="train")
```

## Database Setup

The application uses SQLite for data storage. The database is automatically created when you first run the application. The database file (`imdb_reviews.db`) will be created in your project root directory.

Database schema:
```sql
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    sentiment STRING
)
```

## Running the Application

1. Start the FastAPI server:
```bash
python -m uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`. On first startup, it will:
- Create the SQLite database
- Download the IMDB dataset
- Train the sentiment analysis model (this may take a few minutes)
- Save the trained model as `sentiment_model.pkl`

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing the API

You can test the endpoint using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was fantastic! I really enjoyed every minute of it."}'
```

Or using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was fantastic! I really enjoyed every minute of it."}
)
print(response.json())
```

Example response:
```json
{
    "text": "This movie was fantastic! I really enjoyed every minute of it.",
    "sentiment": "positive",
    "confidence": 0.92
}
```

## Model Information

### Model Architecture
The sentiment analysis model uses a pipeline of:
1. TF-IDF Vectorization (max_features=10000)
2. Logistic Regression Classifier

### Text Preprocessing
- HTML tag removal
- Conversion to lowercase
- Special character and digit removal
- Stop word removal

### Performance Metrics
On the IMDB test set:
- Accuracy: ~87%
- F1-Score: ~0.86
- Precision: ~0.88
- Recall: ~0.85

### Model Persistence
- The trained model is saved as `sentiment_model.pkl`
- The application automatically loads the saved model on startup
- If no saved model exists, it trains a new one

## Additional Features

1. Confidence Scores
   - Each prediction includes a confidence score
   - Scores range from 0 to 1, indicating prediction certainty

2. Database Storage
   - All predictions are stored in the SQLite database
   - Enables future analysis and model improvement

3. Automatic API Documentation
   - Interactive API documentation via Swagger UI
   - Alternative documentation via ReDoc

## Error Handling

The API includes comprehensive error handling:
- Invalid input validation
- Model prediction errors
- Database connection issues

## Future Improvements

Potential enhancements:
1. Implementation of BERT or other transformer models
2. Batch prediction capabilities
3. Model retraining endpoint
4. Performance monitoring and logging
5. Docker containerization








# Assignment_2 RAG (Retrieval-Augmented Generation) Chatbot

A Python-based implementation of a RAG chatbot that combines vector database search with language model generation. The system features both a REST API and a Gradio web interface for easy interaction.

## 🌟 Features

- **RAG-based Question Answering**: Combines retrieval and generation for accurate responses
- **Dual Interface**: 
  - REST API for programmatic access
- **Vector Search**: Uses FAISS for efficient similarity search
- **Wikipedia Integration**: Automatically fetches and indexes AI-related content
- **Chat History**: Stores conversation history in MySQL database
- **Real-time Response**: Fast response generation using GPT-2

## 🛠️ Technologies Used

- Flask (REST API)
- Gradio (Web UI)
- FAISS (Vector Database)
- Sentence Transformers (Embeddings)
- MySQL (Chat History)
- Hugging Face Transformers (Text Generation)
- Wikipedia API (Content Fetching)

## 📋 Prerequisites

- Python 3.10 or higher
- MySQL Server
- pip package manager

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-https://github.com/NIKHIL-58/AI-ML/tree/main/Assignment_2>
cd rag-chatbot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up MySQL database:
```sql
CREATE DATABASE ragchatbot;
```

4. Configure environment variables (create .env file):
```
MYSQL_HOST=localhost
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DB=ragchatbot
```

## 🎯 Usage

### Running the Application

Start the application:
```bash
python app.py
```

This will:
1. Initialize the vector store with Wikipedia content
2. Start the Flask API server on port 5000

### REST API Endpoints

1. **Chat Endpoint**
   - URL: `http://127.0.0.1:5000/chat`
   - Method: `POST`
   - Request Body:
     ```json
     {
         "query": "What is artificial intelligence?"
     }
     ```
   - Example Response:
     ```json
     {
       "answer": "Context: Artificial intelligence (AI), in its broadest sense...",
       "retrieved_chunks": [
         {
           "score": 0.33678317070007324,
           "text": "Artificial intelligence (AI), in its broadest sense..."
         },
         {
           "score": 0.5507127046585083,
           "text": "It is a field of research in computer science..."
         },
         {
           "score": 0.7288822531700134,
           "text": "Artificial intelligence was founded as an academic discipline..."
         }
       ]
     }
     ```

2. **History Endpoint**
   - URL: `http://127.0.0.1:5000/history`
   - Method: `GET`
   - Example Response:
     ```json
     [
         {
             "content": "What is artificial intelligence?",
             "id": 1,
             "role": "user",
             "timestamp": "Tue, 28 Jan 2025 12:32:30 GMT"
         },
         {
             "content": "Context: Artificial intelligence (AI)...",
             "id": 2,
             "role": "system",
             "timestamp": "Tue, 28 Jan 2025 12:32:39 GMT"
         }
     ]
     ```


## 📁 Project Structure
```
rag_chatbot/
├── app.py                  # Main application file
├── config.py              # Configuration settings
├── requirements.txt       # Package dependencies
├── database/             # Database related files
│   ├── models.py         # Database models
│   └── db_manager.py     # Database operations
├── embeddings/           # Embedding related files
│   └── embedding_manager.py
└── retrieval/            # Vector store operations
    └── vector_store.py
```

## ⚠️ Notes

- The system uses GPT-2 for text generation. For production use, consider using a more sophisticated model.
- The vector store is initialized with Wikipedia content at startup. This may take a few minutes.
- Chat history is persisted in MySQL and can be accessed via the `/history` endpoint.

## 🤝 Contributing

Feel free to open issues and pull requests for improvements!

