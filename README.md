# IMDB Movie Review Sentiment Analysis

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

