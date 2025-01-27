from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import database, models
from .sentiment_model import SentimentAnalyzer
from datasets import load_dataset
import uvicorn

app = FastAPI(title="Movie Review Sentiment Analysis API")
sentiment_analyzer = SentimentAnalyzer()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    # Load and train the model on startup
    try:
        sentiment_analyzer.load_model("sentiment_model.pkl")
    except:
        print("Training new model...")
        dataset = load_dataset("imdb", split="train")
        texts = dataset["text"]
        labels = [1 if label == 1 else 0 for label in dataset["label"]]
        sentiment_analyzer.train(texts, labels)
        sentiment_analyzer.save_model("sentiment_model.pkl")

@app.post("/predict", response_model=models.SentimentPrediction)
async def predict_sentiment(review: models.ReviewInput, db: Session = Depends(get_db)):
    try:
        sentiment, confidence = sentiment_analyzer.predict(review.text)
        prediction = models.SentimentPrediction(
            text=review.text,
            sentiment="positive" if sentiment == 1 else "negative",
            confidence=float(confidence)
        )
        
        # Store the review in the database
        db_review = database.Review(text=review.text, sentiment=prediction.sentiment)
        db.add(db_review)
        db.commit()
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)