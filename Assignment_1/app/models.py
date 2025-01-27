from pydantic import BaseModel

class ReviewInput(BaseModel):
    text: str

class SentimentPrediction(BaseModel):
    text: str
    sentiment: str
    confidence: float