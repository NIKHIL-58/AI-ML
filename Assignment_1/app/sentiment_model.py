import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

nltk.download('stopwords')
nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('classifier', LogisticRegression())
        ])
        
    def preprocess_text(self, text):
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def train(self, texts, labels):
        processed_texts = [self.preprocess_text(text) for text in texts]
        self.model.fit(processed_texts, labels)
    
    def predict(self, text):
        processed_text = self.preprocess_text(text)
        prediction = self.model.predict([processed_text])[0]
        confidence = max(self.model.predict_proba([processed_text])[0])
        return prediction, confidence
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)