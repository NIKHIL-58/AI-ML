# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DB = os.getenv('MYSQL_DB', 'ragchatbot')
    CHUNK_SIZE = 300
    TOP_K_RESULTS = 3
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
