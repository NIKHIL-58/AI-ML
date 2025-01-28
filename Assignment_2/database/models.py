# database/models.py
from datetime import datetime
import mysql.connector
from config import Config

def create_tables():
    conn = mysql.connector.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB
    )
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            role VARCHAR(10) NOT NULL,
            content TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()
