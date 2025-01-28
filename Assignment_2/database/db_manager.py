# database/db_manager.py
import mysql.connector
from config import Config
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.config = {
            'host': Config.MYSQL_HOST,
            'user': Config.MYSQL_USER,
            'password': Config.MYSQL_PASSWORD,
            'database': Config.MYSQL_DB
        }

    def save_message(self, role, content):
        conn = mysql.connector.connect(**self.config)
        cursor = conn.cursor()
        
        query = "INSERT INTO chat_messages (role, content) VALUES (%s, %s)"
        cursor.execute(query, (role, content))
        
        conn.commit()
        cursor.close()
        conn.close()

    def get_chat_history(self):
        conn = mysql.connector.connect(**self.config)
        cursor = conn.cursor(dictionary=True)
        
        query = "SELECT * FROM chat_messages ORDER BY timestamp ASC"
        cursor.execute(query)
        history = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return history
