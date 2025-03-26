# db_utils.py
import os
import redis
import json
from datetime import datetime
import time
import mysql.connector
from dotenv import load_dotenv
from app import global_memory  # Import only global_memory from app

load_dotenv()

DB_CONFIG = {
    'host': os.getenv("MYSQL_HOST", "localhost"),
    'user': os.getenv("MYSQL_USER", "root"),
    'password': os.getenv("MYSQL_PASSWORD", ""),
    'database': os.getenv("MYSQL_DB", "atrip-LLM")
}

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)
CACHE_SIZE = 10
CACHE_TTL = 24 * 60 * 60

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise

def get_short_term_history(conversation_id, limit=5):
    key = f"conversation:{conversation_id}:short_term"
    messages = redis_client.lrange(key, 0, limit - 1)
    return [json.loads(msg) for msg in messages]

def get_conversation_history(user_id, conversation_id, limit=10, use_global_memory=False):
    short_term = get_short_term_history(conversation_id, limit=limit)
    history_size = len(short_term)
    adjusted_limit = max(5, min(50, history_size + 5))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        '''
        SELECT role, content, timestamp 
        FROM conversations 
        WHERE user_id = %s AND conversation_id = %s
        ORDER BY timestamp DESC 
        LIMIT %s
        ''',
        (user_id, conversation_id, adjusted_limit - len(short_term))
    )
    long_term = [
        {"role": row['role'], "content": row['content'], 
         "timestamp": row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"), "type": "text"}
        for row in cursor.fetchall()
    ]
    
    global_results = global_memory.search_user_memory(user_id, "", top_k=adjusted_limit)
    image_context = [
        {"role": "assistant", "content": doc.page_content, 
         "timestamp": datetime.fromtimestamp(doc.metadata.get("timestamp", time.time())).strftime("%Y-%m-%d %H:%M:%S"), 
         "type": "image", "path": doc.metadata.get("local_path")}
        for doc in global_results if doc.metadata.get("context") == "image_analysis"
    ]
    
    combined_history = sorted(short_term + long_term + image_context, key=lambda x: x["timestamp"])[-adjusted_limit:]
    cursor.close()
    conn.close()
    return combined_history[::-1]

def get_user_conversations(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        '''
        SELECT id, conversation_name, created_at 
        FROM named_conversations 
        WHERE user_id = %s 
        ORDER BY created_at DESC
        ''',
        (user_id,)
    )
    conversations = cursor.fetchall()
    cursor.close()
    conn.close()
    return conversations