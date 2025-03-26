# utils.py
from asyncio.log import logger
from flask import session
import redis
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import mysql.connector

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
        logger.error(f"Error connecting to MySQL: {e}")
        raise

def clear_conversation_cache(conversation_id):
    try:
        redis_client.delete(f"conversation:{conversation_id}:short_term")
        logger.info(f"Cleared Redis cache for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error clearing cache for conversation {conversation_id}: {e}")

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    return session['session_id']

def get_short_term_history(conversation_id, limit=5):
    key = f"conversation:{conversation_id}:short_term"
    messages = redis_client.lrange(key, 0, limit - 1)
    return [json.loads(msg) for msg in messages]

def get_user_id():
    return session.get('user_id')

def store_short_term_message(conversation_id, role, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "user_id": get_user_id(),
        "conversation_id": conversation_id
    }
    key = f"conversation:{conversation_id}:short_term"
    redis_client.lpush(key, json.dumps(message))
    redis_client.ltrim(key, 0, CACHE_SIZE - 1)
    redis_client.expire(key, CACHE_TTL)


def get_conversation_history(user_id, conversation_id, limit=10, use_global_memory=False):
    # Note: This version will be updated in app.py with global_memory, so we keep it minimal here
    short_term = get_short_term_history(conversation_id, limit)
    if len(short_term) >= limit:
        return short_term[::-1]
    
    try:
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
            (user_id, conversation_id, limit - len(short_term))
        )
        long_term = [
            {"role": row['role'], "content": row['content'], 
             "timestamp": row['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if row['timestamp'] else "N/A"}
            for row in cursor.fetchall()
        ]
        combined_history = (short_term + long_term)[-limit:][::-1]
        return combined_history
    except mysql.connector.Error as e:
        logger.error(f"Error fetching history: {e}")
        return short_term[::-1]
    finally:
        cursor.close()
        conn.close()

def get_user_conversations(user_id):
    try:
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
        return conversations
    except mysql.connector.Error as e:
        logger.error(f"Error fetching user conversations: {e}")
        raise
    finally:
        cursor.close()
        conn.close()