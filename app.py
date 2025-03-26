# No need for eventlet anymore
import os
import json
import tempfile
import traceback
import shutil
from datetime import datetime, timedelta
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from multi_model_rag import MultiModalRAGManager, GlobalFAISSManager
from flask_cors import CORS
import logging
import redis
import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool
import asyncio
import threading
import time
from flask_session import Session


load_dotenv()

app = Flask(__name__, static_folder='extracted_images', static_url_path='/extracted_images', template_folder='templates')
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "your-secret-key")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
Session(app)

# Configure SocketIO with gevent explicitly
socketio = SocketIO(
    app,
    async_mode='gevent',
    cors_allowed_origins="*",
    engineio_logger=True  # Enable logging for debugging
    
)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PDF_FOLDER = os.path.abspath(r'D:\multi-rag\pdfs')
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(PDF_FOLDER, exist_ok=True)

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)
CACHE_SIZE = 20
CACHE_TTL = 24 * 60 * 60

DB_CONFIG = {
    'host': os.getenv("MYSQL_HOST", "localhost"),
    'user': os.getenv("MYSQL_USER", "root"),
    'password': os.getenv("MYSQL_PASSWORD", "0000"),
    'raise_on_warnings': True,
    'database': 'atrip-LLM'
}

DB_CONFIG['pool_name'] = 'atrip_pool'
DB_CONFIG['pool_size'] = 10
try:
    db_pool = MySQLConnectionPool(**DB_CONFIG)
    logger.info("MySQL connection pool initialized successfully")
except mysql.connector.Error as e:
    logger.error(f"Failed to initialize MySQL connection pool: {e}")
    raise

def get_db_connection():
    try:
        conn = db_pool.get_connection()
        if conn.database != 'atrip-LLM':
            conn.database = 'atrip-LLM'
        return conn
    except mysql.connector.Error as e:
        logger.error(f"Error getting connection from pool: {e}")
        raise

def init_mysql_db(max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            conn = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            cursor = conn.cursor()
            logger.info("Connected to MySQL server successfully")

            cursor.execute("SHOW DATABASES LIKE 'atrip-LLM'")
            database_exists = cursor.fetchone() is not None

            if not database_exists:
                cursor.execute("CREATE DATABASE `atrip-LLM`")
                logger.info("Created database 'atrip-LLM'")
            else:
                logger.info("Database 'atrip-LLM' already exists, skipping creation")

            cursor.execute("USE `atrip-LLM`")
            logger.info("Selected database 'atrip-LLM' using USE statement")

            tables = {
                'users': '''
                    CREATE TABLE IF NOT EXISTS `users` (
                        `id` INT AUTO_INCREMENT PRIMARY KEY,
                        `username` VARCHAR(50) NOT NULL UNIQUE,
                        `password` VARCHAR(255) NOT NULL,
                        `is_admin` BOOLEAN DEFAULT FALSE,
                        `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''',
                'user_requests': '''
                    CREATE TABLE IF NOT EXISTS `user_requests` (
                        `id` INT AUTO_INCREMENT PRIMARY KEY,
                        `username` VARCHAR(50) NOT NULL UNIQUE,
                        `password` VARCHAR(255) NOT NULL,
                        `request_status` ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
                        `requested_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        `reviewed_at` TIMESTAMP NULL
                    )
                ''',
                'named_conversations': '''
                    CREATE TABLE IF NOT EXISTS `named_conversations` (
                        `id` VARCHAR(36) PRIMARY KEY,
                        `user_id` INT NOT NULL,
                        `conversation_name` VARCHAR(255) NOT NULL,
                        `session_id` VARCHAR(255) DEFAULT (UUID()),
                        `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        `committed` BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE,
                        UNIQUE (`user_id`, `conversation_name`),
                        INDEX `idx_user_conversation` (`user_id`, `conversation_name`)
                    )
                ''',
                'conversations': '''
                    CREATE TABLE IF NOT EXISTS `conversations` (
                        `id` INT AUTO_INCREMENT PRIMARY KEY,
                        `user_id` INT NOT NULL,
                        `conversation_id` VARCHAR(36) NOT NULL,
                        `role` ENUM('user', 'assistant', 'system') NOT NULL,
                        `content` TEXT NOT NULL,
                        `timestamp` DATETIME NOT NULL,
                        `summary_flag` BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE,
                        FOREIGN KEY (`conversation_id`) REFERENCES `named_conversations`(`id`) ON DELETE CASCADE,
                        INDEX `idx_user_timestamp` (`user_id`, `timestamp`),
                        INDEX `idx_conversation_timestamp` (`conversation_id`, `timestamp`)
                    )
                ''',
                'conversation_sessions': '''
                    CREATE TABLE IF NOT EXISTS `conversation_sessions` (
                        `id` INT AUTO_INCREMENT PRIMARY KEY,
                        `conversation_id` VARCHAR(36) NOT NULL,
                        `session_id` VARCHAR(36) NOT NULL,
                        `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (`conversation_id`) REFERENCES `named_conversations`(`id`) ON DELETE CASCADE
                    )
                ''',
                'conversation_summaries': '''
                    CREATE TABLE IF NOT EXISTS `conversation_summaries` (
                        `id` INT AUTO_INCREMENT PRIMARY KEY,
                        `user_id` INT NOT NULL,
                        `conversation_id` VARCHAR(36) NOT NULL,
                        `summary` TEXT,
                        `timestamp` FLOAT,
                        FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE,
                        FOREIGN KEY (`conversation_id`) REFERENCES `named_conversations`(`id`) ON DELETE CASCADE
                    )
                '''
            }

            for table_name, create_statement in tables.items():
                try:
                    cursor.execute(create_statement)
                    logger.info(f"Table '{table_name}' created or already exists")
                except mysql.connector.errors.DatabaseError as e:
                    if e.errno == 1050:
                        logger.info(f"Table '{table_name}' already exists, checking schema")
                        if table_name == 'named_conversations':
                            cursor.execute("SHOW COLUMNS FROM `named_conversations` LIKE 'committed'")
                            if not cursor.fetchone():
                                cursor.execute("ALTER TABLE `named_conversations` ADD COLUMN `committed` BOOLEAN DEFAULT FALSE")
                                logger.info("Added 'committed' column to named_conversations table")
                    else:
                        logger.error(f"Error creating table '{table_name}': {e}")
                        raise

            conn.commit()

            cursor.execute("SELECT COUNT(*) FROM `users` WHERE `is_admin` = TRUE")
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO `users` (`username`, `password`, `is_admin`) VALUES (%s, %s, %s)",
                    ("admin", generate_password_hash("admin123"), True)
                )
                conn.commit()
                logger.info("Created default admin user")

            cursor.close()
            conn.close()
            logger.info("MySQL database and tables initialized successfully")
            return
        except mysql.connector.Error as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} - Error initializing MySQL: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)

init_mysql_db()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY is not set in environment variables.")
    raise ValueError("GROQ_API_KEY is required.")

rag_manager = MultiModalRAGManager(
    groq_api_key=groq_api_key,
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    pdf_folder_path=PDF_FOLDER,
    vector_db_path="./faiss_db/multimodal_index",
    llm_model_name="llama3-70b-8192",
    json_output_dir="./json_outputs"
)

global_memory = GlobalFAISSManager()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_existing_pdf_names():
    try:
        pdf_files = []
        for root, dirs, files in os.walk(PDF_FOLDER):
            for file in files:
                if file.lower().endswith('.pdf'):
                    relative_path = os.path.relpath(os.path.join(root, file), PDF_FOLDER).replace('\\', '/')
                    pdf_files.append(relative_path)
        logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        return pdf_files
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        return []

def ingest_single_document(file_path: str, filename: str):
    final_path = os.path.join(PDF_FOLDER, filename).replace('\\', '/').lower()
    if os.path.exists(final_path):
        existing_file_hash = rag_manager._get_file_hash(final_path)
        uploaded_file_hash = rag_manager._get_file_hash(file_path)
        if existing_file_hash == uploaded_file_hash:
            return {"processed_files": 0, "skipped_files": 1, "total_chunks": 0, "total_images": 0}
    stats = rag_manager.ingest_single_document(file_path=file_path, filename=filename, force_reprocess=False)
    return stats

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

@app.before_request
def ensure_session():
    if 'user_id' in session and 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

def get_user_id():
    return session.get('user_id')

def get_current_conversation_id():
    conversation_id = session.get('current_conversation_id')
    if not conversation_id:
        user_id = get_user_id()
        if user_id:
            conversations = get_user_conversations(user_id)
            if conversations:
                conversation_id = conversations[0]["id"]
            else:
                logger.info("No conversations found. Creating a new one.")
                conversation_id, _ = create_new_conversation(user_id, "Temporary Conversation", commit_conversation=False)
            set_current_conversation(conversation_id)
    return conversation_id

def set_current_conversation(conversation_id):
    session['current_conversation_id'] = conversation_id
    session.modified = True

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

def get_short_term_history(conversation_id, limit=10):
    key = f"conversation:{conversation_id}:short_term"
    messages = redis_client.lrange(key, 0, limit - 1)
    return [json.loads(msg) for msg in messages]

async def store_message(user_id, conversation_id, role, content):
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        conn.start_transaction()

        cursor.execute(
            'SELECT id FROM named_conversations WHERE id = %s AND user_id = %s',
            (conversation_id, user_id)
        )
        if not cursor.fetchone():
            logger.warning(f"Conversation {conversation_id} not found for user {user_id}. Creating a new one.")
            conversation_id, _ = create_new_conversation(user_id, "Temporary Conversation", commit_conversation=False)
            set_current_conversation(conversation_id)

        if isinstance(content, dict):
            content = json.dumps(content)

        cursor.execute(
            'INSERT INTO conversations (user_id, conversation_id, role, content, timestamp) VALUES (%s, %s, %s, %s, %s)',
            (user_id, conversation_id, role, content, timestamp_str)
        )
        conn.commit()

        await rag_manager.store_conversation_embedding(user_id, conversation_id, role, content, time.time())
    except Error as e:
        logger.error(f"Error storing message in MySQL: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_conversation_history(user_id, conversation_id, limit=10, use_global_memory=False):
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
            {"role": row['role'], "content": row['content'], "timestamp": row['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if row['timestamp'] else "N/A"} 
            for row in cursor.fetchall()
        ]
        
        if not long_term and not short_term and use_global_memory:
            global_results = global_memory.search_user_memory(user_id, "", top_k=limit)
            long_term = [{"role": "assistant", "content": doc.page_content, "timestamp": datetime.fromtimestamp(doc.metadata.get("timestamp", time.time())).strftime("%Y-%m-%d %H:%M:%S")} 
                        for doc in global_results]
        
        combined_history = (short_term + long_term)[-limit:][::-1]
        return combined_history
    except Error as e:
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

def create_new_conversation(user_id, conversation_name="Temporary Conversation", commit_conversation=False):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        conversation_id = str(uuid.uuid4())

        base_name = conversation_name
        final_name = conversation_name
        suffix = 1
        while True:
            cursor.execute(
                '''
                SELECT COUNT(*) FROM named_conversations 
                WHERE user_id = %s AND conversation_name = %s
                ''',
                (user_id, final_name)
            )
            if cursor.fetchone()[0] == 0:
                break
            final_name = f"{base_name} ({suffix})"
            suffix += 1

        cursor.execute(
            '''
            INSERT INTO named_conversations (id, user_id, conversation_name, created_at, committed)
            VALUES (%s, %s, %s, %s, %s)
            ''',
            (conversation_id, user_id, final_name, datetime.now(), commit_conversation)
        )
        conn.commit()

        session_id = str(uuid.uuid4())
        cursor.execute(
            'INSERT INTO conversation_sessions (conversation_id, session_id) VALUES (%s, %s)',
            (conversation_id, session_id)
        )
        conn.commit()

        redis_key = f"conversation:{conversation_id}:short_term"
        redis_client.delete(redis_key)
    except Error as e:
        logger.error(f"Error creating new conversation: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
    return conversation_id, session_id

def generate_conversation_title(content):
    if not content or len(content.strip()) == 0:
        return "Untitled Conversation"
    
    filler_words = {'what', 'are', 'the', 'is', 'a', 'an', 'in', 'on', 'for', 'to', 'with', 'about'}
    words = content.lower().split()
    meaningful_words = [word for word in words if word not in filler_words and len(word) > 2]
    
    if not meaningful_words:
        return "Untitled Conversation"
    
    title_words = meaningful_words[:3]
    title = " ".join(word.capitalize() for word in title_words)
    
    if len(title) > 30:
        title = title[:27] + "..."
    
    return title

async def async_query(rag_manager, query, user_id, conversation_id, history):
    try:
        new_topic_indicators = ["let's start", "begin with", "new topic", "switch to", "tell me about"]
        is_new_topic = any(indicator in query.lower() for indicator in new_topic_indicators)

        response = await rag_manager.query(
            query,
            k=5,
            user_id=user_id,
            conversation_id=conversation_id,
            history=history if not is_new_topic else []
        )
        return response
    except Exception as e:
        logger.error(f"Error in async_query: {str(e)}")
        return {"response": f"Error: {str(e)}", "status": "error", "images": []}

def cleanup_uncommitted_conversations(cutoff_hours=24):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cutoff_time = datetime.now() - timedelta(hours=cutoff_hours)
        cursor.execute(
            '''
            DELETE nc FROM named_conversations nc
            LEFT JOIN conversation_sessions cs ON nc.id = cs.conversation_id
            WHERE nc.committed = FALSE 
            AND nc.created_at < %s
            AND cs.session_id IS NULL
            ''',
            (cutoff_time,)
        )
        deleted_count = cursor.rowcount
        conn.commit()
        logger.info(f"Cleaned up {deleted_count} uncommitted conversations")
    except mysql.connector.Error as e:
        logger.error(f"Error cleaning up uncommitted conversations: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def cleanup_temporary_conversations(cutoff_hours=24):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cutoff_time = datetime.now() - timedelta(hours=cutoff_hours)
        cursor.execute(
            '''
            DELETE FROM named_conversations 
            WHERE conversation_name LIKE 'Temporary Conversation%' 
            AND created_at < %s
            ''',
            (cutoff_time,)
        )
        deleted_count = cursor.rowcount
        conn.commit()
        logger.info(f"Cleaned up {deleted_count} temporary conversations")
    except mysql.connector.Error as e:
        logger.error(f"Error cleaning up temporary conversations: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def start_cleanup_thread(interval_seconds=3600, cutoff_hours=24):
    while True:
        try:
            cleanup_uncommitted_conversations(cutoff_hours)
            cleanup_temporary_conversations(cutoff_hours)
        except Exception as e:
            logger.error(f"Fatal error in cleanup thread: {e}")
            break
        time.sleep(interval_seconds)

def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

def admin_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session or not session.get('is_admin'):
            flash("Admin access required.")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/')
@login_required
def index():
    user_id = get_user_id()
    is_admin = session.get('is_admin', False)
    conversations = get_user_conversations(user_id)
    
    if not get_current_conversation_id():
        conversation_id, session_id = create_new_conversation(user_id, "Temporary Conversation", commit_conversation=False)
        set_current_conversation(conversation_id)
    
    return render_template('index.html', is_admin=is_admin, conversations=conversations)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT id, password, is_admin FROM users WHERE username = %s', (username,))
            user = cursor.fetchone()
            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                session['is_admin'] = user[2]
                session['username'] = username
                session.permanent = True
                logger.info(f"User {username} logged in successfully")
                return redirect(url_for('index'))
            flash("Invalid credentials.")
        except Error as e:
            logger.error(f"Error during login: {e}")
            flash("An error occurred. Please try again.")
        finally:
            cursor.close()
            conn.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    user_id = get_user_id()
    conversation_id = get_current_conversation_id()
    if user_id and conversation_id:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag_manager.summarize_session(user_id, conversation_id))
        loop.close()
    session.clear()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO user_requests (username, password) VALUES (%s, %s)',
                (username, generate_password_hash(password))
            )
            conn.commit()
            flash("Registration request submitted. Await admin approval.")
            return redirect(url_for('login'))
        except Error as e:
            logger.error(f"Error registering user: {e}")
            flash("Username already exists or an error occurred.")
        finally:
            cursor.close()
            conn.close()
    return render_template('register.html')

@app.route('/admin', methods=['GET'])
@admin_required
def admin_panel():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, request_status, requested_at FROM user_requests WHERE request_status = "pending"')
        requests = cursor.fetchall()
        return render_template('admin.html', requests=requests)
    except Error as e:
        logger.error(f"Error fetching user requests: {e}")
        flash("Error loading admin panel.")
        return redirect(url_for('index'))
    finally:
        cursor.close()
        conn.close()

@app.route('/admin/approve/<int:request_id>', methods=['POST'])
@admin_required
def approve_user(request_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT username, password FROM user_requests WHERE id = %s AND request_status = "pending"', (request_id,))
        request_data = cursor.fetchone()
        if request_data:
            cursor.execute(
                'INSERT INTO users (username, password) VALUES (%s, %s)',
                (request_data[0], request_data[1])
            )
            cursor.execute('UPDATE user_requests SET request_status = "approved", reviewed_at = NOW() WHERE id = %s', (request_id,))
            conn.commit()
            flash(f"User {request_data[0]} approved.")
        else:
            flash("Request not found or already processed.")
    except Error as e:
        logger.error(f"Error approving user: {e}")
        flash("Error approving user.")
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('admin_panel'))

@app.route('/admin/reject/<int:request_id>', methods=['POST'])
@admin_required
def reject_user(request_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE user_requests SET request_status = "rejected", reviewed_at = NOW() WHERE id = %s', (request_id,))
        conn.commit()
        flash("User request rejected.")
    except Error as e:
        logger.error(f"Error rejecting user: {e}")
        flash("Error rejecting user.")
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('admin_panel'))

@app.route('/status', methods=['GET'])
@login_required
def status():
    try:
        stats = {
            'processed_files': len(rag_manager.processed_files),
            'pdf_folder': PDF_FOLDER,
            'pdf_count': len([f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]),
            'image_count': len(rag_manager.image_vector_store.vector_store.docstore._dict)
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc(), "status": "error"}), 500

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename).lower()
        subfolder = request.form.get('subfolder', '')
        if subfolder:
            final_path = os.path.join(PDF_FOLDER, subfolder, filename).lower()
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
        else:
            final_path = os.path.join(PDF_FOLDER, filename).lower()

        existing_pdf_names = get_existing_pdf_names()
        if os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/') in existing_pdf_names:
            file_hash = rag_manager._get_file_hash(final_path)
            normalized_path = final_path.replace('\\', '/')
            if normalized_path in rag_manager.processed_files and rag_manager.processed_files[normalized_path] == file_hash:
                pass
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        try:
            stats = ingest_single_document(temp_path, os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/'))
            user_id = get_user_id()
            conversation_id = get_current_conversation_id()
            if not conversation_id:
                return jsonify({"error": "No conversation selected", "status": "error"}), 400
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(store_message(user_id, conversation_id, "user", f"Uploaded file: {os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/')}"))
                if stats.get("processed_files", 0) > 0:
                    shutil.move(temp_path, final_path)
                    loop.run_until_complete(store_message(user_id, conversation_id, "assistant", f"File {os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/')} uploaded and processed successfully"))
                    return jsonify({"message": f"File {os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/')} uploaded and processed successfully", "filename": os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/'), "stats": stats, "status": "success"})
                else:
                    os.remove(temp_path)
                    loop.run_until_complete(store_message(user_id, conversation_id, "assistant", f"File {os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/')} is already in the database."))
                    return jsonify({"message": f"File {os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/')} is already in the database.", "filename": os.path.relpath(final_path, PDF_FOLDER).replace('\\', '/'), "status": "exists"})
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error processing uploaded file {filename}: {str(e)}")
            os.remove(temp_path)
            return jsonify({"error": str(e), "traceback": traceback.format_exc(), "status": "error"}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/summarize_history', methods=['POST'])
@login_required
def summarize_history():
    user_id = get_user_id()
    conversation_id = get_current_conversation_id()
    
    if not conversation_id:
        logger.error("No conversation selected for summarization")
        return jsonify({"error": "No conversation selected", "status": "error"}), 400
    
    try:
        history = get_conversation_history(user_id, conversation_id, limit=10)
        if not history:
            logger.info(f"No history found for conversation {conversation_id}")
            return jsonify({"summary": "No conversation history to summarize.", "status": "success", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            summary = loop.run_until_complete(
                rag_manager.summarize_conversation(user_id, conversation_id, history=history)
            )
            
            loop.run_until_complete(
                store_message(user_id, conversation_id, "assistant", f"Conversation Summary:\n{summary}")
            )
            
            logger.info(f"Successfully summarized conversation {conversation_id} for user {user_id}")
            
            return jsonify({
                "summary": summary,
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Error summarizing history for conversation {conversation_id}: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/list_pdfs', methods=['GET'])
@login_required
def list_pdfs():
    try:
        pdf_files = get_existing_pdf_names()
        return jsonify({"pdfs": pdf_files, "status": "success", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/pdfs/<path:filename>', methods=['GET'])
@login_required
def serve_pdf(filename):
    try:
        full_path = os.path.join(PDF_FOLDER, filename)
        if not os.path.exists(full_path):
            base_dir = os.path.dirname(full_path)
            target_filename = os.path.basename(full_path).lower()
            if os.path.exists(base_dir):
                for f in os.listdir(base_dir):
                    if f.lower() == target_filename:
                        full_path = os.path.join(base_dir, f)
                        break
                else:
                    raise Exception(f"File not found at: {full_path}")
            else:
                raise Exception(f"Directory not found: {base_dir}")

        with open(full_path, 'rb') as f:
            f.read(1)

        relative_path = os.path.relpath(full_path, PDF_FOLDER).replace('\\', '/')
        return send_from_directory(directory=PDF_FOLDER, path=relative_path)
    except Exception as e:
        logger.error(f"Error serving PDF {filename}: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 404

@app.route('/get_history', methods=['GET'])
@login_required
def get_history():
    user_id = get_user_id()
    conversation_id = get_current_conversation_id()
    if not conversation_id:
        return jsonify({"status": "error", "message": "No conversation selected"}), 400
    try:
        history = get_conversation_history(user_id, conversation_id, limit=50, use_global_memory=False)
        history = sorted(history, key=lambda x: x['timestamp'])
        return jsonify({"status": "success", "history": history, "conversation_id": conversation_id})
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_user_conversations', methods=['GET'])
@login_required
def get_user_conversations_route():
    user_id = get_user_id()
    try:
        conversations = get_user_conversations(user_id)
        return jsonify({"status": "success", "conversations": conversations})
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/delete_message/<int:message_id>', methods=['POST'])
@login_required
def delete_message(message_id):
    user_id = get_user_id()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT conversation_id FROM conversations WHERE id = %s AND user_id = %s',
            (message_id, user_id)
        )
        result = cursor.fetchone()
        if not result:
            return jsonify({"status": "error", "message": "Message not found or not authorized"}), 404

        cursor.execute('DELETE FROM conversations WHERE id = %s AND user_id = %s', (message_id, user_id))
        conn.commit()
        return jsonify({"status": "success", "message": "Message deleted"})
    except Error as e:
        logger.error(f"Error deleting message {message_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/new_chat', methods=['POST'])
@login_required
def new_chat():
    user_id = get_user_id()
    try:
        conversation_id, session_id = create_new_conversation(user_id, "Temporary Conversation", commit_conversation=False)
        set_current_conversation(conversation_id)
        return jsonify({
            "status": "success",
            "conversation_id": conversation_id,
            "session_id": session_id,
            "conversation_name": "New Conversation"
        })
    except Exception as e:
        logger.error(f"Error creating new chat: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/select_conversation/<conversation_id>', methods=['POST'])
@login_required
def select_conversation(conversation_id):
    user_id = get_user_id()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id FROM named_conversations WHERE id = %s AND user_id = %s',
            (conversation_id, user_id)
        )
        if not cursor.fetchone():
            return jsonify({"status": "error", "message": "Conversation not found"}), 404
        set_current_conversation(conversation_id)
        return jsonify({"status": "success", "message": "Conversation selected"})
    except Error as e:
        logger.error(f"Error selecting conversation {conversation_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/rename_conversation/<conversation_id>', methods=['POST'])
@login_required
def rename_conversation(conversation_id):
    user_id = get_user_id()
    new_name = request.form.get('new_name')
    if not new_name or len(new_name.strip()) == 0:
        return jsonify({"status": "error", "message": "New name cannot be empty"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT id FROM named_conversations 
            WHERE id = %s AND user_id = %s
            ''',
            (conversation_id, user_id)
        )
        if not cursor.fetchone():
            return jsonify({"status": "error", "message": "Conversation not found"}), 404

        cursor.execute(
            '''
            SELECT COUNT(*) FROM named_conversations 
            WHERE user_id = %s AND conversation_name = %s AND id != %s
            ''',
            (user_id, new_name, conversation_id)
        )
        if cursor.fetchone()[0] > 0:
            return jsonify({"status": "error", "message": "Conversation name already exists"}), 400

        cursor.execute(
            '''
            UPDATE named_conversations 
            SET conversation_name = %s 
            WHERE id = %s
            ''',
            (new_name, conversation_id)
        )
        conn.commit()
        return jsonify({"status": "success", "message": "Conversation renamed", "new_name": new_name})
    except Error as e:
        logger.error(f"Error renaming conversation {conversation_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/delete_conversation/<conversation_id>', methods=['POST'])
@login_required
def delete_conversation(conversation_id):
    user_id = get_user_id()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT id FROM named_conversations 
            WHERE id = %s AND user_id = %s
            ''',
            (conversation_id, user_id)
        )
        if not cursor.fetchone():
            return jsonify({"status": "error", "message": "Conversation not found"}), 404

        cursor.execute(
            'DELETE FROM named_conversations WHERE id = %s AND user_id = %s',
            (conversation_id, user_id)
        )
        conn.commit()

        redis_key = f"conversation:{conversation_id}:short_term"
        redis_client.delete(redis_key)
        return jsonify({"status": "success", "message": "Conversation deleted"})
    except Error as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@socketio.on('message')
def handle_message(data):
    query = data.get('question', '')
    user_id = session.get('user_id')
    conversation_id = session.get('conversation_id', str(uuid.uuid4()))
    history = get_short_term_history(conversation_id)
    socketio.emit('typing', room=request.sid)
    response = asyncio.run(async_query(rag_manager, query, user_id, conversation_id, history))
    socketio.emit('message', {'status': 'complete', 'response': response['response'], 'images': response.get('images', [])}, room=request.sid)

@socketio.on('file_query')
async def handle_file_query(data):
    if 'user_id' not in session:
        emit('message', {'response': "Please log in to continue.", 'status': 'error', 'requires_login': True})
        return

    question = data.get('question')
    file_content = data.get('file_content')
    file_name = data.get('file_name')
    file_type = data.get('file_type')
    history = data.get('history', [])
    user_id = get_user_id()
    conversation_id = get_current_conversation_id()
    if not conversation_id:
        emit('message', {'response': "No conversation selected.", 'status': 'error', 'images': []})
        return

    try:
        await store_message(user_id, conversation_id, "user", f"{question} (with file: {file_name})")
        join_room(str(conversation_id))
        emit('typing', {'status': 'typing'}, room=str(conversation_id))

        result = await async_query(rag_manager, f"{question} (with file: {file_name})", user_id, conversation_id, history)
        await store_message(user_id, conversation_id, "assistant", result['response'])
        emit('message', result, room=str(conversation_id))
    except Exception as e:
        logger.error(f"Error processing file query: {str(e)}")
        emit('message', {'response': f"Error: {str(e)}", 'status': 'error', 'images': []}, room=str(conversation_id))
    finally:
        leave_room(str(conversation_id))

if __name__ == '__main__':
    cleanup_interval = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600"))
    cleanup_cutoff = int(os.getenv("CLEANUP_CUTOFF_HOURS", "24"))
    threading.Thread(target=start_cleanup_thread, args=(cleanup_interval, cleanup_cutoff), daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)