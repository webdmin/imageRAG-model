import mysql.connector
from mysql.connector import Error
import os

DB_CONFIG = {
    'host': os.getenv("MYSQL_HOST", "localhost"),
    'user': os.getenv("MYSQL_USER", "root"),
    'password': os.getenv("MYSQL_PASSWORD", ""),
    'raise_on_warnings': True,
    'database': 'atrip-LLM'
}

def alter_conversation_summaries():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Ensure the table exists
        cursor.execute("SHOW TABLES LIKE 'conversation_summaries'")
        if not cursor.fetchone():
            print("Table 'conversation_summaries' does not exist.")
            return

        print("[INFO] Editing conversation_summaries table...")

        # ✅ Example 1: Modify the timestamp column to DATETIME
        cursor.execute("""
            ALTER TABLE conversation_summaries
            MODIFY timestamp DATETIME DEFAULT CURRENT_TIMESTAMP;
        """)
        print("[SUCCESS] Updated timestamp column to DATETIME format.")

        # ✅ Example 2: Add a new 'validated_images' column (for image references)
        cursor.execute("""
            ALTER TABLE conversation_summaries
            ADD COLUMN validated_images JSON AFTER summary;
        """)
        print("[SUCCESS] Added 'validated_images' column.")

        # ✅ Example 3: Add an index on the 'user_id' column for faster queries
        cursor.execute("""
            CREATE INDEX idx_user_id ON conversation_summaries (user_id);
        """)
        print("[SUCCESS] Created index on 'user_id' column.")

        # ✅ Example 4: Drop a column if no longer needed (e.g., "timestamp")
        # cursor.execute("""
        #     ALTER TABLE conversation_summaries
        #     DROP COLUMN timestamp;
        # """)
        # print("[SUCCESS] Dropped 'timestamp' column.")

        conn.commit()
        print("[INFO] All table modifications applied successfully.")

    except Error as e:
        print(f"[ERROR] Failed to alter table: {e}")

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("[INFO] MySQL connection closed.")

# Execute the alterations
alter_conversation_summaries()
