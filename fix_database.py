#!/usr/bin/env python3
"""
Fix database schema and add ML tables
"""

import sqlite3
import sys
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent / 'data' / 'db.sqlite'

def fix_database():
    """Fix database schema issues and add new ML tables"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        print("Starting database fixes...")
        
        # Add missing columns to messages table
        print("Adding missing columns to messages table...")
        
        columns_to_add = [
            ('author_id', 'INTEGER'),
            ('author_name', 'TEXT'),
            ('message_type', 'TEXT DEFAULT "normal"'),
            ('mentions', 'TEXT'),
            ('attachments', 'TEXT')
        ]
        
        for column_name, column_type in columns_to_add:
            try:
                cursor.execute(f'ALTER TABLE messages ADD COLUMN {column_name} {column_type}')
                print(f"  ✓ Added column: {column_name}")
            except sqlite3.OperationalError:
                print(f"  - Column {column_name} already exists")
        
        # Create new ML tables
        print("\nCreating ML tables...")
        
        # Purchase predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS purchase_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER,
            channel_id INTEGER,
            probability REAL,
            prediction_type TEXT,
            metadata TEXT,
            predicted_at INTEGER,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
        ''')
        print("  ✓ Created purchase_predictions table")
        
        # Deadlines table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS deadlines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER,
            channel_id INTEGER,
            deadline_date INTEGER,
            deadline_text TEXT,
            urgency_level INTEGER,
            reminder_sent BOOLEAN DEFAULT 0,
            created_at INTEGER,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
        ''')
        print("  ✓ Created deadlines table")
        
        # Conversation threads table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            message_id INTEGER,
            channel_id INTEGER,
            position INTEGER,
            thread_type TEXT,
            created_at INTEGER,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
        ''')
        print("  ✓ Created conversation_threads table")
        
        # Sentiment scores table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            message_id INTEGER PRIMARY KEY,
            excitement_level REAL,
            sentiment_type TEXT,
            confidence REAL,
            emoji_count INTEGER,
            exclamation_count INTEGER,
            analyzed_at INTEGER,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
        ''')
        print("  ✓ Created sentiment_scores table")
        
        # Create indexes for performance
        print("\nCreating indexes...")
        
        indexes = [
            ('idx_purchase_pred_prob', 'purchase_predictions(probability DESC)'),
            ('idx_purchase_pred_time', 'purchase_predictions(predicted_at DESC)'),
            ('idx_deadlines_date', 'deadlines(deadline_date)'),
            ('idx_deadlines_urgency', 'deadlines(urgency_level DESC)'),
            ('idx_threads_thread_id', 'conversation_threads(thread_id)'),
            ('idx_threads_channel', 'conversation_threads(channel_id)'),
            ('idx_sentiment_excitement', 'sentiment_scores(excitement_level DESC)'),
        ]
        
        for index_name, index_def in indexes:
            try:
                cursor.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON {index_def}')
                print(f"  ✓ Created index: {index_name}")
            except sqlite3.OperationalError as e:
                print(f"  - Index {index_name} issue: {e}")
        
        # Commit all changes
        conn.commit()
        
        print("\n✅ Database fixes completed successfully!")
        
        # Verify the changes
        print("\nVerifying changes...")
        cursor.execute("PRAGMA table_info(messages)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Messages columns: {columns}")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"All tables: {tables}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
        sys.exit(1)
    
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database()