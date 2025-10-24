#!/usr/bin/env python3
"""
Migration script to upgrade existing Discord database to optimized schema
"""
import sqlite3
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from sqlite_performance_config import SQLitePerformanceOptimizer, run_performance_optimization
from lib.database_optimized import OptimizedDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('db_migration')

def backup_database(db_path: str) -> str:
    """Create backup of existing database"""
    
    backup_path = f"{db_path}.backup_{int(time.time())}"
    
    try:
        # Simple file copy for backup
        import shutil
        shutil.copy2(db_path, backup_path)
        logger.info(f"✅ Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Failed to backup database: {e}")
        raise

def check_existing_schema(conn: sqlite3.Connection) -> dict:
    """Check existing database schema"""
    
    cursor = conn.cursor()
    
    # Get existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    # Get table row counts
    table_counts = {}
    for table in ['servers', 'channels', 'messages']:
        if table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            table_counts[table] = cursor.fetchone()[0]
    
    # Get existing indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = [row[0] for row in cursor.fetchall()]
    
    return {
        'tables': tables,
        'table_counts': table_counts,
        'indexes': indexes,
        'has_intelligence_layer': 'message_importance' in tables
    }

def migrate_add_intelligence_tables(conn: sqlite3.Connection):
    """Add intelligence layer tables to existing database"""
    
    cursor = conn.cursor()
    
    logger.info("📊 Adding intelligence layer tables...")
    
    # Add columns to existing messages table
    try:
        cursor.execute('ALTER TABLE messages ADD COLUMN author_id INTEGER')
        cursor.execute('ALTER TABLE messages ADD COLUMN author_name TEXT')
        cursor.execute('ALTER TABLE messages ADD COLUMN message_type TEXT DEFAULT "default"')
        cursor.execute('ALTER TABLE messages ADD COLUMN has_attachments BOOLEAN DEFAULT 0')
        cursor.execute('ALTER TABLE messages ADD COLUMN attachment_count INTEGER DEFAULT 0')
        cursor.execute('ALTER TABLE messages ADD COLUMN reply_to_id INTEGER')
        cursor.execute('ALTER TABLE messages ADD COLUMN thread_id INTEGER')
        cursor.execute('ALTER TABLE messages ADD COLUMN created_at INTEGER DEFAULT (strftime("%s", "now"))')
        cursor.execute('ALTER TABLE messages ADD COLUMN processed_at INTEGER')
        logger.info("✅ Enhanced messages table")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.info("ℹ️  Messages table already has new columns")
        else:
            logger.warning(f"Could not add columns to messages table: {e}")
    
    # Add columns to servers table
    try:
        cursor.execute('ALTER TABLE servers ADD COLUMN created_at INTEGER DEFAULT (strftime("%s", "now"))')
        cursor.execute('ALTER TABLE servers ADD COLUMN last_activity INTEGER DEFAULT (strftime("%s", "now"))')
        cursor.execute('ALTER TABLE servers ADD COLUMN message_count INTEGER DEFAULT 0')
        cursor.execute('ALTER TABLE servers ADD COLUMN avg_importance REAL DEFAULT 0.0')
        logger.info("✅ Enhanced servers table")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.info("ℹ️  Servers table already has new columns")
        else:
            logger.warning(f"Could not add columns to servers table: {e}")
    
    # Add columns to channels table
    try:
        cursor.execute('ALTER TABLE channels ADD COLUMN created_at INTEGER DEFAULT (strftime("%s", "now"))')
        cursor.execute('ALTER TABLE channels ADD COLUMN last_activity INTEGER DEFAULT (strftime("%s", "now"))')
        cursor.execute('ALTER TABLE channels ADD COLUMN message_count INTEGER DEFAULT 0')
        cursor.execute('ALTER TABLE channels ADD COLUMN avg_importance REAL DEFAULT 0.0')
        cursor.execute('ALTER TABLE channels ADD COLUMN channel_type TEXT DEFAULT "text"')
        cursor.execute('ALTER TABLE channels ADD COLUMN is_monitored BOOLEAN DEFAULT 1')
        logger.info("✅ Enhanced channels table")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower():
            logger.info("ℹ️  Channels table already has new columns")
        else:
            logger.warning(f"Could not add columns to channels table: {e}")
    
    # Create intelligence layer tables using OptimizedDatabase
    db = OptimizedDatabase()
    db.conn = conn  # Use existing connection
    db._create_intelligence_tables(cursor)
    
    conn.commit()
    logger.info("✅ Intelligence layer tables created")

def migrate_create_indexes(conn: sqlite3.Connection):
    """Create performance indexes"""
    
    logger.info("🚀 Creating performance indexes...")
    
    db = OptimizedDatabase()
    db.conn = conn
    db._create_indexes(conn.cursor())
    
    conn.commit()
    logger.info("✅ Performance indexes created")

def migrate_create_views(conn: sqlite3.Connection):
    """Create optimized views"""
    
    logger.info("👁️  Creating optimized views...")
    
    db = OptimizedDatabase()
    db.conn = conn
    db._create_views(conn.cursor())
    
    conn.commit()
    logger.info("✅ Optimized views created")

def migrate_calculate_importance_scores(conn: sqlite3.Connection, batch_size: int = 1000):
    """Calculate importance scores for existing messages"""
    
    cursor = conn.cursor()
    
    # Get total message count
    cursor.execute("SELECT COUNT(*) FROM messages WHERE content IS NOT NULL")
    total_messages = cursor.fetchone()[0]
    
    if total_messages == 0:
        logger.info("ℹ️  No messages to process for importance scoring")
        return
    
    logger.info(f"📈 Calculating importance scores for {total_messages} existing messages...")
    
    db = OptimizedDatabase()
    processed = 0
    
    # Process messages in batches
    offset = 0
    while offset < total_messages:
        cursor.execute("""
        SELECT id, content, author_name, reply_to_id
        FROM messages 
        WHERE content IS NOT NULL
        ORDER BY id
        LIMIT ? OFFSET ?
        """, (batch_size, offset))
        
        messages = cursor.fetchall()
        if not messages:
            break
        
        for message_id, content, author_name, reply_to_id in messages:
            try:
                # Calculate importance
                has_mention = '@' in (content or '')
                link_count = (content or '').count('http')
                
                importance = db.calculate_message_importance(
                    message_id, content, author_name, has_mention, link_count, reply_to_id
                )
                
                # Save importance data
                cursor.execute('''
                INSERT OR REPLACE INTO message_importance
                (message_id, importance_score, urgency_score, relevance_score, social_score,
                 keyword_matches, mention_count, link_count, has_mention, has_keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message_id,
                    importance.score,
                    importance.factors.get('urgency', 0),
                    importance.factors.get('group_activity', 0) + importance.factors.get('events', 0),
                    importance.factors.get('mentions', 0),
                    json.dumps(importance.keywords),
                    len(importance.mentions),
                    len(importance.links),
                    has_mention,
                    len(importance.keywords) > 0
                ))
                
                processed += 1
                
            except Exception as e:
                logger.warning(f"Error processing message {message_id}: {e}")
        
        # Commit batch
        conn.commit()
        
        # Progress update
        if processed % 1000 == 0:
            progress = (processed / total_messages) * 100
            logger.info(f"⏳ Progress: {processed}/{total_messages} ({progress:.1f}%)")
        
        offset += batch_size
    
    logger.info(f"✅ Calculated importance scores for {processed} messages")

def migrate_populate_fts_index(conn: sqlite3.Connection):
    """Populate Full-Text Search index"""
    
    cursor = conn.cursor()
    
    logger.info("🔍 Populating Full-Text Search index...")
    
    try:
        # Insert all messages into FTS index
        cursor.execute('''
        INSERT INTO messages_fts (rowid, content, author_name)
        SELECT id, COALESCE(content, ''), COALESCE(author_name, '')
        FROM messages
        ''')
        
        conn.commit()
        
        # Get count of indexed messages
        cursor.execute("SELECT COUNT(*) FROM messages_fts")
        fts_count = cursor.fetchone()[0]
        
        logger.info(f"✅ Populated FTS index with {fts_count} messages")
        
    except Exception as e:
        logger.error(f"❌ Error populating FTS index: {e}")

def run_migration(db_path: str, skip_backup: bool = False, batch_size: int = 1000):
    """Run complete database migration"""
    
    db_path = Path(db_path)
    if not db_path.exists():
        logger.error(f"❌ Database file not found: {db_path}")
        return False
    
    logger.info(f"🚀 Starting database migration for: {db_path}")
    start_time = time.time()
    
    try:
        # Create backup unless skipped
        if not skip_backup:
            backup_path = backup_database(str(db_path))
            logger.info(f"💾 Backup created: {backup_path}")
        
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        
        # Check existing schema
        schema_info = check_existing_schema(conn)
        logger.info(f"📋 Current schema: {schema_info['table_counts']}")
        
        if schema_info['has_intelligence_layer']:
            logger.info("ℹ️  Intelligence layer already exists, skipping table creation")
        else:
            # Add intelligence tables
            migrate_add_intelligence_tables(conn)
        
        # Create indexes (safe to run multiple times)
        migrate_create_indexes(conn)
        
        # Create views
        migrate_create_views(conn)
        
        # Populate FTS index
        migrate_populate_fts_index(conn)
        
        # Calculate importance scores for existing messages
        if not schema_info['has_intelligence_layer']:
            migrate_calculate_importance_scores(conn, batch_size)
        else:
            logger.info("ℹ️  Importance scores already exist, skipping calculation")
        
        # Run performance optimization
        logger.info("⚡ Running performance optimization...")
        optimizer = SQLitePerformanceOptimizer(str(db_path))
        optimizer.optimize_database_maintenance(conn)
        
        conn.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✅ Migration completed successfully in {duration:.2f} seconds!")
        logger.info("🎯 Your database is now optimized for real-time monitoring!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Discord database to optimized schema")
    parser.add_argument(
        "--db-path", 
        default="data/db.sqlite", 
        help="Path to database file (default: data/db.sqlite)"
    )
    parser.add_argument(
        "--skip-backup", 
        action="store_true", 
        help="Skip creating backup (not recommended)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1000, 
        help="Batch size for processing messages (default: 1000)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be migrated without making changes"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No changes will be made")
        
        if Path(args.db_path).exists():
            conn = sqlite3.connect(args.db_path)
            schema_info = check_existing_schema(conn)
            conn.close()
            
            logger.info("📊 Current database status:")
            logger.info(f"  • Tables: {', '.join(schema_info['tables'])}")
            logger.info(f"  • Row counts: {schema_info['table_counts']}")
            logger.info(f"  • Indexes: {len(schema_info['indexes'])}")
            logger.info(f"  • Intelligence layer: {'✅ Present' if schema_info['has_intelligence_layer'] else '❌ Missing'}")
            
            if not schema_info['has_intelligence_layer']:
                logger.info("\n🚀 Migration would add:")
                logger.info("  • Intelligence layer tables (message_importance, detected_patterns, etc.)")
                logger.info("  • Performance indexes for fast querying")
                logger.info("  • Full-text search capabilities")
                logger.info("  • Materialized views for common queries")
                logger.info(f"  • Importance scoring for ~{schema_info['table_counts'].get('messages', 0)} messages")
            else:
                logger.info("\n✅ Database already has intelligence layer")
                logger.info("  Migration would only update indexes and views")
        else:
            logger.error(f"❌ Database file not found: {args.db_path}")
        
        return
    
    # Run actual migration
    success = run_migration(args.db_path, args.skip_backup, args.batch_size)
    
    if success:
        logger.info("\n🎉 Next steps:")
        logger.info("  1. Use 'load_messages_optimized.py' for future message imports")
        logger.info("  2. Run 'streamlit run streamlit_monitoring.py' to access the monitoring interface")
        logger.info("  3. Check the new monitoring dashboard for real-time insights")
    else:
        logger.error("\n💥 Migration failed. Check the logs above for details.")
        if not args.skip_backup:
            logger.info("  Your original database backup is available for restore if needed.")

if __name__ == "__main__":
    main()