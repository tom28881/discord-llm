"""
SQLite Performance Configuration and Optimization Scripts
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger('sqlite_performance')

class SQLitePerformanceOptimizer:
    """SQLite performance optimization utilities"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def get_connection(self) -> sqlite3.Connection:
        """Get optimized SQLite connection"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        self.apply_performance_pragmas(conn)
        return conn
    
    def apply_performance_pragmas(self, conn: sqlite3.Connection):
        """Apply comprehensive performance pragmas"""
        
        # Journal mode optimization
        # WAL mode allows concurrent readers and writer
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Synchronous mode - balance safety and performance
        # NORMAL is safe for most applications and faster than FULL
        conn.execute("PRAGMA synchronous = NORMAL")
        
        # Cache size - use negative value for KB
        # 64MB cache for better performance with large datasets  
        conn.execute("PRAGMA cache_size = -64000")
        
        # Memory-mapped I/O
        # 256MB mmap size for faster file operations
        conn.execute("PRAGMA mmap_size = 268435456")
        
        # Temporary storage in memory
        conn.execute("PRAGMA temp_store = MEMORY")
        
        # Optimize for faster INSERTs at slight read cost
        conn.execute("PRAGMA optimize")
        
        # Enable foreign key constraints for data integrity
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Page size optimization (set at DB creation)
        # conn.execute("PRAGMA page_size = 4096")  # Only works during creation
        
        # Auto-vacuum for space management
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        
        # Busy timeout for handling locks
        conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
        
        logger.debug("Applied SQLite performance pragmas")
    
    def analyze_database_performance(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Analyze current database performance metrics"""
        
        cursor = conn.cursor()
        performance_metrics = {}
        
        # Database size information
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        
        db_size_mb = (page_count * page_size) / (1024 * 1024)
        performance_metrics['database_size_mb'] = round(db_size_mb, 2)
        
        # Index usage statistics
        cursor.execute("PRAGMA compile_options")
        compile_options = [row[0] for row in cursor.fetchall()]
        performance_metrics['compile_options'] = compile_options
        
        # Cache hit ratio (approximation)
        cursor.execute("PRAGMA cache_size")
        cache_size = cursor.fetchone()[0]
        performance_metrics['cache_size_kb'] = abs(cache_size) if cache_size < 0 else cache_size * page_size // 1024
        
        # Table and index information
        cursor.execute("""
        SELECT name, type, sql 
        FROM sqlite_master 
        WHERE type IN ('table', 'index')
        ORDER BY type, name
        """)
        schema_objects = cursor.fetchall()
        
        tables = [obj for obj in schema_objects if obj[1] == 'table']
        indexes = [obj for obj in schema_objects if obj[1] == 'index']
        
        performance_metrics['table_count'] = len(tables)
        performance_metrics['index_count'] = len(indexes)
        
        # Row counts for main tables
        table_stats = {}
        for table in ['messages', 'message_importance', 'detected_patterns']:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                table_stats[table] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                pass  # Table doesn't exist
        
        performance_metrics['table_row_counts'] = table_stats
        
        return performance_metrics
    
    def get_slow_queries(self, conn: sqlite3.Connection) -> List[str]:
        """Get recommendations for potentially slow queries"""
        
        recommendations = []
        cursor = conn.cursor()
        
        try:
            # Check for tables without indexes on commonly queried columns
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                if table in ['messages', 'message_importance']:
                    # Check if sent_at/calculated_at columns are indexed
                    cursor.execute(f"PRAGMA index_list('{table}')")
                    indexes = cursor.fetchall()
                    
                    time_column = 'sent_at' if table == 'messages' else 'calculated_at'
                    
                    # Check if time column is indexed
                    has_time_index = False
                    for index_info in indexes:
                        cursor.execute(f"PRAGMA index_info('{index_info[1]}')")
                        index_columns = [col[2] for col in cursor.fetchall()]
                        if time_column in index_columns:
                            has_time_index = True
                            break
                    
                    if not has_time_index:
                        recommendations.append(
                            f"Consider adding index on {table}.{time_column} for time-range queries"
                        )
            
            # Check for FTS table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'")
            fts_tables = cursor.fetchall()
            
            if not fts_tables:
                recommendations.append(
                    "Consider creating FTS (Full-Text Search) index for content searches"
                )
            
        except Exception as e:
            logger.error(f"Error analyzing slow queries: {e}")
        
        return recommendations
    
    def optimize_database_maintenance(self, conn: sqlite3.Connection):
        """Run optimization and maintenance tasks"""
        
        cursor = conn.cursor()
        
        try:
            # Update table statistics for query optimizer
            logger.info("Updating database statistics...")
            cursor.execute("ANALYZE")
            
            # Incremental vacuum to reclaim space
            logger.info("Running incremental vacuum...")
            cursor.execute("PRAGMA incremental_vacuum(1000)")  # Vacuum up to 1000 pages
            
            # Optimize database
            logger.info("Optimizing database...")
            cursor.execute("PRAGMA optimize")
            
            conn.commit()
            logger.info("Database optimization completed successfully")
            
        except Exception as e:
            logger.error(f"Error during database optimization: {e}")
            conn.rollback()
            raise
    
    def create_performance_indexes(self, conn: sqlite3.Connection):
        """Create specific indexes for performance optimization"""
        
        cursor = conn.cursor()
        
        # Performance-critical indexes
        performance_indexes = [
            # Core time-based queries (most important for "what happened while away")
            "CREATE INDEX IF NOT EXISTS idx_perf_messages_time_desc ON messages(sent_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_perf_messages_server_time ON messages(server_id, sent_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_perf_messages_channel_time ON messages(channel_id, sent_at DESC)",
            
            # Importance-based queries (critical for monitoring)
            "CREATE INDEX IF NOT EXISTS idx_perf_importance_score_time ON message_importance(importance_score DESC, calculated_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_perf_importance_urgency ON message_importance(urgency_score DESC) WHERE urgency_score > 0",
            
            # Composite index for the most common query pattern
            "CREATE INDEX IF NOT EXISTS idx_perf_messages_server_channel_time_score ON messages(server_id, channel_id, sent_at DESC) WHERE id IN (SELECT message_id FROM message_importance WHERE importance_score >= 0.3)",
            
            # Pattern detection indexes
            "CREATE INDEX IF NOT EXISTS idx_perf_patterns_active_time ON detected_patterns(status, detected_at DESC) WHERE status = 'active'",
            "CREATE INDEX IF NOT EXISTS idx_perf_patterns_type_confidence ON detected_patterns(pattern_type, confidence DESC)",
            
            # Author-based queries for tracking specific users
            "CREATE INDEX IF NOT EXISTS idx_perf_messages_author_time ON messages(author_id, sent_at DESC) WHERE author_id IS NOT NULL",
            
            # Reply chain analysis
            "CREATE INDEX IF NOT EXISTS idx_perf_messages_reply_chain ON messages(reply_to_id) WHERE reply_to_id IS NOT NULL",
            
            # Cache optimization
            "CREATE INDEX IF NOT EXISTS idx_perf_cache_expires ON query_cache(expires_at) WHERE expires_at > strftime('%s', 'now')",
        ]
        
        created_count = 0
        for index_sql in performance_indexes:
            try:
                cursor.execute(index_sql)
                created_count += 1
                logger.debug(f"Created performance index: {index_sql.split('idx_perf_')[1].split(' ')[0]}")
            except Exception as e:
                logger.warning(f"Failed to create performance index: {e}")
        
        conn.commit()
        logger.info(f"Created {created_count} performance indexes")
        
        return created_count

# Utility functions for real-time processing

def setup_wal_checkpoint_optimization(db_path: str):
    """Setup WAL checkpointing for real-time processing"""
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Configure WAL mode
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Set checkpoint threshold (auto-checkpoint after 1000 pages)
        conn.execute("PRAGMA wal_autocheckpoint = 1000")
        
        # Optimize WAL checkpoint mode for concurrent access
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        
        logger.info("WAL checkpoint optimization configured")
        
    except Exception as e:
        logger.error(f"Error configuring WAL optimization: {e}")
    finally:
        conn.close()

def create_materialized_views(conn: sqlite3.Connection):
    """Create materialized views for common queries"""
    
    cursor = conn.cursor()
    
    # Create a table that acts as a materialized view for high-importance recent messages
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mv_recent_important_messages AS
    SELECT 
        m.id,
        m.server_id,
        m.channel_id,
        m.content,
        m.sent_at,
        m.author_name,
        mi.importance_score,
        s.name as server_name,
        c.name as channel_name,
        strftime('%s', 'now') as cached_at
    FROM messages m
    JOIN message_importance mi ON m.id = mi.message_id
    JOIN servers s ON m.server_id = s.id  
    JOIN channels c ON m.channel_id = c.id
    WHERE m.sent_at >= strftime('%s', 'now', '-24 hours')
      AND mi.importance_score >= 0.5
    ORDER BY mi.importance_score DESC, m.sent_at DESC
    LIMIT 100
    ''')
    
    # Index on the materialized view
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_mv_recent_score 
    ON mv_recent_important_messages(importance_score DESC, sent_at DESC)
    ''')
    
    conn.commit()
    logger.info("Created materialized view for recent important messages")

def setup_connection_pooling_simulation(db_path: str, pool_size: int = 5):
    """Simulate connection pooling by pre-creating optimized connections"""
    
    connections = []
    optimizer = SQLitePerformanceOptimizer(db_path)
    
    for i in range(pool_size):
        conn = optimizer.get_connection()
        connections.append(conn)
        
    logger.info(f"Created connection pool with {pool_size} connections")
    return connections

def benchmark_query_performance(conn: sqlite3.Connection, query: str, params: tuple = ()) -> Dict[str, Any]:
    """Benchmark query performance"""
    
    import time
    cursor = conn.cursor()
    
    # Get query plan
    explain_query = f"EXPLAIN QUERY PLAN {query}"
    cursor.execute(explain_query, params)
    query_plan = cursor.fetchall()
    
    # Time the query
    start_time = time.time()
    cursor.execute(query, params)
    results = cursor.fetchall()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return {
        'execution_time_seconds': execution_time,
        'result_count': len(results),
        'query_plan': query_plan,
        'performance_rating': 'fast' if execution_time < 0.1 else 'moderate' if execution_time < 1.0 else 'slow'
    }

# Example usage function
def run_performance_optimization(db_path: str):
    """Run complete performance optimization on database"""
    
    optimizer = SQLitePerformanceOptimizer(db_path)
    
    with optimizer.get_connection() as conn:
        # Analyze current performance
        logger.info("Analyzing current database performance...")
        metrics = optimizer.analyze_database_performance(conn)
        logger.info(f"Database metrics: {metrics}")
        
        # Get recommendations
        recommendations = optimizer.get_slow_queries(conn)
        if recommendations:
            logger.info("Performance recommendations:")
            for rec in recommendations:
                logger.info(f"  • {rec}")
        
        # Create performance indexes
        logger.info("Creating performance indexes...")
        index_count = optimizer.create_performance_indexes(conn)
        
        # Run maintenance
        logger.info("Running database maintenance...")
        optimizer.optimize_database_maintenance(conn)
        
        # Setup materialized views
        logger.info("Creating materialized views...")
        create_materialized_views(conn)
        
        logger.info(f"Performance optimization completed. Created {index_count} indexes.")

if __name__ == "__main__":
    # Example usage
    db_path = "data/db.sqlite"
    run_performance_optimization(db_path)