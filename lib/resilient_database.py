"""
Enhanced database layer with comprehensive error handling and recovery.

Implements connection pooling, transaction management, deadlock detection,
corruption recovery, and automatic backup strategies for reliable data operations.
"""

import sqlite3
import logging
import os
import shutil
import time
import threading
import queue
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

from .exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseLockError,
    DatabaseIntegrityError
)
from .resilience import resilient_database_call

logger = logging.getLogger('discord_bot.database')


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_path: str
    connection_timeout: float = 30.0
    max_retries: int = 3
    backup_interval_hours: int = 6
    max_backups: int = 48  # Keep 2 days of backups
    wal_mode: bool = True
    synchronous: str = "NORMAL"  # OFF, NORMAL, FULL
    journal_size_limit: int = 100 * 1024 * 1024  # 100MB
    auto_vacuum: str = "INCREMENTAL"  # NONE, FULL, INCREMENTAL
    page_size: int = 4096


class ConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, db_path: str, max_connections: int = 10, timeout: float = 30.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self.pool = queue.Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        
        # Pre-populate pool
        for _ in range(min(3, max_connections)):
            self._create_connection()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=False
            )
            
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            
            # Foreign key support
            conn.execute("PRAGMA foreign_keys=ON")
            
            conn.commit()
            return conn
            
        except sqlite3.Error as e:
            raise DatabaseConnectionError(f"Failed to create connection: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic return."""
        conn = None
        try:
            # Try to get connection from pool
            try:
                conn = self.pool.get(timeout=5.0)
            except queue.Empty:
                # Pool empty, create new connection if under limit
                with self.lock:
                    if self.active_connections < self.max_connections:
                        conn = self._create_connection()
                        self.active_connections += 1
                    else:
                        # Wait for connection to become available
                        conn = self.pool.get(timeout=self.timeout)
            
            # Test connection
            try:
                conn.execute("SELECT 1").fetchone()
            except sqlite3.Error:
                # Connection is stale, create new one
                conn.close()
                conn = self._create_connection()
            
            yield conn
            
        finally:
            if conn:
                try:
                    # Return connection to pool
                    self.pool.put(conn, timeout=1.0)
                except queue.Full:
                    # Pool is full, close connection
                    conn.close()
                    with self.lock:
                        self.active_connections -= 1
    
    def close_all(self):
        """Close all connections in pool."""
        with self.lock:
            while not self.pool.empty():
                try:
                    conn = self.pool.get_nowait()
                    conn.close()
                except (queue.Empty, sqlite3.Error):
                    pass
            self.active_connections = 0


class TransactionManager:
    """Manages database transactions with retry and rollback logic."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.connection_pool = connection_pool
    
    @contextmanager
    def transaction(self, isolation_level: str = "DEFERRED"):
        """Execute operations within a transaction."""
        with self.connection_pool.get_connection() as conn:
            original_isolation = conn.isolation_level
            savepoint_name = f"sp_{int(time.time() * 1000000)}"
            
            try:
                # Set isolation level
                conn.isolation_level = isolation_level
                
                # Create savepoint
                conn.execute(f"SAVEPOINT {savepoint_name}")
                
                yield conn
                
                # Commit transaction
                conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                conn.commit()
                
            except sqlite3.Error as e:
                # Rollback to savepoint
                try:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                except sqlite3.Error:
                    pass  # Rollback might fail if connection is broken
                
                # Classify error
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    raise DatabaseLockError(f"Database locked: {e}")
                elif "constraint" in str(e).lower() or "unique" in str(e).lower():
                    raise DatabaseIntegrityError(f"Integrity constraint: {e}")
                else:
                    raise DatabaseError(f"Transaction failed: {e}")
                    
            finally:
                # Restore original isolation level
                conn.isolation_level = original_isolation


class BackupManager:
    """Handles database backup and recovery operations."""
    
    def __init__(self, db_path: str, backup_dir: str = None):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir) if backup_dir else self.db_path.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_backup(self, suffix: str = None) -> Path:
        """Create database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{suffix}" if suffix else ""
        backup_name = f"db_backup_{timestamp}{suffix}.sqlite"
        backup_path = self.backup_dir / backup_name
        
        try:
            # Use SQLite backup API for consistency
            source_conn = sqlite3.connect(str(self.db_path))
            backup_conn = sqlite3.connect(str(backup_path))
            
            source_conn.backup(backup_conn)
            
            source_conn.close()
            backup_conn.close()
            
            # Create metadata file
            metadata = {
                'original_path': str(self.db_path),
                'backup_time': datetime.now().isoformat(),
                'size_bytes': backup_path.stat().st_size
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            # Clean up partial backup
            if backup_path.exists():
                backup_path.unlink()
            raise DatabaseError(f"Backup failed: {e}")
    
    def restore_backup(self, backup_path: Path) -> bool:
        """Restore database from backup."""
        try:
            if not backup_path.exists():
                raise DatabaseError(f"Backup file not found: {backup_path}")
            
            # Create safety backup of current database
            safety_backup = self.create_backup("pre_restore")
            
            # Stop any active connections
            # (In practice, you'd want to coordinate this with the application)
            
            # Replace database file
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise DatabaseError(f"Restore failed: {e}")
    
    def cleanup_old_backups(self, max_age_hours: int = 48):
        """Remove old backup files."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        for backup_file in self.backup_dir.glob("db_backup_*.sqlite"):
            try:
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_time:
                    backup_file.unlink()
                    # Also remove metadata file
                    metadata_file = backup_file.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove old backup {backup_file}: {e}")
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old backup files")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups with metadata."""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("db_backup_*.sqlite")):
            metadata_file = backup_file.with_suffix('.json')
            
            backup_info = {
                'path': str(backup_file),
                'size_bytes': backup_file.stat().st_size,
                'created': datetime.fromtimestamp(backup_file.stat().st_ctime).isoformat()
            }
            
            # Add metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    backup_info.update(metadata)
                except Exception:
                    pass
            
            backups.append(backup_info)
        
        return backups


class HealthMonitor:
    """Monitors database health and performs maintenance."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.connection_pool = connection_pool
        
    def check_integrity(self) -> Dict[str, Any]:
        """Check database integrity."""
        with self.connection_pool.get_connection() as conn:
            try:
                # PRAGMA integrity_check
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                
                # PRAGMA foreign_key_check
                cursor = conn.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                
                # Get database stats
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                size_bytes = page_count * page_size
                
                return {
                    'integrity_ok': integrity_result == 'ok',
                    'integrity_result': integrity_result,
                    'foreign_key_violations': len(fk_violations),
                    'fk_violations_details': fk_violations,
                    'size_bytes': size_bytes,
                    'page_count': page_count,
                    'page_size': page_size
                }
                
            except sqlite3.Error as e:
                return {
                    'integrity_ok': False,
                    'error': str(e)
                }
    
    def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance."""
        with self.connection_pool.get_connection() as conn:
            try:
                # VACUUM to reclaim space
                conn.execute("VACUUM")
                
                # ANALYZE to update statistics
                conn.execute("ANALYZE")
                
                # Incremental vacuum for WAL
                conn.execute("PRAGMA wal_checkpoint(FULL)")
                
                return {
                    'optimized': True,
                    'timestamp': datetime.now().isoformat()
                }
                
            except sqlite3.Error as e:
                return {
                    'optimized': False,
                    'error': str(e)
                }


class EnhancedDatabase:
    """Enhanced database interface with comprehensive error handling."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize components
        self.connection_pool = ConnectionPool(
            str(self.db_path),
            max_connections=10,
            timeout=config.connection_timeout
        )
        self.transaction_manager = TransactionManager(self.connection_pool)
        self.backup_manager = BackupManager(str(self.db_path))
        self.health_monitor = HealthMonitor(self.connection_pool)
        
        # Background maintenance
        self.maintenance_thread = None
        self.maintenance_running = False
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema and settings."""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS servers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY,
                server_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (server_id) REFERENCES servers(id)
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                server_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                content TEXT,
                sent_at INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (server_id) REFERENCES servers(id),
                FOREIGN KEY (channel_id) REFERENCES channels(id)
            )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_server_sent_at 
            ON messages(server_id, sent_at)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_channel_sent_at 
            ON messages(channel_id, sent_at)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_content 
            ON messages(content) WHERE content IS NOT NULL
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @resilient_database_call
    def save_server(self, server_id: int, name: str) -> bool:
        """Save or update server information."""
        try:
            with self.transaction_manager.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO servers (id, name, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (server_id, name))
                
            return True
            
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to save server {server_id}: {e}")
    
    @resilient_database_call
    def save_channel(self, channel_id: int, server_id: int, name: str) -> bool:
        """Save or update channel information."""
        try:
            with self.transaction_manager.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO channels (id, server_id, name, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (channel_id, server_id, name))
                
            return True
            
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to save channel {channel_id}: {e}")
    
    @resilient_database_call
    def save_messages(self, messages: List[Tuple[int, int, int, str, int]]) -> int:
        """Save multiple messages in a single transaction."""
        if not messages:
            return 0
        
        try:
            with self.transaction_manager.transaction() as conn:
                cursor = conn.cursor()
                
                saved_count = 0
                for server_id, channel_id, message_id, content, sent_at in messages:
                    try:
                        cursor.execute('''
                        INSERT OR IGNORE INTO messages (id, server_id, channel_id, content, sent_at)
                        VALUES (?, ?, ?, ?, ?)
                        ''', (message_id, server_id, channel_id, content, sent_at))
                        
                        if cursor.rowcount > 0:
                            saved_count += 1
                            
                    except sqlite3.Error as e:
                        logger.warning(f"Failed to save message {message_id}: {e}")
                        continue
                
                logger.info(f"Saved {saved_count}/{len(messages)} messages")
                return saved_count
                
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to save messages batch: {e}")
    
    @resilient_database_call
    def get_last_message_id(self, server_id: int, channel_id: int) -> Optional[int]:
        """Get the ID of the last message in a channel."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT id FROM messages
                WHERE server_id = ? AND channel_id = ?
                ORDER BY sent_at DESC, id DESC
                LIMIT 1
                ''', (server_id, channel_id))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get last message ID: {e}")
    
    @resilient_database_call
    def get_recent_messages(self, server_id: int, hours: int = 24, 
                          keywords: Optional[List[str]] = None, 
                          channel_id: Optional[int] = None) -> List[str]:
        """Get recent messages with optional filtering."""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate timestamp threshold
                time_threshold = int((datetime.now() - timedelta(hours=hours)).timestamp())
                
                # Build query
                query = """
                SELECT content FROM messages
                WHERE server_id = ? AND sent_at >= ? AND content IS NOT NULL
                """
                params = [server_id, time_threshold]
                
                # Add channel filter
                if channel_id:
                    query += " AND channel_id = ?"
                    params.append(channel_id)
                
                # Add keyword filter
                if keywords:
                    keyword_conditions = " OR ".join(["content LIKE ? COLLATE NOCASE"] * len(keywords))
                    query += f" AND ({keyword_conditions})"
                    params.extend([f"%{keyword}%" for keyword in keywords])
                
                query += " ORDER BY sent_at ASC LIMIT 1000"  # Limit for performance
                
                cursor.execute(query, params)
                messages = cursor.fetchall()
                
                logger.debug(f"Retrieved {len(messages)} recent messages")
                return [msg[0] for msg in messages if msg[0] and msg[0].strip()]
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get recent messages: {e}")
    
    def get_servers(self) -> List[Tuple[int, str]]:
        """Get all servers."""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name FROM servers ORDER BY name')
            return cursor.fetchall()
    
    def get_channels(self, server_id: int) -> List[Tuple[int, str]]:
        """Get all channels for a server."""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, name FROM channels WHERE server_id = ? ORDER BY name',
                (server_id,)
            )
            return cursor.fetchall()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records
            cursor.execute("SELECT COUNT(*) FROM servers")
            stats['server_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM channels")
            stats['channel_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM messages")
            stats['message_count'] = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(sent_at), MAX(sent_at) FROM messages")
            min_ts, max_ts = cursor.fetchone()
            if min_ts and max_ts:
                stats['date_range'] = {
                    'earliest': datetime.fromtimestamp(min_ts).isoformat(),
                    'latest': datetime.fromtimestamp(max_ts).isoformat()
                }
            
            # Database health
            health = self.health_monitor.check_integrity()
            stats['health'] = health
            
            return stats
    
    def start_maintenance(self):
        """Start background maintenance thread."""
        if self.maintenance_running:
            return
        
        self.maintenance_running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        logger.info("Database maintenance thread started")
    
    def stop_maintenance(self):
        """Stop background maintenance."""
        self.maintenance_running = False
        if self.maintenance_thread:
            self.maintenance_thread.join()
        logger.info("Database maintenance thread stopped")
    
    def _maintenance_loop(self):
        """Background maintenance loop."""
        last_backup = datetime.now()
        last_cleanup = datetime.now()
        
        while self.maintenance_running:
            try:
                now = datetime.now()
                
                # Create backup if needed
                if (now - last_backup).total_seconds() > (self.config.backup_interval_hours * 3600):
                    try:
                        self.backup_manager.create_backup("scheduled")
                        last_backup = now
                    except Exception as e:
                        logger.error(f"Scheduled backup failed: {e}")
                
                # Clean up old backups daily
                if (now - last_cleanup).total_seconds() > 86400:  # 24 hours
                    try:
                        self.backup_manager.cleanup_old_backups(self.config.max_backups)
                        last_cleanup = now
                    except Exception as e:
                        logger.error(f"Backup cleanup failed: {e}")
                
                # Sleep for 1 hour
                for _ in range(3600):
                    if not self.maintenance_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def close(self):
        """Close database and cleanup resources."""
        self.stop_maintenance()
        self.connection_pool.close_all()
        logger.info("Database closed")