import sqlite3
import logging
from typing import List, Tuple, Optional, Iterable, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger('discord_bot')
# Get the project root directory (parent of lib directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DB_NAME = DATA_DIR / 'db.sqlite'

def _messages_schema_requires_migration(cursor: sqlite3.Cursor) -> bool:
    """Detect whether the messages table needs to be recreated for new schema."""
    cursor.execute("PRAGMA table_info(messages)")
    columns = cursor.fetchall()
    if not columns:
        return False

    pk_columns = sorted(((row[5], row[1]) for row in columns if row[5]), key=lambda item: item[0])
    primary_keys = [name for _seq, name in pk_columns]

    if primary_keys != ["id", "channel_id"]:
        return True

    return False


def _migrate_messages_table(cursor: sqlite3.Cursor) -> None:
    """Migrate existing messages table to the new composite primary key schema."""
    logger.info("Renaming legacy messages table for schema migration.")
    cursor.execute("ALTER TABLE messages RENAME TO messages_old")
    _create_messages_table(cursor)

    logger.info("Copying data into new messages table with per-channel uniqueness.")
    cursor.execute('''
        INSERT OR IGNORE INTO messages (id, server_id, channel_id, content, sent_at)
        SELECT id, server_id, channel_id, content, sent_at FROM messages_old
    ''')

    logger.info("Dropping legacy messages table after successful migration.")
    cursor.execute("DROP TABLE messages_old")


def _create_messages_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER NOT NULL,
        server_id INTEGER NOT NULL,
        channel_id INTEGER NOT NULL,
        content TEXT,
        sent_at INTEGER,
        PRIMARY KEY (id, channel_id),
        FOREIGN KEY (server_id) REFERENCES servers(id),
        FOREIGN KEY (channel_id) REFERENCES channels(id)
    )
    ''')


def _get_connection() -> sqlite3.Connection:
    """Create a SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _deduplicate_messages(rows: List[Tuple[int, int, str]]) -> List[str]:
    """Return messages with unique message IDs preserving order across channels."""
    seen = set()
    deduped = []
    for channel_id, message_id, content in rows:
        if message_id in seen:
            continue
        seen.add(message_id)
        deduped.append(content)
    return deduped


def init_db():
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    conn = _get_connection()
    cursor = conn.cursor()
    
    # Create servers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS servers (
        id INTEGER PRIMARY KEY,  -- This will be the Discord server ID
        name TEXT NOT NULL
    )
    ''')
    
    # Create channels table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS channels (
        id INTEGER PRIMARY KEY,  -- This will be the Discord channel ID
        server_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        FOREIGN KEY (server_id) REFERENCES servers(id)
    )
    ''')
    
    # Ensure messages table uses per-channel primary key to avoid cross-channel conflicts
    if _messages_schema_requires_migration(cursor):
        logger.info("Migrating messages table schema for per-channel message uniqueness.")
        _migrate_messages_table(cursor)
    else:
        _create_messages_table(cursor)
    
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized.")

def save_server(server_id: int, name: str):
    """Save or update server information."""
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT OR REPLACE INTO servers (id, name)
        VALUES (?, ?)
        ''', (server_id, name))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving server {server_id} ({name}): {e}")
    finally:
        conn.close()

def save_channel(channel_id: int, server_id: int, name: str):
    """Save or update channel information."""
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT OR REPLACE INTO channels (id, server_id, name)
        VALUES (?, ?, ?)
        ''', (channel_id, server_id, name))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving channel {channel_id} ({name}) for server {server_id}: {e}")
    finally:
        conn.close()


def _normalize_timestamp(value) -> int:
    """Convert various timestamp representations to unix epoch seconds."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, str):
        try:
            iso_value = value.replace('Z', '+00:00') if value.endswith('Z') else value
            return int(datetime.fromisoformat(iso_value).timestamp())
        except ValueError as exc:
            logger.error(f"Failed to parse timestamp string: {value}")
            raise exc
    raise TypeError(f"Unsupported timestamp type: {type(value)}")

def save_messages(messages: List[Tuple[int, int, int, str, int]]):
    """Save messages to the database."""
    conn = _get_connection()
    cursor = conn.cursor()

    try:
        for server_id, channel_id, message_id, content, sent_at in messages:
            try:
                cursor.execute('SELECT 1 FROM servers WHERE id = ?', (server_id,))
                server_exists = cursor.fetchone() is not None

                cursor.execute('SELECT 1 FROM channels WHERE id = ?', (channel_id,))
                channel_exists = cursor.fetchone() is not None

                if not server_exists or not channel_exists:
                    missing_parts = []
                    if not server_exists:
                        missing_parts.append('server')
                    if not channel_exists:
                        missing_parts.append('channel')
                    missing_str = ' and '.join(missing_parts)
                    logger.error(
                        "Cannot save message %s: missing %s reference (server_id=%s, channel_id=%s)",
                        message_id,
                        missing_str,
                        server_id,
                        channel_id,
                    )
                    continue

                normalized_sent_at = _normalize_timestamp(sent_at)
                cursor.execute(
                    '''
                    INSERT OR IGNORE INTO messages (id, server_id, channel_id, content, sent_at)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (message_id, server_id, channel_id, content, normalized_sent_at),
                )

                if cursor.rowcount == 0:
                    logger.warning(
                        "Message %s for server %s and channel %s was not saved (duplicate or constraint).",
                        message_id,
                        server_id,
                        channel_id,
                    )
            except sqlite3.IntegrityError as exc:
                logger.error("Error saving message %s: %s", message_id, exc)
                logger.error(
                    "Message data: server_id=%s, channel_id=%s, content_length=%s",
                    server_id,
                    channel_id,
                    len(content) if content else 0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error saving message %s: %s", message_id, exc)
                logger.error(
                    "Message data: server_id=%s, channel_id=%s, content_length=%s",
                    server_id,
                    channel_id,
                    len(content) if content else 0,
                )
        conn.commit()
    except Exception as exc:
        logger.error("Error saving messages: %s", exc)
    finally:
        conn.close()

def get_last_message_id(server_id: int, channel_id: int) -> Optional[int]:
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT id
        FROM messages
        WHERE server_id = ? AND channel_id = ?
        ORDER BY sent_at DESC
        LIMIT 1
        ''',
        (server_id, channel_id),
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_recent_message_records(
    server_id: int,
    hours: Optional[int] = 24,
    keywords: Optional[List[str]] = None,
    channel_ids: Optional[Iterable[int]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return recent messages with metadata for downstream consumers."""

    conn = _get_connection()
    cursor = conn.cursor()

    conditions = ["server_id = ?"]
    params: List[Any] = [server_id]

    if hours is not None:
        time_threshold = int((datetime.now() - timedelta(hours=hours)).timestamp())
        conditions.append("sent_at >= ?")
        params.append(time_threshold)

    channel_id_list: Optional[List[int]] = None
    if channel_ids is not None:
        channel_id_list = [int(ch_id) for ch_id in channel_ids]
        if channel_id_list:
            placeholders = ",".join(["?"] * len(channel_id_list))
            conditions.append(f"channel_id IN ({placeholders})")
            params.extend(channel_id_list)
        else:
            # Empty list means no channels match, short circuit
            return []

    if keywords:
        keyword_conditions = " OR ".join(["content LIKE ? COLLATE NOCASE"] * len(keywords))
        conditions.append(f"({keyword_conditions})")
        params.extend([f"%{keyword}%" for keyword in keywords])

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT id, channel_id, content, sent_at
        FROM messages
        WHERE {where_clause}
        ORDER BY sent_at DESC
    """

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    records = [
        {
            "id": row[0],
            "channel_id": row[1],
            "content": row[2],
            "sent_at": row[3],
        }
        for row in rows
    ]
    return records


def get_recent_messages(
    server_id: int,
    hours: int = 24,
    keywords: Optional[List[str]] = None,
    channel_id: Optional[int] = None,
    channel_ids: Optional[Iterable[int]] = None,
) -> List[str]:
    """Compatibility wrapper returning only message content strings."""

    effective_channel_ids: Optional[List[int]] = None
    if channel_id is not None and channel_ids is not None:
        combined = list(channel_ids) + [channel_id]
        # Deduplicate while preserving order
        seen: set[int] = set()
        effective_channel_ids = []
        for ch in combined:
            ch_int = int(ch)
            if ch_int not in seen:
                seen.add(ch_int)
                effective_channel_ids.append(ch_int)
    elif channel_id is not None:
        effective_channel_ids = [channel_id]
    elif channel_ids is not None:
        effective_channel_ids = [int(ch) for ch in channel_ids]

    log_channels = (
        f" channels {effective_channel_ids}"
        if effective_channel_ids
        else (f" channel {channel_id}" if channel_id else "")
    )
    logger.info(
        "Fetching messages from the last %s hours for server %s%s",
        hours,
        server_id,
        log_channels,
    )

    records = get_recent_message_records(
        server_id,
        hours=hours,
        keywords=keywords,
        channel_ids=effective_channel_ids,
    )

    rows = [(record["channel_id"], record["id"], record["content"]) for record in records]
    messages = _deduplicate_messages(rows)

    if not messages and hours is not None:
        logger.debug(
            "No messages found within %s hours for server %s; returning most recent messages without time filter.",
            hours,
            server_id,
        )
        fallback_records = get_recent_message_records(
            server_id,
            hours=None,
            keywords=keywords,
            channel_ids=effective_channel_ids,
        )
        rows = [
            (record["channel_id"], record["id"], record["content"]) for record in fallback_records
        ]
        messages = _deduplicate_messages(rows)

    logger.info("Retrieved %d messages", len(messages))
    return messages


def get_latest_message_timestamp(
    server_id: int,
    channel_ids: Optional[Iterable[int]] = None,
) -> Optional[int]:
    """Return the newest message timestamp for the given scope."""

    conn = _get_connection()
    cursor = conn.cursor()

    conditions = ["server_id = ?"]
    params: List[Any] = [server_id]

    if channel_ids is not None:
        channel_id_list = [int(ch) for ch in channel_ids]
        if channel_id_list:
            placeholders = ",".join(["?"] * len(channel_id_list))
            conditions.append(f"channel_id IN ({placeholders})")
            params.extend(channel_id_list)
        else:
            return None

    where_clause = " AND ".join(conditions)
    query = f"SELECT MAX(sent_at) FROM messages WHERE {where_clause}"

    cursor.execute(query, tuple(params))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result and result[0] is not None else None

def get_unique_server_ids() -> List[int]:
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM servers ORDER BY id")
    result = cursor.fetchall()
    conn.close()
    return [row[0] for row in result]

def get_servers():
    """Get all servers with their IDs and names."""
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, name FROM servers ORDER BY name')
        return cursor.fetchall()
    finally:
        conn.close()

def get_channels(server_id: int):
    """Get all channels for a specific server."""
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, name FROM channels WHERE server_id = ? ORDER BY name', (server_id,))
        return cursor.fetchall()
    finally:
        conn.close()
