"""
Integration layer to work with existing codebase while providing optimized functionality
"""
import logging
from typing import List, Tuple, Optional, Dict
from .database_optimized import db

logger = logging.getLogger('discord_bot')

def init_db():
    """Initialize the optimized database (replaces original init_db)"""
    db.init_optimized_db()
    logger.info("Database initialized with optimization layer")

def save_server(server_id: int, name: str):
    """Save or update server information (compatible with original)"""
    conn = db.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT OR REPLACE INTO servers (id, name, last_activity)
        VALUES (?, ?, strftime('%s', 'now'))
        ''', (server_id, name))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving server {server_id} ({name}): {e}")

def save_channel(channel_id: int, server_id: int, name: str):
    """Save or update channel information (compatible with original)"""
    conn = db.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT OR REPLACE INTO channels (id, server_id, name, last_activity)
        VALUES (?, ?, ?, strftime('%s', 'now'))
        ''', (channel_id, server_id, name))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving channel {channel_id} ({name}) for server {server_id}: {e}")

def save_messages(messages: List[Tuple[int, int, int, str, int]]):
    """Save messages with intelligence processing (enhanced version)"""
    for message_data in messages:
        server_id, channel_id, message_id, content, sent_at = message_data
        try:
            # Use the intelligence-enhanced save method
            importance_score = db.save_message_with_intelligence(
                message_id=message_id,
                server_id=server_id,
                channel_id=channel_id,
                content=content,
                sent_at=sent_at,
                author_name=None,  # Could be enhanced to capture author info
                has_attachments=False,  # Could be enhanced
                reply_to_id=None  # Could be enhanced
            )
            
            if importance_score > 0.7:  # Log high-importance messages
                logger.info(f"High importance message detected (score: {importance_score:.2f}): {message_id}")
                
        except Exception as e:
            logger.error(f"Error saving message {message_id}: {e}")

def get_last_message_id(server_id: int, channel_id: int) -> Optional[int]:
    """Get last message ID (compatible with original)"""
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id FROM messages
        WHERE server_id = ? AND channel_id = ?
        ORDER BY sent_at DESC
        LIMIT 1
    ''', (server_id, channel_id))
    result = cursor.fetchone()
    return result[0] if result else None

def get_recent_messages(server_id: int, hours: int = 24, 
                       keywords: Optional[List[str]] = None, 
                       channel_id: Optional[int] = None) -> List[str]:
    """Enhanced version that leverages importance scoring and FTS"""
    
    # If keywords are provided, use semantic search
    if keywords:
        query = ' '.join(keywords)
        results = db.search_messages_semantic(
            query=query,
            hours=hours,
            server_id=server_id,
            limit=50
        )
        
        # Filter by channel if specified
        if channel_id:
            results = [r for r in results if r['channel_id'] == channel_id]
            
        messages = [r['content'] for r in results if r['content']]
        logger.info(f"Retrieved {len(messages)} messages via semantic search")
        return messages
    
    # Otherwise, get important messages from the time period
    importance_threshold = 0.3  # Lower threshold for broader results
    results = db.get_important_messages_since(
        hours=hours,
        importance_threshold=importance_threshold,
        server_id=server_id,
        limit=100
    )
    
    # Filter by channel if specified
    if channel_id:
        results = [r for r in results if r['channel_id'] == channel_id]
    
    messages = [r['content'] for r in results if r['content']]
    logger.info(f"Retrieved {len(messages)} important messages")
    return messages

def get_unique_server_ids() -> List[int]:
    """Get unique server IDs (compatible with original)"""
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT server_id FROM messages")
    result = cursor.fetchall()
    return [row[0] for row in result]

def get_servers():
    """Get all servers (compatible with original)"""
    conn = db.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, name FROM servers ORDER BY name')
        return cursor.fetchall()
    finally:
        pass  # Connection managed by db object

def get_channels(server_id: int):
    """Get all channels for a server (compatible with original)"""
    conn = db.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, name FROM channels WHERE server_id = ? ORDER BY name', (server_id,))
        return cursor.fetchall()
    finally:
        pass  # Connection managed by db object

# New enhanced functions for monitoring

def get_digest_summary(hours: int = 24, importance_threshold: float = 0.5) -> Dict:
    """Get a comprehensive digest of recent important activity"""
    return db.generate_digest(hours, importance_threshold)

def get_high_priority_alerts(hours: int = 6) -> List[Dict]:
    """Get high-priority messages that need immediate attention"""
    return db.get_important_messages_since(
        hours=hours,
        importance_threshold=0.8,  # Very high importance only
        limit=20
    )

def search_messages_intelligent(query: str, hours: int = 168, 
                              server_id: int = None) -> List[Dict]:
    """Intelligent search combining FTS and importance scoring"""
    return db.search_messages_semantic(query, hours, server_id)

def detect_group_activities(server_id: int, hours: int = 24) -> List[Dict]:
    """Detect group buy and activity patterns"""
    patterns = db.detect_group_buy_pattern(server_id, hours)
    
    # Save detected patterns
    for pattern in patterns:
        if pattern.confidence > 0.6:  # Only save high-confidence patterns
            db.save_detected_pattern(pattern, server_id, pattern.data.get('channel_id', 0))
    
    return [
        {
            'type': p.pattern_type,
            'confidence': p.confidence,
            'data': p.data,
            'message_count': len(p.related_messages)
        }
        for p in patterns
    ]

def run_background_processing(server_id: int = None):
    """Run background processing for pattern detection and optimization"""
    try:
        # Detect patterns for all servers or specific server
        if server_id:
            server_ids = [server_id]
        else:
            server_ids = get_unique_server_ids()
        
        total_patterns = 0
        for sid in server_ids:
            patterns = detect_group_activities(sid, hours=24)
            total_patterns += len(patterns)
        
        # Run database optimization
        db.optimize_database()
        
        logger.info(f"Background processing completed: {total_patterns} patterns detected")
        return total_patterns
        
    except Exception as e:
        logger.error(f"Error in background processing: {e}")
        return 0

def get_monitoring_stats() -> Dict:
    """Get statistics for monitoring dashboard"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        # Message counts by importance
        cursor.execute('''
        SELECT 
            CASE 
                WHEN importance_score >= 0.8 THEN 'Critical'
                WHEN importance_score >= 0.6 THEN 'High'
                WHEN importance_score >= 0.4 THEN 'Medium'
                ELSE 'Low'
            END as importance_level,
            COUNT(*) as count
        FROM message_importance
        GROUP BY importance_level
        ORDER BY importance_score DESC
        ''')
        importance_stats = dict(cursor.fetchall())
        
        # Recent activity (last 24 hours)
        cursor.execute('''
        SELECT COUNT(*) 
        FROM messages 
        WHERE sent_at >= strftime('%s', 'now', '-24 hours')
        ''')
        recent_messages = cursor.fetchone()[0]
        
        # Active patterns
        cursor.execute('''
        SELECT pattern_type, COUNT(*) 
        FROM detected_patterns 
        WHERE status = 'active' 
          AND detected_at >= strftime('%s', 'now', '-7 days')
        GROUP BY pattern_type
        ''')
        active_patterns = dict(cursor.fetchall())
        
        # Top servers by activity
        cursor.execute('''
        SELECT s.name, COUNT(m.id) as message_count
        FROM servers s
        JOIN messages m ON s.id = m.server_id
        WHERE m.sent_at >= strftime('%s', 'now', '-24 hours')
        GROUP BY s.id, s.name
        ORDER BY message_count DESC
        LIMIT 5
        ''')
        top_servers = cursor.fetchall()
        
        return {
            'importance_distribution': importance_stats,
            'recent_messages_24h': recent_messages,
            'active_patterns': active_patterns,
            'top_servers': dict(top_servers),
            'last_updated': db.get_connection().execute('SELECT strftime("%s", "now")').fetchone()[0]
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e}")
        return {}

# Cleanup function
def cleanup_and_optimize():
    """Run cleanup and optimization tasks"""
    db.optimize_database()
    logger.info("Database cleanup and optimization completed")