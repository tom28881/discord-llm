"""
Enhanced Database Functions for Predictive Analytics

Extended database operations optimized for predictive analytics queries
and pattern detection in Discord message data.

Author: Senior Backend Architect
"""

import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import logging

from .database import DB_NAME

logger = logging.getLogger(__name__)

class PredictiveDatabase:
    """Enhanced database operations for predictive analytics"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_NAME
        self._ensure_predictive_tables()
    
    def _ensure_predictive_tables(self):
        """Create additional tables for predictive analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for storing prediction results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_id INTEGER NOT NULL,
            prediction_type TEXT NOT NULL,
            prediction_data TEXT NOT NULL,  -- JSON blob
            confidence REAL NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER,
            FOREIGN KEY (server_id) REFERENCES servers(id)
        )
        ''')
        
        # Table for storing user behavior patterns
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_id INTEGER NOT NULL,
            pattern_type TEXT NOT NULL,
            pattern_data TEXT NOT NULL,  -- JSON blob
            last_updated INTEGER NOT NULL,
            FOREIGN KEY (server_id) REFERENCES servers(id)
        )
        ''')
        
        # Table for storing event history and patterns
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            server_id INTEGER NOT NULL,
            channel_id INTEGER,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,  -- JSON blob
            detected_at INTEGER NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (server_id) REFERENCES servers(id),
            FOREIGN KEY (channel_id) REFERENCES channels(id)
        )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_server_type ON predictions(server_id, prediction_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_patterns_server ON user_patterns(server_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_server_time ON detected_events(server_id, detected_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_server_time ON messages(server_id, sent_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_channel_time ON messages(channel_id, sent_at)')
        
        conn.commit()
        conn.close()
        logger.info("Predictive analytics tables initialized")
    
    def get_message_analytics(self, server_id: int, days: int = 30) -> pd.DataFrame:
        """Get comprehensive message analytics for a server"""
        conn = sqlite3.connect(self.db_path)
        
        time_threshold = int((datetime.now() - timedelta(days=days)).timestamp())
        
        query = '''
        SELECT 
            m.id as message_id,
            m.server_id,
            m.channel_id,
            m.content,
            m.sent_at,
            c.name as channel_name,
            s.name as server_name,
            -- Time-based features
            datetime(m.sent_at, 'unixepoch') as timestamp,
            strftime('%H', datetime(m.sent_at, 'unixepoch')) as hour,
            strftime('%w', datetime(m.sent_at, 'unixepoch')) as day_of_week,
            strftime('%d', datetime(m.sent_at, 'unixepoch')) as day_of_month,
            strftime('%m', datetime(m.sent_at, 'unixepoch')) as month,
            strftime('%Y-%W', datetime(m.sent_at, 'unixepoch')) as year_week,
            -- Message features
            length(m.content) as message_length,
            (length(m.content) - length(replace(m.content, ' ', '')) + 1) as word_count,
            CASE WHEN m.content LIKE '%http%://%' OR m.content LIKE '%https%://%' THEN 1 ELSE 0 END as has_url,
            CASE WHEN m.content LIKE '%@%' THEN 1 ELSE 0 END as has_mention,
            (length(m.content) - length(replace(m.content, '!', ''))) as exclamation_count,
            (length(m.content) - length(replace(m.content, '?', ''))) as question_count
        FROM messages m
        JOIN channels c ON m.channel_id = c.id
        JOIN servers s ON m.server_id = s.id
        WHERE m.server_id = ? AND m.sent_at >= ?
        ORDER BY m.sent_at ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=[server_id, time_threshold])
        conn.close()
        
        if not df.empty:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['hour'].astype(int)
            df['day_of_week'] = df['day_of_week'].astype(int)
            df['day_of_month'] = df['day_of_month'].astype(int)
            df['month'] = df['month'].astype(int)
            
        return df
    
    def get_temporal_patterns(self, server_id: int, days: int = 60) -> Dict[str, Any]:
        """Analyze temporal patterns in message activity"""
        conn = sqlite3.connect(self.db_path)
        
        time_threshold = int((datetime.now() - timedelta(days=days)).timestamp())
        
        # Hourly patterns
        hourly_query = '''
        SELECT 
            strftime('%H', datetime(sent_at, 'unixepoch')) as hour,
            COUNT(*) as message_count,
            AVG(length(content)) as avg_length
        FROM messages 
        WHERE server_id = ? AND sent_at >= ?
        GROUP BY strftime('%H', datetime(sent_at, 'unixepoch'))
        ORDER BY hour
        '''
        
        hourly_df = pd.read_sql_query(hourly_query, conn, params=[server_id, time_threshold])
        
        # Daily patterns
        daily_query = '''
        SELECT 
            strftime('%w', datetime(sent_at, 'unixepoch')) as day_of_week,
            COUNT(*) as message_count,
            AVG(length(content)) as avg_length
        FROM messages 
        WHERE server_id = ? AND sent_at >= ?
        GROUP BY strftime('%w', datetime(sent_at, 'unixepoch'))
        ORDER BY day_of_week
        '''
        
        daily_df = pd.read_sql_query(daily_query, conn, params=[server_id, time_threshold])
        
        # Weekly patterns
        weekly_query = '''
        SELECT 
            strftime('%Y-%W', datetime(sent_at, 'unixepoch')) as year_week,
            COUNT(*) as message_count,
            AVG(length(content)) as avg_length
        FROM messages 
        WHERE server_id = ? AND sent_at >= ?
        GROUP BY strftime('%Y-%W', datetime(sent_at, 'unixepoch'))
        ORDER BY year_week
        '''
        
        weekly_df = pd.read_sql_query(weekly_query, conn, params=[server_id, time_threshold])
        
        conn.close()
        
        patterns = {
            'hourly': hourly_df.to_dict('records') if not hourly_df.empty else [],
            'daily': daily_df.to_dict('records') if not daily_df.empty else [],
            'weekly': weekly_df.to_dict('records') if not weekly_df.empty else []
        }
        
        # Calculate peak times
        if not hourly_df.empty:
            patterns['peak_hour'] = int(hourly_df.loc[hourly_df['message_count'].idxmax(), 'hour'])
            patterns['low_hour'] = int(hourly_df.loc[hourly_df['message_count'].idxmin(), 'hour'])
        
        if not daily_df.empty:
            patterns['peak_day'] = int(daily_df.loc[daily_df['message_count'].idxmax(), 'day_of_week'])
            patterns['low_day'] = int(daily_df.loc[daily_df['message_count'].idxmin(), 'day_of_week'])
        
        return patterns
    
    def get_keyword_trends(self, server_id: int, days: int = 30, min_occurrences: int = 5) -> List[Dict[str, Any]]:
        """Analyze trending keywords and topics"""
        conn = sqlite3.connect(self.db_path)
        
        time_threshold = int((datetime.now() - timedelta(days=days)).timestamp())
        
        # Get all messages for text analysis
        query = '''
        SELECT content, sent_at, channel_id
        FROM messages 
        WHERE server_id = ? AND sent_at >= ? AND content IS NOT NULL AND content != ''
        ORDER BY sent_at ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=[server_id, time_threshold])
        conn.close()
        
        if df.empty:
            return []
        
        # Simple keyword extraction (can be enhanced with NLP)
        import re
        from collections import Counter, defaultdict
        
        # Split time into windows
        df['timestamp'] = pd.to_datetime(df['sent_at'], unit='s')
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        keyword_trends = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            content = row['content'].lower()
            words = re.findall(r'\b\w{4,}\b', content)  # Words with 4+ characters
            week = row['week']
            
            for word in words:
                if len(word) > 15:  # Skip very long words
                    continue
                keyword_trends[word][week] += 1
        
        # Calculate trends
        trends = []
        current_week = datetime.now().isocalendar().week
        
        for keyword, weekly_counts in keyword_trends.items():
            total_occurrences = sum(weekly_counts.values())
            if total_occurrences < min_occurrences:
                continue
            
            weeks = sorted(weekly_counts.keys())
            if len(weeks) < 2:
                continue
            
            # Calculate trend (recent vs older weeks)
            recent_weeks = [w for w in weeks if w >= current_week - 2]
            older_weeks = [w for w in weeks if w < current_week - 2]
            
            recent_count = sum(weekly_counts[w] for w in recent_weeks) if recent_weeks else 0
            older_count = sum(weekly_counts[w] for w in older_weeks) if older_weeks else 0
            
            # Calculate trend score
            if older_count > 0:
                trend_score = (recent_count - older_count) / older_count
            else:
                trend_score = 1.0 if recent_count > 0 else 0.0
            
            trends.append({
                'keyword': keyword,
                'total_occurrences': total_occurrences,
                'recent_count': recent_count,
                'trend_score': trend_score,
                'weekly_distribution': dict(weekly_counts)
            })
        
        # Sort by trend score
        trends.sort(key=lambda x: x['trend_score'], reverse=True)
        
        return trends[:20]  # Top 20 trending keywords
    
    def get_channel_activity_comparison(self, server_id: int, days: int = 30) -> Dict[str, Any]:
        """Compare activity levels across channels"""
        conn = sqlite3.connect(self.db_path)
        
        time_threshold = int((datetime.now() - timedelta(days=days)).timestamp())
        
        query = '''
        SELECT 
            c.id as channel_id,
            c.name as channel_name,
            COUNT(m.id) as message_count,
            AVG(length(m.content)) as avg_message_length,
            COUNT(DISTINCT strftime('%Y-%m-%d', datetime(m.sent_at, 'unixepoch'))) as active_days,
            MIN(m.sent_at) as first_message,
            MAX(m.sent_at) as last_message
        FROM channels c
        LEFT JOIN messages m ON c.id = m.channel_id AND m.sent_at >= ?
        WHERE c.server_id = ?
        GROUP BY c.id, c.name
        ORDER BY message_count DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=[time_threshold, server_id])
        conn.close()
        
        if df.empty:
            return {'channels': [], 'summary': {}}
        
        # Calculate activity scores
        df['messages_per_day'] = df['message_count'] / df['active_days'].replace(0, 1)
        df['activity_score'] = (
            df['message_count'] / df['message_count'].max() * 0.6 +
            df['messages_per_day'] / df['messages_per_day'].max() * 0.4
        )
        
        channels = df.to_dict('records')
        
        summary = {
            'total_channels': len(channels),
            'active_channels': len(df[df['message_count'] > 0]),
            'most_active_channel': channels[0]['channel_name'] if channels else None,
            'total_messages': int(df['message_count'].sum()),
            'avg_messages_per_channel': float(df['message_count'].mean())
        }
        
        return {
            'channels': channels,
            'summary': summary
        }
    
    def store_prediction(self, server_id: int, prediction_type: str, 
                        prediction_data: Dict[str, Any], confidence: float, 
                        expires_in_hours: int = 24):
        """Store prediction results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        import json
        
        expires_at = int((datetime.now() + timedelta(hours=expires_in_hours)).timestamp())
        created_at = int(datetime.now().timestamp())
        
        cursor.execute('''
        INSERT INTO predictions (server_id, prediction_type, prediction_data, confidence, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (server_id, prediction_type, json.dumps(prediction_data), confidence, created_at, expires_at))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {prediction_type} prediction for server {server_id}")
    
    def get_predictions(self, server_id: int, prediction_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve stored predictions"""
        conn = sqlite3.connect(self.db_path)
        
        current_time = int(datetime.now().timestamp())
        
        query = '''
        SELECT prediction_type, prediction_data, confidence, created_at
        FROM predictions 
        WHERE server_id = ? AND (expires_at IS NULL OR expires_at > ?)
        '''
        params = [server_id, current_time]
        
        if prediction_type:
            query += ' AND prediction_type = ?'
            params.append(prediction_type)
        
        query += ' ORDER BY created_at DESC'
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        import json
        
        predictions = []
        for row in results:
            predictions.append({
                'type': row[0],
                'data': json.loads(row[1]),
                'confidence': row[2],
                'created_at': datetime.fromtimestamp(row[3])
            })
        
        return predictions
    
    def cleanup_expired_predictions(self):
        """Remove expired predictions from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = int(datetime.now().timestamp())
        
        cursor.execute('DELETE FROM predictions WHERE expires_at IS NOT NULL AND expires_at <= ?', (current_time,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired predictions")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Basic table counts
        tables = ['servers', 'channels', 'messages', 'predictions', 'user_patterns', 'detected_events']
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            stats[f'{table}_count'] = cursor.fetchone()[0]
        
        # Message date range
        cursor.execute('SELECT MIN(sent_at), MAX(sent_at) FROM messages')
        min_time, max_time = cursor.fetchone()
        if min_time and max_time:
            stats['message_date_range'] = {
                'earliest': datetime.fromtimestamp(min_time).isoformat(),
                'latest': datetime.fromtimestamp(max_time).isoformat(),
                'days_covered': (max_time - min_time) / 86400
            }
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_size = cursor.fetchone()[0]
        stats['database_size_mb'] = db_size / (1024 * 1024)
        
        conn.close()
        
        return stats


# Utility functions
def get_predictive_db() -> PredictiveDatabase:
    """Get PredictiveDatabase instance"""
    return PredictiveDatabase()

def initialize_predictive_db(db_path: str = None) -> PredictiveDatabase:
    """Initialize predictive database with all required tables"""
    db = PredictiveDatabase(db_path)
    logger.info("Predictive database initialized")
    return db