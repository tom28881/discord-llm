"""
Conversation Threader Module for Discord Monitoring
Groups related messages into conversation threads for better context
"""

import sqlite3
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class ConversationThreader:
    """Groups related messages into conversation threads"""
    
    def __init__(self, db_path: str = 'data/db.sqlite'):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=self._get_stop_words()
        )
        
        # Thread detection parameters
        self.max_time_gap_minutes = 30  # Max gap between messages in same thread
        self.min_similarity = 0.3  # Minimum semantic similarity
        self.reply_patterns = [
            r'^@\w+',  # @mentions
            r'^\s*>',  # Discord reply format (quote)
            r'^\^',  # Reference to message above
            r'^this|^that|^it',  # Referential language
            r'^ano|^ne|^jo|^yes|^no',  # Direct responses
        ]
        
        self.compiled_reply_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.reply_patterns
        ]
    
    def _get_stop_words(self) -> List[str]:
        """Get Czech and English stop words"""
        return [
            # Czech stop words
            'a', 'aby', 'aj', 'ale', 'ani', 'ano', 'asi', 'až', 'bez', 'být',
            'byl', 'byla', 'bylo', 'byly', 'co', 'což', 'či', 'další', 'do',
            'ho', 'i', 'já', 'je', 'jeho', 'její', 'jejich', 'jen', 'ještě',
            'jí', 'již', 'jsem', 'jsi', 'jsme', 'jsou', 'jste', 'k', 'kam',
            'kde', 'kdo', 'když', 'ke', 'která', 'které', 'který', 'mezi',
            'mí', 'mně', 'mnou', 'mně', 'můj', 'může', 'my', 'na', 'nad',
            'nam', 'námi', 'nas', 'náš', 'naše', 'naši', 'ne', 'nebo', 'nejsou',
            'než', 'nic', 'nich', 'ním', 'no', 'nový', 'o', 'od', 'on', 'ona',
            'oni', 'ono', 'pak', 'po', 'pod', 'podle', 'pokud', 'pouze', 'právě',
            'pro', 'proč', 'proto', 'protože', 'před', 'přes', 'při', 'pta',
            're', 's', 'se', 'si', 'sice', 'jsou', 'svůj', 'svých', 'svým',
            'svými', 'ta', 'tak', 'také', 'tato', 'te', 'tedy', 'ten', 'tento',
            'teto', 'ti', 'tím', 'tímto', 'tipy', 'to', 'tohle', 'toho', 'tohoto',
            'tom', 'tomto', 'tomu', 'tomuto', 'tu', 'tuto', 'tvůj', 'ty', 'tyto',
            'u', 'už', 'v', 've', 'vedle', 'více', 'vlastně', 'vsak', 'vy',
            'vždy', 'z', 'za', 'zač', 'zatímco', 'ze', 'že', 'zpet',
            # English stop words
            'the', 'be', 'to', 'of', 'and', 'in', 'that', 'have', 'it',
            'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
            'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
            'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
            'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
            'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also'
        ]
    
    def thread_messages(self, messages: List[Dict[str, Any]], 
                       channel_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Group messages into conversation threads"""
        if not messages:
            return []
        
        # Sort messages by timestamp
        messages = sorted(messages, key=lambda m: m.get('sent_at', 0))
        
        # Extract features for clustering
        features = self._extract_features(messages)
        
        # Cluster messages
        threads = self._cluster_messages(messages, features)
        
        # Save threads to database
        for thread in threads:
            self._save_thread(thread, channel_id)
        
        return threads
    
    def _extract_features(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for message clustering"""
        # Get text content
        texts = [msg.get('content', '') for msg in messages]
        
        if not any(texts):
            return np.zeros((len(messages), 1))
        
        # TF-IDF features
        try:
            tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        except:
            # Fallback if vectorization fails
            tfidf_features = np.zeros((len(messages), 1))
        
        # Time features (normalized)
        timestamps = [msg.get('sent_at', 0) for msg in messages]
        if timestamps:
            min_time = min(timestamps)
            max_time = max(timestamps)
            if max_time > min_time:
                time_features = np.array([
                    [(t - min_time) / (max_time - min_time)] 
                    for t in timestamps
                ])
            else:
                time_features = np.zeros((len(messages), 1))
        else:
            time_features = np.zeros((len(messages), 1))
        
        # Reply pattern features
        reply_features = []
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            is_reply = 0
            
            # Check if message is a reply
            for pattern in self.compiled_reply_patterns:
                if pattern.match(content):
                    is_reply = 1
                    break
            
            # Check if message references previous message
            if i > 0 and not is_reply:
                prev_content = messages[i-1].get('content', '')
                # Simple check for topic continuity
                if any(word in content.lower() for word in prev_content.lower().split()[:5]):
                    is_reply = 0.5
            
            reply_features.append([is_reply])
        
        reply_features = np.array(reply_features)
        
        # Combine features
        if tfidf_features.shape[1] > 0:
            features = np.hstack([
                tfidf_features * 0.5,  # Semantic similarity weight
                time_features * 0.3,   # Time proximity weight
                reply_features * 0.2   # Reply pattern weight
            ])
        else:
            features = np.hstack([time_features, reply_features])
        
        return features
    
    def _cluster_messages(self, messages: List[Dict[str, Any]], 
                         features: np.ndarray) -> List[Dict[str, Any]]:
        """Cluster messages into threads using DBSCAN"""
        threads = []
        
        if len(messages) < 2:
            # Single message = single thread
            if messages:
                threads.append({
                    'thread_id': self._generate_thread_id(messages[0]),
                    'messages': messages,
                    'thread_type': 'single',
                    'start_time': messages[0].get('sent_at'),
                    'end_time': messages[0].get('sent_at'),
                    'participant_count': 1
                })
            return threads
        
        # Use time-based grouping with semantic similarity
        current_thread = []
        last_timestamp = 0
        
        for i, msg in enumerate(messages):
            timestamp = msg.get('sent_at', 0)
            
            # Check time gap
            time_gap_minutes = (timestamp - last_timestamp) / 60 if last_timestamp else 0
            
            # Check semantic similarity with current thread
            is_similar = False
            if current_thread and i > 0:
                # Simple similarity check based on common words
                current_words = set()
                for thread_msg in current_thread[-3:]:  # Check last 3 messages
                    current_words.update(thread_msg.get('content', '').lower().split())
                
                msg_words = set(msg.get('content', '').lower().split())
                common_words = current_words & msg_words
                
                # Remove stop words from comparison
                stop_words = set(self._get_stop_words())
                common_words = common_words - stop_words
                
                if len(common_words) >= 2:  # At least 2 common meaningful words
                    is_similar = True
            
            # Decide if message belongs to current thread
            if (current_thread and 
                time_gap_minutes <= self.max_time_gap_minutes and 
                (is_similar or time_gap_minutes <= 5)):  # Tighter time constraint without similarity
                current_thread.append(msg)
            else:
                # Save current thread and start new one
                if current_thread:
                    thread_type = self._determine_thread_type(current_thread)
                    threads.append({
                        'thread_id': self._generate_thread_id(current_thread[0]),
                        'messages': current_thread,
                        'thread_type': thread_type,
                        'start_time': current_thread[0].get('sent_at'),
                        'end_time': current_thread[-1].get('sent_at'),
                        'participant_count': len(set(m.get('author_id', m.get('id', i)) 
                                                    for i, m in enumerate(current_thread)))
                    })
                current_thread = [msg]
            
            last_timestamp = timestamp
        
        # Don't forget the last thread
        if current_thread:
            thread_type = self._determine_thread_type(current_thread)
            threads.append({
                'thread_id': self._generate_thread_id(current_thread[0]),
                'messages': current_thread,
                'thread_type': thread_type,
                'start_time': current_thread[0].get('sent_at'),
                'end_time': current_thread[-1].get('sent_at'),
                'participant_count': len(set(m.get('author_id', m.get('id', i)) 
                                            for i, m in enumerate(current_thread)))
            })
        
        return threads
    
    def _determine_thread_type(self, messages: List[Dict[str, Any]]) -> str:
        """Determine the type of conversation thread"""
        if len(messages) == 1:
            return 'single'
        elif len(messages) == 2:
            return 'exchange'
        elif len(messages) <= 5:
            return 'discussion'
        else:
            # Check for specific patterns
            all_text = ' '.join([m.get('content', '') for m in messages]).lower()
            
            if any(word in all_text for word in ['koupit', 'buy', 'nákup', 'purchase', 'cena', 'price']):
                return 'purchase_discussion'
            elif any(word in all_text for word in ['kdy', 'when', 'kde', 'where', 'meeting', 'event']):
                return 'planning'
            elif any(word in all_text for word in ['hlasovat', 'vote', 'poll', 'rozhodnout', 'decide']):
                return 'decision'
            elif '?' in all_text and len([m for m in messages if '?' in m.get('content', '')]) >= 2:
                return 'qa_session'
            else:
                return 'conversation'
    
    def _generate_thread_id(self, first_message: Dict[str, Any]) -> str:
        """Generate unique thread ID"""
        timestamp = first_message.get('sent_at', 0)
        message_id = first_message.get('id', 0)
        return f"thread_{timestamp}_{message_id}"
    
    def _save_thread(self, thread: Dict[str, Any], channel_id: Optional[int] = None):
        """Save thread to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            thread_id = thread['thread_id']
            
            # Save each message in the thread
            for i, msg in enumerate(thread['messages']):
                cursor.execute('''
                    INSERT OR REPLACE INTO conversation_threads 
                    (thread_id, message_id, channel_id, position, thread_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    thread_id,
                    msg.get('id'),
                    channel_id or msg.get('channel_id'),
                    i,
                    thread['thread_type'],
                    int(datetime.now().timestamp())
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving thread: {e}")
    
    def get_thread_by_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        """Get the full thread containing a specific message"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Find thread ID for this message
            cursor.execute('''
                SELECT thread_id FROM conversation_threads 
                WHERE message_id = ?
            ''', (message_id,))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return None
            
            thread_id = result['thread_id']
            
            # Get all messages in this thread
            cursor.execute('''
                SELECT ct.*, m.content, m.sent_at, m.author_name
                FROM conversation_threads ct
                JOIN messages m ON ct.message_id = m.id
                WHERE ct.thread_id = ?
                ORDER BY ct.position
            ''', (thread_id,))
            
            messages = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            if not messages:
                return None
            
            return {
                'thread_id': thread_id,
                'messages': messages,
                'message_count': len(messages),
                'thread_type': messages[0].get('thread_type') if messages else 'unknown',
                'start_time': messages[0].get('sent_at') if messages else None,
                'end_time': messages[-1].get('sent_at') if messages else None
            }
            
        except Exception as e:
            print(f"Error getting thread: {e}")
            return None
    
    def get_active_threads(self, channel_id: Optional[int] = None, 
                          hours: int = 24) -> List[Dict[str, Any]]:
        """Get recently active conversation threads"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            time_threshold = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            query = '''
                SELECT 
                    thread_id,
                    thread_type,
                    COUNT(*) as message_count,
                    MIN(created_at) as start_time,
                    MAX(created_at) as end_time
                FROM conversation_threads
                WHERE created_at >= ?
            '''
            params = [time_threshold]
            
            if channel_id:
                query += ' AND channel_id = ?'
                params.append(channel_id)
            
            query += '''
                GROUP BY thread_id
                HAVING message_count >= 3
                ORDER BY end_time DESC
                LIMIT 20
            '''
            
            cursor.execute(query, params)
            
            threads = []
            for row in cursor.fetchall():
                thread_data = dict(row)
                
                # Get sample messages from thread
                cursor.execute('''
                    SELECT m.content
                    FROM conversation_threads ct
                    JOIN messages m ON ct.message_id = m.id
                    WHERE ct.thread_id = ?
                    ORDER BY ct.position
                    LIMIT 3
                ''', (thread_data['thread_id'],))
                
                sample_messages = [r[0] for r in cursor.fetchall()]
                thread_data['preview'] = ' ... '.join(sample_messages)
                
                threads.append(thread_data)
            
            conn.close()
            return threads
            
        except Exception as e:
            print(f"Error getting active threads: {e}")
            return []
    
    def analyze_thread_patterns(self, channel_id: int, days: int = 7) -> Dict[str, Any]:
        """Analyze conversation patterns in a channel"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            time_threshold = int((datetime.now() - timedelta(days=days)).timestamp())
            
            # Get thread statistics
            cursor.execute('''
                SELECT 
                    thread_type,
                    COUNT(DISTINCT thread_id) as thread_count,
                    AVG(message_count) as avg_length,
                    MAX(message_count) as max_length
                FROM (
                    SELECT thread_id, thread_type, COUNT(*) as message_count
                    FROM conversation_threads
                    WHERE channel_id = ? AND created_at >= ?
                    GROUP BY thread_id
                )
                GROUP BY thread_type
            ''', (channel_id, time_threshold))
            
            patterns = {}
            for row in cursor.fetchall():
                patterns[row['thread_type']] = {
                    'count': row['thread_count'],
                    'avg_length': round(row['avg_length'], 1),
                    'max_length': row['max_length']
                }
            
            # Get peak conversation times
            cursor.execute('''
                SELECT 
                    strftime('%H', datetime(created_at, 'unixepoch', 'localtime')) as hour,
                    COUNT(DISTINCT thread_id) as thread_count
                FROM conversation_threads
                WHERE channel_id = ? AND created_at >= ?
                GROUP BY hour
                ORDER BY thread_count DESC
                LIMIT 3
            ''', (channel_id, time_threshold))
            
            peak_hours = [f"{row['hour']}:00" for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                'channel_id': channel_id,
                'thread_patterns': patterns,
                'peak_conversation_hours': peak_hours,
                'analysis_period_days': days
            }
            
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return {
                'channel_id': channel_id,
                'error': str(e)
            }