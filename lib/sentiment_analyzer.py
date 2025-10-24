"""
Sentiment Analyzer Module for Discord Monitoring
Analyzes group excitement and sentiment with Czech language support
"""

import re
import sqlite3
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from collections import Counter

class SentimentAnalyzer:
    """Analyzes sentiment and excitement levels in Discord messages"""
    
    def __init__(self, db_path: str = 'data/db.sqlite'):
        self.db_path = db_path
        
        # Czech and English sentiment patterns
        self.sentiment_patterns = {
            'positive_strong': {
                'patterns': [
                    r'\b(super|skvělé|skvělý|úžasné|úžasný|bomba|paráda|fantastické)\b',
                    r'\b(awesome|amazing|fantastic|excellent|perfect|great)\b',
                    r'\b(nejlepší|best|top|perfektní|dokonalé)\b',
                    r'\b(miluju|love|zbožňuju|adore)\b',
                ],
                'weight': 1.0,
                'excitement_boost': 0.3
            },
            'positive_moderate': {
                'patterns': [
                    r'\b(dobré|dobrý|fajn|ok|prima|pohoda)\b',
                    r'\b(good|nice|cool|fine|neat)\b',
                    r'\b(líbí|like|baví|enjoy)\b',
                    r'\b(zajímavé|interesting|zábavné|fun)\b',
                ],
                'weight': 0.5,
                'excitement_boost': 0.1
            },
            'negative_strong': {
                'patterns': [
                    r'\b(hrozné|hrozný|strašné|strašný|otřesné|děsné)\b',
                    r'\b(terrible|horrible|awful|disgusting|worst)\b',
                    r'\b(nenávidím|hate|nesnáším|detest)\b',
                    r'\b(katastrofa|disaster|průser|nightmare)\b',
                ],
                'weight': -1.0,
                'excitement_boost': 0.1  # Negative excitement still shows engagement
            },
            'negative_moderate': {
                'patterns': [
                    r'\b(špatné|špatný|blbé|blbý|divné)\b',
                    r'\b(bad|poor|weird|strange|wrong)\b',
                    r'\b(nelíbí|dislike|nezajímá|boring)\b',
                    r'\b(problém|problem|issue|chyba|error)\b',
                ],
                'weight': -0.5,
                'excitement_boost': 0.05
            },
            'excitement': {
                'patterns': [
                    r'!{2,}',  # Multiple exclamation marks
                    r'\b[A-Z]{3,}\b',  # CAPS WORDS
                    r'\b(wow|wau|jéé|hurá|yes|jo{2,}|jojo)\b',
                    r'\b(konečně|finally|už|already)\b',
                    r'\b(nemůžu\s+se\s+dočkat|can\'t\s+wait|těším\s+se)\b',
                ],
                'weight': 0,
                'excitement_boost': 0.4
            },
            'urgency': {
                'patterns': [
                    r'\b(rychle|quick|fast|hurry|pospěš)\b',
                    r'\b(hned|now|immediately|okamžitě)\b',
                    r'\b(musíme|must|have\s+to|need\s+to)\b',
                    r'\b(důležité|important|kritické|critical)\b',
                ],
                'weight': 0.1,
                'excitement_boost': 0.2
            },
            'agreement': {
                'patterns': [
                    r'\b(souhlasím|agree|ano|yes|jo|yeah|jj)\b',
                    r'\b(přesně|exactly|správně|right|pravda|true)\b',
                    r'\b(taky|také|too|also|me\s+too)\b',
                    r'\+1|\b👍\b',
                ],
                'weight': 0.3,
                'excitement_boost': 0.1
            },
            'disagreement': {
                'patterns': [
                    r'\b(nesouhlasím|disagree|ne|no|nope)\b',
                    r'\b(naopak|opposite|jinak|differently)\b',
                    r'\b(ale|but|však|however)\b',
                    r'\-1|\b👎\b',
                ],
                'weight': -0.2,
                'excitement_boost': 0.05
            }
        }
        
        # Emoji sentiment mappings
        self.emoji_sentiments = {
            'positive': ['😀', '😃', '😄', '😊', '😍', '🥰', '😘', '🤗', '🎉', '🎊', 
                        '✨', '💖', '💕', '❤️', '🔥', '👍', '👏', '🙌', '💪', '🚀'],
            'negative': ['😢', '😭', '😞', '😔', '😟', '😕', '🙁', '☹️', '😣', '😖',
                        '😫', '😩', '😤', '😠', '😡', '🤬', '💔', '👎', '😰', '😨'],
            'excitement': ['🎉', '🎊', '🔥', '💥', '⚡', '🚀', '✨', '🌟', '💫', '🎯',
                          '🏆', '🥳', '🤩', '😱', '🤯', '😵', '🤪', '🤠', '🥵', '🔥'],
            'neutral': ['😐', '😑', '🤔', '🤷', '😶', '🙄', '😏', '😌', '😴', '💤']
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for category, data in self.sentiment_patterns.items():
            self.compiled_patterns[category] = []
            for pattern in data['patterns']:
                compiled = re.compile(pattern, re.IGNORECASE | re.UNICODE)
                self.compiled_patterns[category].append(compiled)
    
    def analyze_sentiment(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment and excitement in messages"""
        if not messages:
            return {
                'sentiment_score': 0.0,
                'excitement_level': 0.0,
                'sentiment_type': 'neutral',
                'confidence': 0.0,
                'details': {}
            }
        
        # Analyze individual messages
        message_sentiments = []
        for message in messages:
            sentiment = self._analyze_message(message)
            message_sentiments.append(sentiment)
            
            # Save to database
            if sentiment['confidence'] > 0.3:
                self._save_sentiment(message, sentiment)
        
        # Aggregate results
        aggregated = self._aggregate_sentiments(message_sentiments)
        
        return aggregated
    
    def _analyze_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of a single message"""
        content = message.get('content', '')
        if not content:
            return self._neutral_sentiment()
        
        sentiment_score = 0.0
        excitement_level = 0.0
        pattern_matches = Counter()
        emoji_counts = Counter()
        
        # Check sentiment patterns
        for category, patterns in self.compiled_patterns.items():
            category_data = self.sentiment_patterns[category]
            for pattern in patterns:
                matches = pattern.findall(content)
                if matches:
                    pattern_matches[category] += len(matches)
                    sentiment_score += category_data['weight'] * len(matches)
                    excitement_level += category_data['excitement_boost'] * len(matches)
        
        # Analyze emojis
        emoji_sentiment, emoji_excitement, emoji_counts = self._analyze_emojis(content)
        sentiment_score += emoji_sentiment
        excitement_level += emoji_excitement
        
        # Analyze punctuation
        exclamation_count = content.count('!')
        question_count = content.count('?')
        
        if exclamation_count > 0:
            excitement_level += min(exclamation_count * 0.1, 0.5)
        
        # Check for CAPS LOCK
        caps_words = re.findall(r'\b[A-Z]{3,}\b', content)
        if caps_words:
            excitement_level += min(len(caps_words) * 0.15, 0.5)
        
        # Normalize scores
        sentiment_score = max(min(sentiment_score / 10, 1.0), -1.0)  # Normalize to -1 to 1
        excitement_level = min(excitement_level, 1.0)  # Cap at 1.0
        
        # Determine sentiment type
        if sentiment_score > 0.3:
            sentiment_type = 'positive'
        elif sentiment_score < -0.3:
            sentiment_type = 'negative'
        else:
            sentiment_type = 'neutral'
        
        # Calculate confidence
        total_patterns = sum(pattern_matches.values())
        confidence = min((total_patterns + len(emoji_counts)) / 10, 1.0)
        
        return {
            'sentiment_score': sentiment_score,
            'excitement_level': excitement_level,
            'sentiment_type': sentiment_type,
            'confidence': confidence,
            'pattern_matches': dict(pattern_matches),
            'emoji_counts': dict(emoji_counts),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'caps_words': len(caps_words)
        }
    
    def _analyze_emojis(self, content: str) -> Tuple[float, float, Counter]:
        """Analyze emoji sentiment and excitement"""
        sentiment_score = 0.0
        excitement_level = 0.0
        emoji_counts = Counter()
        
        for emoji_type, emoji_list in self.emoji_sentiments.items():
            for emoji in emoji_list:
                count = content.count(emoji)
                if count > 0:
                    emoji_counts[emoji] += count
                    
                    if emoji_type == 'positive':
                        sentiment_score += 0.2 * count
                        excitement_level += 0.1 * count
                    elif emoji_type == 'negative':
                        sentiment_score -= 0.2 * count
                        excitement_level += 0.05 * count  # Negative still shows engagement
                    elif emoji_type == 'excitement':
                        excitement_level += 0.2 * count
                        sentiment_score += 0.1 * count
        
        return sentiment_score, excitement_level, emoji_counts
    
    def _aggregate_sentiments(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple sentiment analyses"""
        if not sentiments:
            return self._neutral_sentiment()
        
        # Calculate weighted averages
        total_weight = sum(s['confidence'] for s in sentiments)
        if total_weight == 0:
            return self._neutral_sentiment()
        
        avg_sentiment = sum(s['sentiment_score'] * s['confidence'] for s in sentiments) / total_weight
        avg_excitement = sum(s['excitement_level'] * s['confidence'] for s in sentiments) / total_weight
        avg_confidence = np.mean([s['confidence'] for s in sentiments])
        
        # Determine overall type
        if avg_sentiment > 0.3:
            sentiment_type = 'positive'
        elif avg_sentiment < -0.3:
            sentiment_type = 'negative'
        else:
            sentiment_type = 'neutral'
        
        # Calculate trending
        if len(sentiments) > 3:
            recent = sentiments[-3:]
            older = sentiments[:-3]
            recent_sentiment = np.mean([s['sentiment_score'] for s in recent])
            older_sentiment = np.mean([s['sentiment_score'] for s in older])
            
            if recent_sentiment > older_sentiment + 0.2:
                trend = 'improving'
            elif recent_sentiment < older_sentiment - 0.2:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        # Aggregate pattern matches
        all_patterns = Counter()
        for s in sentiments:
            if 'pattern_matches' in s:
                all_patterns.update(s['pattern_matches'])
        
        return {
            'sentiment_score': avg_sentiment,
            'excitement_level': avg_excitement,
            'sentiment_type': sentiment_type,
            'confidence': avg_confidence,
            'trend': trend,
            'message_count': len(sentiments),
            'details': {
                'pattern_summary': dict(all_patterns.most_common(5)),
                'excitement_peaks': sum(1 for s in sentiments if s['excitement_level'] > 0.7),
                'positive_ratio': sum(1 for s in sentiments if s['sentiment_type'] == 'positive') / len(sentiments),
                'engagement_score': avg_excitement * avg_confidence
            }
        }
    
    def _neutral_sentiment(self) -> Dict[str, Any]:
        """Return neutral sentiment result"""
        return {
            'sentiment_score': 0.0,
            'excitement_level': 0.0,
            'sentiment_type': 'neutral',
            'confidence': 0.0,
            'details': {}
        }
    
    def _save_sentiment(self, message: Dict[str, Any], sentiment: Dict[str, Any]):
        """Save sentiment analysis to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO sentiment_scores 
                (message_id, excitement_level, sentiment_type, confidence, 
                 emoji_count, exclamation_count, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.get('id'),
                sentiment['excitement_level'],
                sentiment['sentiment_type'],
                sentiment['confidence'],
                len(sentiment.get('emoji_counts', {})),
                sentiment.get('exclamation_count', 0),
                int(datetime.now().timestamp())
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving sentiment: {e}")
    
    def get_excitement_peaks(self, channel_id: int = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get messages with high excitement levels"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            time_threshold = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            query = '''
                SELECT s.*, m.content, m.sent_at, c.name as channel_name
                FROM sentiment_scores s
                JOIN messages m ON s.message_id = m.id
                LEFT JOIN channels c ON m.channel_id = c.id
                WHERE s.excitement_level >= 0.6 
                    AND m.sent_at >= ?
            '''
            params = [time_threshold]
            
            if channel_id:
                query += ' AND m.channel_id = ?'
                params.append(channel_id)
            
            query += ' ORDER BY s.excitement_level DESC LIMIT 20'
            
            cursor.execute(query, params)
            
            peaks = []
            for row in cursor.fetchall():
                peak = dict(row)
                # Add time info
                if peak.get('sent_at'):
                    peak['time_ago'] = self._format_time_ago(peak['sent_at'])
                peaks.append(peak)
            
            conn.close()
            return peaks
            
        except Exception as e:
            print(f"Error getting excitement peaks: {e}")
            return []
    
    def _format_time_ago(self, timestamp: int) -> str:
        """Format timestamp as time ago"""
        now = datetime.now()
        then = datetime.fromtimestamp(timestamp)
        diff = now - then
        
        if diff.days > 0:
            return f"{diff.days} dní zpět"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600} hodin zpět"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60} minut zpět"
        else:
            return "právě teď"
    
    def get_channel_mood(self, channel_id: int, hours: int = 6) -> Dict[str, Any]:
        """Get overall mood/sentiment of a channel"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            time_threshold = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            # Get recent messages
            cursor.execute('''
                SELECT m.*, s.excitement_level, s.sentiment_type
                FROM messages m
                LEFT JOIN sentiment_scores s ON m.id = s.message_id
                WHERE m.channel_id = ? AND m.sent_at >= ?
                ORDER BY m.sent_at DESC
            ''', (channel_id, time_threshold))
            
            messages = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            if not messages:
                return {
                    'channel_id': channel_id,
                    'mood': 'quiet',
                    'activity_level': 0,
                    'sentiment': self._neutral_sentiment()
                }
            
            # Analyze recent sentiment
            sentiment = self.analyze_sentiment(messages)
            
            # Determine mood
            if sentiment['excitement_level'] > 0.7:
                mood = 'hyped'
            elif sentiment['excitement_level'] > 0.4:
                mood = 'active'
            elif sentiment['sentiment_type'] == 'positive':
                mood = 'positive'
            elif sentiment['sentiment_type'] == 'negative':
                mood = 'tense'
            else:
                mood = 'neutral'
            
            return {
                'channel_id': channel_id,
                'mood': mood,
                'activity_level': len(messages),
                'sentiment': sentiment
            }
            
        except Exception as e:
            print(f"Error getting channel mood: {e}")
            return {
                'channel_id': channel_id,
                'error': str(e)
            }