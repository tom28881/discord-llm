"""
ML-based importance scoring system for Discord messages.
Detects important messages, group activities, and FOMO moments.
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger('discord_bot')


@dataclass
class ImportanceFactors:
    """Factors that contribute to message importance."""
    mention_score: float = 0.0
    keyword_score: float = 0.0
    urgency_score: float = 0.0
    social_score: float = 0.0
    recency_score: float = 0.0
    author_score: float = 0.0
    pattern_score: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Calculate weighted total score."""
        weights = {
            'mention': 0.25,
            'keyword': 0.20,
            'urgency': 0.20,
            'social': 0.15,
            'recency': 0.05,
            'author': 0.05,
            'pattern': 0.10
        }
        
        total = (
            self.mention_score * weights['mention'] +
            self.keyword_score * weights['keyword'] +
            self.urgency_score * weights['urgency'] +
            self.social_score * weights['social'] +
            self.recency_score * weights['recency'] +
            self.author_score * weights['author'] +
            self.pattern_score * weights['pattern']
        )
        
        return min(1.0, max(0.0, total))


class ImportanceScorer:
    """Score message importance using multiple factors."""
    
    def __init__(self, user_preferences: Optional[Dict[str, float]] = None):
        """Initialize scorer with user preferences."""
        self.user_preferences = user_preferences or {}
        self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load default importance patterns."""
        self.patterns = {
            'mentions': {
                'everyone': r'@everyone',
                'here': r'@here',
                'user': r'@\w+',
                'role': r'@&\d+'
            },
            'urgency': {
                'critical': r'\b(critical|emergency|urgent|asap|immediately)\b',
                'deadline': r'\b(deadline|due|expires?|ends?|closing)\b',
                'time_sensitive': r'\b(today|tonight|tomorrow|now|soon|quickly)\b',
                'action_required': r'\b(action required|needs?|must|required|mandatory)\b'
            },
            'group_activity': {
                'purchase': r'\b(group\s*buy|split\s*cost|bulk\s*order|who\'s\s*in|count\s*me\s*in)\b',
                'event': r'\b(meeting|event|party|gathering|hangout|get[\s-]*together)\b',
                'decision': r'\b(vote|poll|decide|choose|opinion|thoughts?|what\s*do\s*you\s*think)\b',
                'coordination': r'\b(everyone|all|together|join|participate)\b'
            },
            'keywords': {
                'important': r'\b(important|significance|critical|vital|essential)\b',
                'announcement': r'\b(announcement|update|news|notice|fyi|psa)\b',
                'question': r'\?|^\s*(can|could|would|should|is|are|do|does|what|when|where|why|how)',
                'money': r'\$\d+|\d+\s*(usd|eur|gbp)|cost|price|payment|pay'
            },
            'fomo': {
                'limited': r'\b(limited|exclusive|only|last\s*chance|ending\s*soon)\b',
                'participation': r'\b(sign\s*up|register|rsvp|confirm|interested)\b',
                'consensus': r'\b(everyone\'s|we\'re\s*all|most\s*people|majority)\b'
            }
        }
    
    def score_message(self, message: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Score a message's importance.
        
        Args:
            message: Message data with content, author, timestamp, etc.
            context: Optional context like recent messages, channel activity
            
        Returns:
            Tuple of (importance_score, analysis_details)
        """
        content = message.get('content', '').lower()
        factors = ImportanceFactors()
        matched_keywords = []
        category = 'normal'
        
        # 1. Mention scoring
        factors.mention_score = self._score_mentions(content)
        
        # 2. Keyword scoring
        factors.keyword_score, keywords = self._score_keywords(content)
        matched_keywords.extend(keywords)
        
        # 3. Urgency scoring
        factors.urgency_score, urgency_keywords = self._score_urgency(content)
        matched_keywords.extend(urgency_keywords)
        
        # 4. Social scoring (requires context)
        if context:
            factors.social_score = self._score_social_signals(message, context)
        
        # 5. Recency scoring
        factors.recency_score = self._score_recency(message.get('sent_at'))
        
        # 6. Author scoring (if we have author importance data)
        factors.author_score = self._score_author(message.get('author_id'), message.get('author_name'))
        
        # 7. Pattern scoring (group activities, FOMO)
        factors.pattern_score, pattern_type = self._score_patterns(content)
        if pattern_type:
            category = pattern_type
        
        # Calculate final score
        importance_score = factors.total_score
        
        # Determine urgency level
        urgency_level = self._calculate_urgency_level(factors)
        
        # Override category based on score
        if importance_score > 0.8:
            if factors.urgency_score > 0.8:
                category = 'urgent'
            elif factors.pattern_score > 0.7:
                category = 'group_activity'
            else:
                category = 'important'
        
        analysis = {
            'importance_score': importance_score,
            'urgency_level': urgency_level,
            'category': category,
            'keywords_matched': list(set(matched_keywords)),
            'factors': {
                'mention': factors.mention_score,
                'keyword': factors.keyword_score,
                'urgency': factors.urgency_score,
                'social': factors.social_score,
                'recency': factors.recency_score,
                'author': factors.author_score,
                'pattern': factors.pattern_score
            }
        }
        
        return importance_score, analysis
    
    def _score_mentions(self, content: str) -> float:
        """Score based on mentions."""
        if re.search(self.patterns['mentions']['everyone'], content, re.IGNORECASE):
            return 0.9
        elif re.search(self.patterns['mentions']['here'], content, re.IGNORECASE):
            return 0.7
        elif re.search(self.patterns['mentions']['role'], content):
            return 0.6
        elif re.search(self.patterns['mentions']['user'], content):
            return 0.4
        return 0.0
    
    def _score_keywords(self, content: str) -> Tuple[float, List[str]]:
        """Score based on important keywords."""
        score = 0.0
        matched = []
        
        # Check user preferences first
        for keyword, weight in self.user_preferences.items():
            if keyword.lower() in content:
                score = max(score, weight)
                matched.append(keyword)
        
        # Check default keywords
        for category, pattern in self.patterns['keywords'].items():
            if re.search(pattern, content, re.IGNORECASE):
                category_scores = {
                    'important': 0.7,
                    'announcement': 0.6,
                    'question': 0.3,
                    'money': 0.5
                }
                score = max(score, category_scores.get(category, 0.3))
                matched.append(category)
        
        return min(1.0, score), matched
    
    def _score_urgency(self, content: str) -> Tuple[float, List[str]]:
        """Score based on urgency indicators."""
        score = 0.0
        matched = []
        
        urgency_scores = {
            'critical': 1.0,
            'deadline': 0.8,
            'time_sensitive': 0.6,
            'action_required': 0.7
        }
        
        for urgency_type, pattern in self.patterns['urgency'].items():
            if re.search(pattern, content, re.IGNORECASE):
                score = max(score, urgency_scores[urgency_type])
                matched.append(urgency_type)
        
        return score, matched
    
    def _score_social_signals(self, message: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Score based on social signals from context."""
        score = 0.0
        
        # Check if multiple people are discussing the same topic
        recent_messages = context.get('recent_messages', [])
        if recent_messages:
            # Count unique participants in recent conversation
            participants = set()
            topic_keywords = set()
            
            for msg in recent_messages[-10:]:  # Last 10 messages
                participants.add(msg.get('author_id'))
                # Extract potential topic keywords
                words = msg.get('content', '').lower().split()
                topic_keywords.update(word for word in words if len(word) > 4)
            
            # High participation = higher score
            if len(participants) >= 5:
                score = 0.8
            elif len(participants) >= 3:
                score = 0.5
            elif len(participants) >= 2:
                score = 0.2
        
        # Check message velocity (many messages in short time)
        message_velocity = context.get('message_velocity', 0)
        if message_velocity > 10:  # More than 10 messages per minute
            score = max(score, 0.6)
        
        return score
    
    def _score_recency(self, sent_at: Optional[int]) -> float:
        """Score based on message recency."""
        if not sent_at:
            return 0.5
        
        now = time.time()
        age_hours = (now - sent_at) / 3600
        
        if age_hours < 1:
            return 1.0
        elif age_hours < 6:
            return 0.7
        elif age_hours < 24:
            return 0.4
        elif age_hours < 72:
            return 0.2
        else:
            return 0.0
    
    def _score_author(self, author_id: Optional[int], author_name: Optional[str]) -> float:
        """Score based on author importance."""
        # This could be enhanced with actual author importance data
        # For now, use simple heuristics
        
        if not author_name:
            return 0.0
        
        # Check for bot/system accounts (often important)
        if 'bot' in author_name.lower() or 'system' in author_name.lower():
            return 0.6
        
        # Check for admin/mod indicators
        if any(role in author_name.lower() for role in ['admin', 'mod', 'owner']):
            return 0.7
        
        return 0.0
    
    def _score_patterns(self, content: str) -> Tuple[float, Optional[str]]:
        """Score based on pattern detection (group activities, FOMO)."""
        max_score = 0.0
        detected_pattern = None
        
        # Check group activity patterns
        for activity_type, pattern in self.patterns['group_activity'].items():
            if re.search(pattern, content, re.IGNORECASE):
                scores = {
                    'purchase': 0.9,
                    'event': 0.7,
                    'decision': 0.6,
                    'coordination': 0.5
                }
                score = scores.get(activity_type, 0.5)
                if score > max_score:
                    max_score = score
                    detected_pattern = f'group_{activity_type}'
        
        # Check FOMO patterns
        for fomo_type, pattern in self.patterns['fomo'].items():
            if re.search(pattern, content, re.IGNORECASE):
                scores = {
                    'limited': 0.8,
                    'participation': 0.7,
                    'consensus': 0.6
                }
                score = scores.get(fomo_type, 0.5)
                if score > max_score:
                    max_score = score
                    detected_pattern = f'fomo_{fomo_type}'
        
        return max_score, detected_pattern
    
    def _calculate_urgency_level(self, factors: ImportanceFactors) -> int:
        """Calculate urgency level (0-5)."""
        if factors.urgency_score > 0.9:
            return 5
        elif factors.urgency_score > 0.7:
            return 4
        elif factors.urgency_score > 0.5:
            return 3
        elif factors.urgency_score > 0.3:
            return 2
        elif factors.urgency_score > 0.1:
            return 1
        else:
            return 0
    
    def update_preferences(self, keyword: str, weight: float):
        """Update user preference for a keyword."""
        self.user_preferences[keyword.lower()] = min(1.0, max(0.0, weight))
    
    def detect_group_activity(self, messages: List[Dict[str, Any]], 
                            time_window_minutes: int = 30) -> Optional[Dict[str, Any]]:
        """
        Detect group activities from a cluster of messages.
        
        Args:
            messages: List of messages to analyze
            time_window_minutes: Time window to consider messages as related
            
        Returns:
            Detected group activity or None
        """
        if len(messages) < 3:
            return None
        
        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.get('sent_at', 0))
        
        # Group messages by time windows
        activities = []
        current_group = []
        last_timestamp = 0
        
        for msg in sorted_messages:
            timestamp = msg.get('sent_at', 0)
            
            if not current_group or (timestamp - last_timestamp) < (time_window_minutes * 60):
                current_group.append(msg)
                last_timestamp = timestamp
            else:
                # Analyze the current group
                if len(current_group) >= 3:
                    activity = self._analyze_message_group(current_group)
                    if activity:
                        activities.append(activity)
                
                # Start new group
                current_group = [msg]
                last_timestamp = timestamp
        
        # Don't forget the last group
        if len(current_group) >= 3:
            activity = self._analyze_message_group(current_group)
            if activity:
                activities.append(activity)
        
        # Return the most significant activity
        if activities:
            return max(activities, key=lambda a: a['confidence'])
        
        return None
    
    def _analyze_message_group(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze a group of messages for activity patterns."""
        # Combine all message content
        combined_content = ' '.join(msg.get('content', '') for msg in messages)
        
        # Count unique participants
        participants = list(set(msg.get('author_id') for msg in messages if msg.get('author_id')))
        
        # Detect activity type
        activity_type = None
        confidence = 0.0
        
        # Check for purchase activity
        purchase_keywords = ['group buy', 'split', 'cost', 'price', 'order', 'payment']
        purchase_count = sum(1 for kw in purchase_keywords if kw in combined_content.lower())
        if purchase_count >= 2:
            activity_type = 'group_purchase'
            confidence = min(1.0, purchase_count * 0.3)
        
        # Check for event planning
        event_keywords = ['meeting', 'event', 'when', 'time', 'date', 'where', 'location']
        event_count = sum(1 for kw in event_keywords if kw in combined_content.lower())
        if event_count >= 3 and not activity_type:
            activity_type = 'event_planning'
            confidence = min(1.0, event_count * 0.25)
        
        # Check for decision making
        decision_keywords = ['vote', 'choose', 'decide', 'option', 'preference', 'agree']
        decision_count = sum(1 for kw in decision_keywords if kw in combined_content.lower())
        if decision_count >= 2 and not activity_type:
            activity_type = 'group_decision'
            confidence = min(1.0, decision_count * 0.35)
        
        if activity_type and len(participants) >= 3:
            return {
                'type': activity_type,
                'participants': participants,
                'participant_count': len(participants),
                'message_count': len(messages),
                'start_time': messages[0].get('sent_at'),
                'end_time': messages[-1].get('sent_at'),
                'confidence': confidence * (1 + len(participants) * 0.1),  # Boost by participation
                'sample_content': combined_content[:500]
            }
        
        return None


class BatchImportanceScorer:
    """Score importance for batches of messages efficiently."""
    
    def __init__(self, user_preferences: Optional[Dict[str, float]] = None):
        self.scorer = ImportanceScorer(user_preferences)
    
    def score_messages_batch(self, messages: List[Dict[str, Any]], 
                            include_context: bool = True) -> List[Dict[str, Any]]:
        """
        Score a batch of messages.
        
        Args:
            messages: List of messages to score
            include_context: Whether to include context for social scoring
            
        Returns:
            List of messages with importance scores added
        """
        scored_messages = []
        
        for i, message in enumerate(messages):
            context = None
            
            if include_context and i > 0:
                # Provide recent messages as context
                context = {
                    'recent_messages': messages[max(0, i-10):i],
                    'message_velocity': self._calculate_velocity(messages[max(0, i-10):i+1])
                }
            
            score, analysis = self.scorer.score_message(message, context)
            
            # Add scoring to message
            message['importance_score'] = score
            message['importance_analysis'] = analysis
            
            scored_messages.append(message)
        
        # Detect group activities across the batch
        group_activity = self.scorer.detect_group_activity(messages)
        if group_activity:
            # Mark messages in the activity window
            for msg in scored_messages:
                if (group_activity['start_time'] <= msg.get('sent_at', 0) <= group_activity['end_time']):
                    msg['group_activity'] = group_activity
                    # Boost importance for group activity messages
                    msg['importance_score'] = min(1.0, msg['importance_score'] * 1.2)
        
        return scored_messages
    
    def _calculate_velocity(self, messages: List[Dict[str, Any]]) -> float:
        """Calculate message velocity (messages per minute)."""
        if len(messages) < 2:
            return 0.0
        
        timestamps = [msg.get('sent_at', 0) for msg in messages]
        time_span = max(timestamps) - min(timestamps)
        
        if time_span > 0:
            return len(messages) / (time_span / 60)  # Messages per minute
        
        return 0.0