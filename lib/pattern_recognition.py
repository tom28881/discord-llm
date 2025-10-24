"""
Group Activity Detection and Pattern Recognition System
Detects group purchases, events, consensus moments, and FOMO opportunities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import logging
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger('pattern_recognition')

@dataclass
class ActivityCluster:
    """Represents a detected group activity"""
    activity_type: str  # 'purchase', 'event', 'consensus', 'discussion'
    messages: List[Dict[str, Any]]
    participants: Set[str]
    confidence: float
    time_span: timedelta
    keywords: List[str]
    summary: str
    urgency_score: float

class GroupActivityDetector:
    """Detects various types of group activities and patterns"""
    
    def __init__(self):
        # Activity patterns
        self.purchase_patterns = {
            'direct_purchase': [
                r'\b(?:buy|bought|purchase|order|ordering)\b',
                r'\$\d+|\d+\s*(?:dollars?|bucks?)',
                r'\b(?:amazon|ebay|store|shop|deal|sale)\b'
            ],
            'group_buy': [
                r'\b(?:group buy|bulk|split|share|together)\b',
                r'\b(?:who wants|anyone interested|count me in)\b',
                r'\b(?:shipping|total cost|divide)\b'
            ],
            'recommendation': [
                r'\b(?:recommend|suggest|should buy|worth it)\b',
                r'\b(?:review|rating|quality|price)\b',
                r'\b(?:tried|using|works well)\b'
            ]
        }
        
        self.event_patterns = {
            'planning': [
                r'\b(?:plan|planning|organize|arrange)\b',
                r'\b(?:meeting|party|hangout|get together)\b',
                r'\b(?:when|where|time|date|schedule)\b'
            ],
            'invitation': [
                r'\b(?:invite|invited|join|come)\b',
                r'\b(?:tonight|tomorrow|this weekend|next week)\b',
                r'\b(?:everyone|all|who\'s in)\b'
            ],
            'confirmation': [
                r'\b(?:confirmed|booked|reserved|set)\b',
                r'\b(?:address|location|venue|place)\b',
                r'\b(?:time|hour|pm|am)\b'
            ]
        }
        
        self.consensus_patterns = [
            r'\b(?:agree|agreed|consensus|unanimous)\b',
            r'\b(?:everyone|all|most|majority)\b',
            r'\b(?:yes|yep|sure|definitely|absolutely)\b',
            r'\b(?:sounds good|looks good|perfect|great idea)\b'
        ]
        
        self.urgency_patterns = [
            r'\b(?:urgent|asap|hurry|deadline|soon)\b',
            r'\b(?:ends today|limited time|last chance)\b',
            r'\b(?:quick|fast|now|immediate)\b'
        ]
        
        # Temporal clustering parameters
        self.time_window = timedelta(hours=2)  # Group messages within 2 hours
        self.min_participants = 2
        self.min_messages = 3
    
    def detect_activities(self, messages_df: pd.DataFrame, 
                         time_window_hours: int = 24) -> List[ActivityCluster]:
        """Detect all types of group activities in recent messages"""
        
        activities = []
        
        # Filter recent messages
        cutoff_time = datetime.now().timestamp() - (time_window_hours * 3600)
        recent_messages = messages_df[messages_df['sent_at'] > cutoff_time].copy()
        
        if len(recent_messages) < self.min_messages:
            return activities
        
        # Sort by timestamp
        recent_messages = recent_messages.sort_values('sent_at')
        
        # Detect different activity types
        activities.extend(self._detect_purchase_activities(recent_messages))
        activities.extend(self._detect_event_activities(recent_messages))
        activities.extend(self._detect_consensus_moments(recent_messages))
        activities.extend(self._detect_discussion_threads(recent_messages))
        
        # Sort by confidence and urgency
        activities.sort(key=lambda x: (x.confidence * x.urgency_score), reverse=True)
        
        return activities
    
    def _detect_purchase_activities(self, messages_df: pd.DataFrame) -> List[ActivityCluster]:
        """Detect group purchase activities"""
        activities = []
        
        # Find messages with purchase indicators
        purchase_messages = []
        for _, row in messages_df.iterrows():
            content = str(row.get('content', '')).lower()
            
            # Check for purchase patterns
            purchase_score = 0
            matched_patterns = []
            
            for pattern_type, patterns in self.purchase_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        purchase_score += 1
                        matched_patterns.append(pattern_type)
            
            if purchase_score >= 2:  # At least 2 purchase indicators
                purchase_messages.append({
                    'message': row.to_dict(),
                    'score': purchase_score,
                    'patterns': matched_patterns
                })
        
        if len(purchase_messages) < self.min_messages:
            return activities
        
        # Cluster temporally related purchase messages
        clusters = self._temporal_clustering(purchase_messages)
        
        for cluster in clusters:
            if len(cluster) >= self.min_messages:
                participants = set(msg['message']['author_id'] for msg in cluster if 'author_id' in msg['message'])
                
                if len(participants) >= self.min_participants:
                    activity = self._create_activity_cluster(
                        activity_type='purchase',
                        messages=cluster,
                        participants=participants
                    )
                    activities.append(activity)
        
        return activities
    
    def _detect_event_activities(self, messages_df: pd.DataFrame) -> List[ActivityCluster]:
        """Detect event planning and coordination activities"""
        activities = []
        
        event_messages = []
        for _, row in messages_df.iterrows():
            content = str(row.get('content', '')).lower()
            
            event_score = 0
            matched_patterns = []
            
            for pattern_type, patterns in self.event_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        event_score += 1
                        matched_patterns.append(pattern_type)
            
            if event_score >= 1:  # More lenient for event detection
                event_messages.append({
                    'message': row.to_dict(),
                    'score': event_score,
                    'patterns': matched_patterns
                })
        
        if len(event_messages) < self.min_messages:
            return activities
        
        # Cluster event-related messages
        clusters = self._temporal_clustering(event_messages)
        
        for cluster in clusters:
            if len(cluster) >= self.min_messages:
                participants = set(msg['message']['author_id'] for msg in cluster if 'author_id' in msg['message'])
                
                if len(participants) >= self.min_participants:
                    activity = self._create_activity_cluster(
                        activity_type='event',
                        messages=cluster,
                        participants=participants
                    )
                    activities.append(activity)
        
        return activities
    
    def _detect_consensus_moments(self, messages_df: pd.DataFrame) -> List[ActivityCluster]:
        """Detect moments when group reaches consensus"""
        activities = []
        
        # Look for agreement patterns in message sequences
        messages_list = messages_df.to_dict('records')
        
        for i in range(len(messages_list) - 2):
            window_messages = messages_list[i:i+5]  # Check 5-message windows
            
            agreement_count = 0
            participants = set()
            
            for msg in window_messages:
                content = str(msg.get('content', '')).lower()
                author_id = msg.get('author_id', '')
                
                # Check for consensus patterns
                for pattern in self.consensus_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        agreement_count += 1
                        participants.add(author_id)
                        break
            
            # Detect consensus if multiple people agree in short time
            if agreement_count >= 3 and len(participants) >= 2:
                time_span = datetime.fromtimestamp(window_messages[-1]['sent_at']) - \
                           datetime.fromtimestamp(window_messages[0]['sent_at'])
                
                if time_span <= timedelta(minutes=30):  # Within 30 minutes
                    activity = ActivityCluster(
                        activity_type='consensus',
                        messages=window_messages,
                        participants=participants,
                        confidence=min(agreement_count / 3.0, 1.0),
                        time_span=time_span,
                        keywords=['agreement', 'consensus'],
                        summary=f"Group consensus reached among {len(participants)} participants",
                        urgency_score=0.8  # Consensus moments are often time-sensitive
                    )
                    activities.append(activity)
        
        return activities
    
    def _detect_discussion_threads(self, messages_df: pd.DataFrame) -> List[ActivityCluster]:
        """Detect important discussion threads using message clustering"""
        activities = []
        
        if len(messages_df) < 5:
            return activities
        
        # Use TF-IDF to find similar messages that form threads
        messages_text = messages_df['content'].fillna('').tolist()
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                min_df=2,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(messages_text)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Use DBSCAN to cluster similar messages
            clustering = DBSCAN(eps=0.3, min_samples=3, metric='precomputed')
            distance_matrix = 1 - similarity_matrix
            clusters = clustering.fit_predict(distance_matrix)
            
            # Process each cluster
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_indices = np.where(clusters == cluster_id)[0]
                cluster_messages = messages_df.iloc[cluster_indices].to_dict('records')
                
                participants = set(msg['author_id'] for msg in cluster_messages if 'author_id' in msg)
                
                if len(cluster_messages) >= 4 and len(participants) >= 2:
                    # Calculate thread importance based on participation and keywords
                    importance_score = self._calculate_thread_importance(cluster_messages)
                    
                    if importance_score > 0.5:  # Only high-importance threads
                        activity = self._create_activity_cluster(
                            activity_type='discussion',
                            messages=[{'message': msg} for msg in cluster_messages],
                            participants=participants
                        )
                        activity.confidence = importance_score
                        activities.append(activity)
        
        except Exception as e:
            logger.warning(f"Error in discussion thread detection: {e}")
        
        return activities
    
    def _temporal_clustering(self, messages_with_scores: List[Dict]) -> List[List[Dict]]:
        """Cluster messages that are temporally close"""
        if not messages_with_scores:
            return []
        
        # Sort by timestamp
        sorted_messages = sorted(messages_with_scores, 
                               key=lambda x: x['message']['sent_at'])
        
        clusters = []
        current_cluster = [sorted_messages[0]]
        
        for i in range(1, len(sorted_messages)):
            current_time = sorted_messages[i]['message']['sent_at']
            last_time = current_cluster[-1]['message']['sent_at']
            
            time_diff = datetime.fromtimestamp(current_time) - \
                       datetime.fromtimestamp(last_time)
            
            if time_diff <= self.time_window:
                current_cluster.append(sorted_messages[i])
            else:
                if len(current_cluster) >= self.min_messages:
                    clusters.append(current_cluster)
                current_cluster = [sorted_messages[i]]
        
        # Add the last cluster
        if len(current_cluster) >= self.min_messages:
            clusters.append(current_cluster)
        
        return clusters
    
    def _create_activity_cluster(self, activity_type: str, messages: List[Dict], 
                               participants: Set[str]) -> ActivityCluster:
        """Create an ActivityCluster from detected messages"""
        
        # Extract message data
        message_objects = [msg.get('message', msg) for msg in messages]
        
        # Calculate time span
        timestamps = [msg['sent_at'] for msg in message_objects]
        time_span = datetime.fromtimestamp(max(timestamps)) - \
                   datetime.fromtimestamp(min(timestamps))
        
        # Extract keywords
        all_content = ' '.join(str(msg.get('content', '')) for msg in message_objects)
        keywords = self._extract_keywords(all_content, activity_type)
        
        # Calculate confidence based on pattern matching
        confidence = self._calculate_confidence(messages, activity_type)
        
        # Calculate urgency score
        urgency_score = self._calculate_urgency(message_objects)
        
        # Generate summary
        summary = self._generate_summary(activity_type, participants, keywords, time_span)
        
        return ActivityCluster(
            activity_type=activity_type,
            messages=message_objects,
            participants=participants,
            confidence=confidence,
            time_span=time_span,
            keywords=keywords,
            summary=summary,
            urgency_score=urgency_score
        )
    
    def _extract_keywords(self, text: str, activity_type: str) -> List[str]:
        """Extract relevant keywords from text based on activity type"""
        text_lower = text.lower()
        keywords = []
        
        if activity_type == 'purchase':
            patterns = self.purchase_patterns
        elif activity_type == 'event':
            patterns = self.event_patterns
        else:
            return []
        
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                keywords.extend([match for match in matches if isinstance(match, str)])
        
        # Remove duplicates and return top keywords
        return list(set(keywords))[:5]
    
    def _calculate_confidence(self, messages: List[Dict], activity_type: str) -> float:
        """Calculate confidence score for detected activity"""
        total_score = sum(msg.get('score', 1) for msg in messages)
        max_possible_score = len(messages) * 3  # Assuming max 3 patterns per message
        
        return min(total_score / max_possible_score, 1.0)
    
    def _calculate_urgency(self, messages: List[Dict]) -> float:
        """Calculate urgency score based on urgency patterns and recency"""
        urgency_score = 0.0
        
        for msg in messages:
            content = str(msg.get('content', '')).lower()
            
            # Check for urgency patterns
            for pattern in self.urgency_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    urgency_score += 0.3
            
            # Time-based urgency (more recent = more urgent)
            hours_ago = (datetime.now().timestamp() - msg['sent_at']) / 3600
            time_urgency = max(0, 1.0 - (hours_ago / 24))  # Decay over 24 hours
            urgency_score += time_urgency * 0.2
        
        return min(urgency_score / len(messages), 1.0)
    
    def _calculate_thread_importance(self, messages: List[Dict]) -> float:
        """Calculate importance of a discussion thread"""
        
        # Factors: number of participants, message frequency, content complexity
        participants = set(msg['author_id'] for msg in messages if 'author_id' in msg)
        
        # Participation score
        participation_score = min(len(participants) / 5.0, 1.0)  # Normalize to max 5 participants
        
        # Frequency score (messages per hour)
        if len(messages) > 1:
            time_span_hours = (messages[-1]['sent_at'] - messages[0]['sent_at']) / 3600
            frequency_score = min(len(messages) / max(time_span_hours, 0.1), 1.0)
        else:
            frequency_score = 0.1
        
        # Content complexity (average message length)
        avg_length = np.mean([len(str(msg.get('content', ''))) for msg in messages])
        complexity_score = min(avg_length / 100.0, 1.0)  # Normalize to 100 chars
        
        return (participation_score + frequency_score + complexity_score) / 3.0
    
    def _generate_summary(self, activity_type: str, participants: Set[str], 
                         keywords: List[str], time_span: timedelta) -> str:
        """Generate a human-readable summary of the activity"""
        
        participant_count = len(participants)
        duration = f"{time_span.total_seconds() / 60:.0f} minutes"
        
        if activity_type == 'purchase':
            return f"Group purchase discussion involving {participant_count} people over {duration}. Keywords: {', '.join(keywords[:3])}"
        elif activity_type == 'event':
            return f"Event planning with {participant_count} participants over {duration}. Keywords: {', '.join(keywords[:3])}"
        elif activity_type == 'consensus':
            return f"Group consensus reached among {participant_count} people in {duration}"
        elif activity_type == 'discussion':
            return f"Active discussion thread with {participant_count} participants over {duration}"
        else:
            return f"{activity_type.title()} activity with {participant_count} people over {duration}"

class FOMODetector:
    """Detect Fear of Missing Out moments"""
    
    def __init__(self):
        self.fomo_indicators = [
            r'\b(?:limited|exclusive|rare|special)\b',
            r'\b(?:ending|expires|deadline|last chance)\b',
            r'\b(?:everyone|all|most people)\b',
            r'\b(?:hurry|quick|fast|now|asap)\b',
            r'\b(?:deal|sale|discount|offer)\b'
        ]
    
    def detect_fomo_moments(self, activities: List[ActivityCluster]) -> List[ActivityCluster]:
        """Identify activities that might trigger FOMO"""
        
        fomo_activities = []
        
        for activity in activities:
            fomo_score = 0.0
            
            # Check all messages for FOMO indicators
            for message in activity.messages:
                content = str(message.get('content', '')).lower()
                
                for indicator in self.fomo_indicators:
                    if re.search(indicator, content, re.IGNORECASE):
                        fomo_score += 1.0
            
            # Normalize by number of messages
            fomo_score = fomo_score / len(activity.messages)
            
            # High participation also indicates potential FOMO
            if len(activity.participants) >= 4:
                fomo_score += 0.5
            
            # Recent activities are more FOMO-inducing
            if activity.time_span < timedelta(hours=1):
                fomo_score += 0.3
            
            # If FOMO score is high enough, mark this activity
            if fomo_score >= 0.8:
                activity.urgency_score = min(activity.urgency_score + 0.2, 1.0)
                fomo_activities.append(activity)
        
        return fomo_activities