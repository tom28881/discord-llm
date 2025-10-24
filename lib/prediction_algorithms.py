"""
Advanced Prediction Algorithms for Discord Monitoring

Specialized algorithms for each predictive capability:
1. Time Series Analysis for Event Prediction
2. Natural Language Processing for Importance Detection
3. Behavioral Pattern Recognition
4. Anomaly Detection for Emerging Trends
5. Decision Tree Models for FOMO Prevention

Author: Predictive Analytics Architect
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
from collections import Counter, defaultdict
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesPattern:
    """Container for time series patterns"""
    pattern_type: str  # 'seasonal', 'trend', 'cyclic', 'irregular'
    strength: float
    period: int  # in hours
    next_occurrence: datetime
    confidence: float

@dataclass
class ImportanceFeatures:
    """Features used for importance prediction"""
    content_features: Dict[str, float]
    temporal_features: Dict[str, float]
    context_features: Dict[str, float]
    social_features: Dict[str, float]

class AdvancedEventPredictor:
    """Advanced time series analysis for event prediction"""
    
    def __init__(self):
        self.seasonal_patterns = {}
        self.trend_models = {}
        
    def analyze_seasonal_patterns(self, df: pd.DataFrame, event_type: str) -> List[TimeSeriesPattern]:
        """Detect seasonal patterns in event occurrences"""
        patterns = []
        
        if df.empty:
            return patterns
        
        # Convert to time series
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Hourly patterns
        hourly_counts = df.groupby(df.index.hour).size()
        if len(hourly_counts) > 3:
            # Find peaks using simple peak detection
            peaks = self._find_peaks(hourly_counts.values)
            for peak_hour in peaks:
                strength = hourly_counts.iloc[peak_hour] / hourly_counts.mean()
                if strength > 1.5:  # Significant peak
                    next_occurrence = self._next_occurrence_at_hour(peak_hour)
                    patterns.append(TimeSeriesPattern(
                        pattern_type='hourly_seasonal',
                        strength=min(strength / 2, 1.0),
                        period=24,
                        next_occurrence=next_occurrence,
                        confidence=min(0.9, strength / 3)
                    ))
        
        # Weekly patterns
        weekly_counts = df.groupby(df.index.dayofweek).size()
        if len(weekly_counts) > 2:
            peaks = self._find_peaks(weekly_counts.values)
            for peak_day in peaks:
                strength = weekly_counts.iloc[peak_day] / weekly_counts.mean()
                if strength > 1.3:
                    next_occurrence = self._next_occurrence_on_day(peak_day)
                    patterns.append(TimeSeriesPattern(
                        pattern_type='weekly_seasonal',
                        strength=min(strength / 2, 1.0),
                        period=168,  # 24*7 hours
                        next_occurrence=next_occurrence,
                        confidence=min(0.8, strength / 2.5)
                    ))
        
        # Monthly patterns (for group buys that follow monthly cycles)
        if len(df) > 30:
            monthly_counts = df.groupby(df.index.day).size()
            peaks = self._find_peaks(monthly_counts.values)
            for peak_day in peaks:
                strength = monthly_counts.iloc[peak_day] / monthly_counts.mean()
                if strength > 1.4:
                    next_occurrence = self._next_occurrence_on_month_day(peak_day + 1)  # 1-indexed
                    patterns.append(TimeSeriesPattern(
                        pattern_type='monthly_seasonal',
                        strength=min(strength / 2, 1.0),
                        period=720,  # 24*30 hours (approximate)
                        next_occurrence=next_occurrence,
                        confidence=min(0.7, strength / 2)
                    ))
        
        return patterns
    
    def _find_peaks(self, data: np.ndarray, min_height: float = None) -> List[int]:
        """Simple peak detection algorithm"""
        if len(data) < 3:
            return []
            
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                if min_height is None or data[i] >= min_height:
                    peaks.append(i)
        
        return peaks
    
    def _next_occurrence_at_hour(self, hour: int) -> datetime:
        """Calculate next occurrence at specific hour"""
        now = datetime.now()
        next_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if next_time <= now:
            next_time += timedelta(days=1)
        return next_time
    
    def _next_occurrence_on_day(self, weekday: int) -> datetime:
        """Calculate next occurrence on specific weekday (0=Monday)"""
        now = datetime.now()
        days_ahead = weekday - now.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return now + timedelta(days=days_ahead)
    
    def _next_occurrence_on_month_day(self, day: int) -> datetime:
        """Calculate next occurrence on specific day of month"""
        now = datetime.now()
        if day <= now.day:
            # Next month
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=day, hour=12, minute=0, second=0)
            else:
                next_month = now.replace(month=now.month + 1, day=day, hour=12, minute=0, second=0)
            return next_month
        else:
            # This month
            return now.replace(day=day, hour=12, minute=0, second=0)
    
    def predict_event_probability(self, patterns: List[TimeSeriesPattern], 
                                  target_time: datetime) -> float:
        """Predict probability of event at target time"""
        if not patterns:
            return 0.1  # Base probability
        
        total_probability = 0.0
        total_weight = 0.0
        
        for pattern in patterns:
            # Calculate time difference from next occurrence
            time_diff = abs((target_time - pattern.next_occurrence).total_seconds() / 3600)  # hours
            
            # Gaussian-like probability distribution around predicted time
            if time_diff <= pattern.period / 4:  # Within quarter period
                prob = pattern.confidence * pattern.strength * np.exp(-time_diff / (pattern.period / 8))
                weight = pattern.confidence
                
                total_probability += prob * weight
                total_weight += weight
        
        return min(total_probability / max(total_weight, 1.0), 1.0) if total_weight > 0 else 0.1

class AdvancedImportanceAnalyzer:
    """Advanced NLP and ML for importance prediction"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.importance_model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Importance keywords with weights
        self.importance_keywords = {
            'critical': 3.0, 'urgent': 2.8, 'important': 2.5, 'emergency': 3.0,
            'breaking': 2.7, 'alert': 2.3, 'attention': 2.0, 'announcement': 2.2,
            'group buy': 2.8, 'drop': 2.5, 'limited': 2.3, 'sold out': 2.6,
            'restock': 2.4, 'price': 1.8, 'discount': 2.0, 'sale': 1.9,
            'meet': 2.1, 'event': 2.0, 'deadline': 2.5, 'decision': 2.2,
            'vote': 2.1, 'poll': 1.8, 'help': 1.9, 'problem': 2.1
        }
    
    def extract_importance_features(self, message_data: Dict[str, Any]) -> ImportanceFeatures:
        """Extract comprehensive features for importance prediction"""
        content = message_data.get('content', '').lower()
        
        # Content features
        content_features = {
            'length': len(content),
            'word_count': len(content.split()),
            'exclamation_ratio': content.count('!') / max(len(content), 1),
            'question_ratio': content.count('?') / max(len(content), 1),
            'caps_ratio': sum(1 for c in content if c.isupper()) / max(len(content), 1),
            'url_count': len(re.findall(r'http[s]?://[^\s]+', content)),
            'mention_count': len(re.findall(r'@\w+', content)),
            'emoji_count': len(re.findall(r':[a-zA-Z0-9_]+:', content)),
            'number_count': len(re.findall(r'\d+', content)),
        }
        
        # Keyword importance score
        keyword_score = 0.0
        for keyword, weight in self.importance_keywords.items():
            if keyword in content:
                keyword_score += weight
        content_features['keyword_score'] = min(keyword_score, 10.0)  # Cap at 10
        
        # Temporal features
        timestamp = message_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        temporal_features = {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'is_business_hours': 9 <= timestamp.hour <= 17,
            'is_evening': 17 <= timestamp.hour <= 22,
            'is_late_night': timestamp.hour >= 23 or timestamp.hour <= 5,
        }
        
        # Context features (channel-based)
        context_features = {
            'channel_importance': self._get_channel_importance(message_data.get('channel_id')),
            'message_position': message_data.get('message_position', 0),  # Position in conversation
            'conversation_length': message_data.get('conversation_length', 1),
        }
        
        # Social features
        social_features = {
            'has_reactions': message_data.get('reaction_count', 0) > 0,
            'reaction_count': message_data.get('reaction_count', 0),
            'reply_count': message_data.get('reply_count', 0),
            'thread_starter': message_data.get('is_thread_starter', False),
        }
        
        return ImportanceFeatures(
            content_features=content_features,
            temporal_features=temporal_features,
            context_features=context_features,
            social_features=social_features
        )
    
    def _get_channel_importance(self, channel_id: int) -> float:
        """Get importance multiplier based on channel type"""
        # This would typically be learned from data
        # For now, use heuristics based on channel names
        channel_importance_map = {
            'announcement': 3.0,
            'general': 1.0,
            'buy': 2.5,
            'sale': 2.3,
            'group-buy': 2.8,
            'urgent': 2.7,
            'important': 2.5,
            'off-topic': 0.5,
            'random': 0.7,
            'chat': 0.8
        }
        
        # Default importance
        return 1.0
    
    def predict_conversation_escalation(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict if conversation will escalate to importance"""
        if not messages:
            return {'escalation_probability': 0.0, 'confidence': 0.0}
        
        # Analyze conversation dynamics
        escalation_indicators = {
            'rapid_succession': 0.0,
            'increasing_length': 0.0,
            'emotional_intensity': 0.0,
            'multiple_participants': 0.0,
            'topic_clustering': 0.0
        }
        
        # Rapid succession (messages within short time)
        if len(messages) >= 3:
            time_diffs = []
            for i in range(1, len(messages)):
                prev_time = datetime.fromisoformat(messages[i-1]['timestamp'])
                curr_time = datetime.fromisoformat(messages[i]['timestamp'])
                time_diffs.append((curr_time - prev_time).total_seconds())
            
            avg_gap = np.mean(time_diffs)
            escalation_indicators['rapid_succession'] = min(300 / max(avg_gap, 30), 1.0)  # 5 min baseline
        
        # Increasing message length
        lengths = [len(msg.get('content', '')) for msg in messages[-5:]]
        if len(lengths) > 1:
            length_trend = np.polyfit(range(len(lengths)), lengths, 1)[0]  # Slope
            escalation_indicators['increasing_length'] = min(max(length_trend / 50, 0), 1.0)
        
        # Emotional intensity (caps, exclamations)
        total_intensity = 0.0
        for msg in messages[-3:]:  # Last 3 messages
            content = msg.get('content', '')
            caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
            exclamation_ratio = content.count('!') / max(len(content), 1)
            total_intensity += caps_ratio + exclamation_ratio
        
        escalation_indicators['emotional_intensity'] = min(total_intensity, 1.0)
        
        # Multiple participants
        unique_users = len(set(msg.get('user_id') for msg in messages if msg.get('user_id')))
        escalation_indicators['multiple_participants'] = min(unique_users / 5, 1.0)
        
        # Calculate overall escalation probability
        weights = [0.3, 0.2, 0.3, 0.2]  # Weights for each indicator
        escalation_prob = sum(indicator * weight for indicator, weight in 
                            zip(escalation_indicators.values(), weights))
        
        confidence = min(len(messages) / 10, 1.0)  # More messages = higher confidence
        
        return {
            'escalation_probability': escalation_prob,
            'confidence': confidence,
            'indicators': escalation_indicators
        }

class BehaviorPatternAnalyzer:
    """Advanced user behavior pattern recognition"""
    
    def __init__(self):
        self.user_profiles = {}
        self.activity_clusters = {}
        
    def build_user_activity_profile(self, user_messages: pd.DataFrame) -> Dict[str, Any]:
        """Build comprehensive user activity profile"""
        if user_messages.empty:
            return {}
        
        profile = {}
        
        # Temporal patterns
        hourly_activity = user_messages.groupby('hour').size()
        profile['peak_hours'] = hourly_activity.nlargest(3).index.tolist()
        profile['low_hours'] = hourly_activity.nsmallest(3).index.tolist()
        
        daily_activity = user_messages.groupby('day_of_week').size()
        profile['peak_days'] = daily_activity.nlargest(2).index.tolist()
        
        # Communication style
        profile['avg_message_length'] = user_messages['message_length'].mean()
        profile['avg_words'] = user_messages['word_count'].mean()
        profile['exclamation_frequency'] = user_messages['exclamation_count'].mean()
        profile['question_frequency'] = user_messages['question_count'].mean()
        profile['url_sharing_rate'] = user_messages['has_url'].mean()
        profile['mention_rate'] = user_messages['has_mention'].mean()
        
        # Activity consistency
        daily_counts = user_messages.groupby(user_messages['timestamp'].dt.date).size()
        profile['consistency_score'] = 1 - (daily_counts.std() / daily_counts.mean()) if daily_counts.mean() > 0 else 0
        
        # Response patterns
        profile['messages_per_day'] = len(user_messages) / max(user_messages['timestamp'].dt.date.nunique(), 1)
        
        return profile
    
    def predict_user_attention(self, user_profile: Dict[str, Any], 
                             current_time: datetime = None) -> Dict[str, float]:
        """Predict user attention/availability based on patterns"""
        if current_time is None:
            current_time = datetime.now()
        
        attention_score = 0.5  # Base attention
        
        hour = current_time.hour
        weekday = current_time.weekday()
        
        # Time-based attention
        peak_hours = user_profile.get('peak_hours', [])
        if hour in peak_hours:
            attention_score += 0.3
        
        low_hours = user_profile.get('low_hours', [])
        if hour in low_hours:
            attention_score -= 0.2
        
        # Day-based attention
        peak_days = user_profile.get('peak_days', [])
        if weekday in peak_days:
            attention_score += 0.2
        
        # Consistency factor
        consistency = user_profile.get('consistency_score', 0.5)
        attention_score = attention_score * (0.5 + consistency * 0.5)
        
        attention_score = max(0.0, min(1.0, attention_score))
        
        return {
            'attention_score': attention_score,
            'confidence': min(consistency * 2, 1.0),
            'recommendation': self._get_attention_recommendation(attention_score)
        }
    
    def _get_attention_recommendation(self, score: float) -> str:
        """Get recommendation based on attention score"""
        if score > 0.8:
            return "High attention expected - Good time for important notifications"
        elif score > 0.6:
            return "Moderate attention - Normal notification timing"
        elif score > 0.4:
            return "Lower attention - Consider delaying non-urgent notifications"
        else:
            return "Low attention period - Schedule important items for later"

class FOMAPreventionSystem:
    """Fear of Missing Out prevention through intelligent filtering"""
    
    def __init__(self):
        self.fomo_patterns = {}
        self.user_interests = {}
        
    def analyze_fomo_risk(self, message_data: Dict[str, Any], 
                         user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Analyze FOMO risk for a specific message"""
        content = message_data.get('content', '').lower()
        
        fomo_indicators = {
            'time_sensitivity': 0.0,
            'exclusivity': 0.0,
            'social_proof': 0.0,
            'personal_relevance': 0.0,
            'scarcity': 0.0
        }
        
        # Time sensitivity keywords
        time_sensitive = ['now', 'today', 'deadline', 'expires', 'limited time', 'hurry', 'quick', 'asap']
        fomo_indicators['time_sensitivity'] = sum(1 for keyword in time_sensitive if keyword in content) / len(time_sensitive)
        
        # Exclusivity indicators
        exclusive = ['exclusive', 'member', 'special', 'invite', 'select', 'vip', 'private']
        fomo_indicators['exclusivity'] = sum(1 for keyword in exclusive if keyword in content) / len(exclusive)
        
        # Social proof
        social_proof = ['everyone', 'join', 'popular', 'trending', 'most', 'best', 'top']
        fomo_indicators['social_proof'] = sum(1 for keyword in social_proof if keyword in content) / len(social_proof)
        
        # Scarcity
        scarcity = ['limited', 'few left', 'running out', 'last chance', 'only', 'rare', 'sold out soon']
        fomo_indicators['scarcity'] = sum(1 for phrase in scarcity if phrase in content) / len(scarcity)
        
        # Personal relevance (based on user profile)
        user_interests = user_profile.get('interests', [])
        relevance_score = 0.0
        if user_interests:
            relevance_matches = sum(1 for interest in user_interests if interest.lower() in content)
            relevance_score = relevance_matches / len(user_interests)
        fomo_indicators['personal_relevance'] = relevance_score
        
        # Calculate overall FOMO risk
        weights = [0.25, 0.2, 0.15, 0.3, 0.1]  # Personal relevance weighted highest
        fomo_risk = sum(indicator * weight for indicator, weight in 
                       zip(fomo_indicators.values(), weights))
        
        return {
            'fomo_risk': min(fomo_risk, 1.0),
            'indicators': fomo_indicators,
            'recommendation': self._get_fomo_recommendation(fomo_risk)
        }
    
    def _get_fomo_recommendation(self, risk: float) -> str:
        """Get FOMO prevention recommendation"""
        if risk > 0.8:
            return "🔴 High FOMO risk - Check immediately to avoid missing out"
        elif risk > 0.6:
            return "🟡 Moderate FOMO risk - Review when convenient"
        elif risk > 0.4:
            return "🟢 Low FOMO risk - Can be reviewed later"
        else:
            return "⚪ Minimal FOMO risk - No urgency"

class EmergingTrendDetector:
    """Detect emerging trends and topics before they become mainstream"""
    
    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self.trend_history = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def detect_emerging_trends(self, messages_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect emerging trends in message content"""
        if messages_df.empty:
            return []
        
        trends = []
        
        # Recent time window
        cutoff_time = datetime.now() - timedelta(hours=self.window_hours)
        recent_messages = messages_df[messages_df['timestamp'] >= cutoff_time]
        
        if recent_messages.empty:
            return trends
        
        # Extract keywords from recent messages
        all_content = ' '.join(recent_messages['content'].dropna().str.lower())
        words = re.findall(r'\b\w{4,15}\b', all_content)  # 4-15 character words
        word_counts = Counter(words)
        
        # Compare with historical frequency
        historical_messages = messages_df[messages_df['timestamp'] < cutoff_time]
        if not historical_messages.empty:
            historical_content = ' '.join(historical_messages['content'].dropna().str.lower())
            historical_words = re.findall(r'\b\w{4,15}\b', historical_content)
            historical_counts = Counter(historical_words)
            
            # Calculate trend scores
            for word, recent_count in word_counts.most_common(50):
                if recent_count < 3:  # Minimum threshold
                    continue
                
                historical_count = historical_counts.get(word, 0)
                historical_rate = historical_count / max(len(historical_messages), 1) * 1000
                recent_rate = recent_count / len(recent_messages) * 1000
                
                # Trend score: recent rate vs historical rate
                if historical_rate > 0:
                    trend_score = (recent_rate - historical_rate) / historical_rate
                else:
                    trend_score = recent_rate  # New word
                
                if trend_score > 0.5:  # Significant increase
                    trends.append({
                        'keyword': word,
                        'trend_score': min(trend_score, 10.0),  # Cap at 10x
                        'recent_count': recent_count,
                        'historical_rate': historical_rate,
                        'recent_rate': recent_rate,
                        'confidence': min(recent_count / 10, 1.0),  # More occurrences = higher confidence
                        'status': 'emerging' if historical_rate == 0 else 'trending_up'
                    })
        
        # Sort by trend score
        trends.sort(key=lambda x: x['trend_score'], reverse=True)
        
        return trends[:10]  # Top 10 trends
    
    def detect_conversation_anomalies(self, hourly_activity: pd.Series) -> List[Dict[str, Any]]:
        """Detect unusual activity patterns that might indicate important events"""
        if len(hourly_activity) < 24:  # Need at least 24 hours of data
            return []
        
        anomalies = []
        
        # Prepare data for anomaly detection
        activity_data = hourly_activity.values.reshape(-1, 1)
        
        # Fit anomaly detector
        self.anomaly_detector.fit(activity_data)
        anomaly_scores = self.anomaly_detector.decision_function(activity_data)
        is_anomaly = self.anomaly_detector.predict(activity_data) == -1
        
        # Find anomalous hours
        for i, (is_anom, score) in enumerate(zip(is_anomaly, anomaly_scores)):
            if is_anom and score < -0.1:  # Significant anomaly
                hour_timestamp = hourly_activity.index[i]
                anomalies.append({
                    'timestamp': hour_timestamp,
                    'activity_level': hourly_activity.iloc[i],
                    'anomaly_score': abs(score),
                    'type': 'high_activity' if hourly_activity.iloc[i] > hourly_activity.mean() else 'low_activity',
                    'description': f"Unusual {'high' if hourly_activity.iloc[i] > hourly_activity.mean() else 'low'} activity detected"
                })
        
        return sorted(anomalies, key=lambda x: x['anomaly_score'], reverse=True)[:5]


# Integration functions
def create_comprehensive_prediction_engine(db_path: str) -> Dict[str, Any]:
    """Create integrated prediction engine with all components"""
    return {
        'event_predictor': AdvancedEventPredictor(),
        'importance_analyzer': AdvancedImportanceAnalyzer(),
        'behavior_analyzer': BehaviorPatternAnalyzer(),
        'fomo_system': FOMAPreventionSystem(),
        'trend_detector': EmergingTrendDetector()
    }

def generate_comprehensive_insights(engines: Dict[str, Any], 
                                   messages_df: pd.DataFrame,
                                   user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate comprehensive insights using all prediction engines"""
    insights = {
        'event_predictions': [],
        'importance_alerts': [],
        'behavior_insights': [],
        'fomo_alerts': [],
        'emerging_trends': [],
        'anomalies': []
    }
    
    if messages_df.empty:
        return insights
    
    # Event predictions
    event_patterns = engines['event_predictor'].analyze_seasonal_patterns(
        messages_df.copy(), 'general_events'
    )
    insights['event_predictions'] = [
        {
            'type': pattern.pattern_type,
            'next_occurrence': pattern.next_occurrence.isoformat(),
            'confidence': pattern.confidence,
            'strength': pattern.strength
        }
        for pattern in event_patterns
    ]
    
    # Emerging trends
    insights['emerging_trends'] = engines['trend_detector'].detect_emerging_trends(messages_df)
    
    # Anomaly detection
    if 'timestamp' in messages_df.columns:
        hourly_activity = messages_df.set_index('timestamp').resample('H').size()
        insights['anomalies'] = engines['trend_detector'].detect_conversation_anomalies(hourly_activity)
    
    # User behavior insights (if profile provided)
    if user_profile:
        current_attention = engines['behavior_analyzer'].predict_user_attention(user_profile)
        insights['behavior_insights'] = {
            'current_attention': current_attention,
            'profile_summary': user_profile
        }
    
    return insights