"""
Predictive Analytics Engine for Discord Monitoring System

This module provides comprehensive predictive capabilities to anticipate:
1. Event Prediction - When group buys, announcements, and high-activity periods will occur
2. Importance Prediction - Which messages/conversations will be important
3. User Behavior Modeling - User activity patterns and information needs
4. Pattern Discovery - Recurring events and hidden dynamics
5. Actionable Insights - Forward-looking recommendations

Author: Predictive Analytics Architect
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter, defaultdict
import pickle
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictiveInsight:
    """Container for predictive insights"""
    insight_type: str  # 'event', 'importance', 'behavior', 'pattern'
    message: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class EventPrediction:
    """Container for event predictions"""
    event_type: str
    predicted_time: datetime
    confidence: float
    description: str
    triggers: List[str]

class DiscordPredictiveEngine:
    """Main predictive analytics engine for Discord monitoring"""
    
    def __init__(self, db_path: str, model_dir: str = "models"):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}
        
        # Pattern storage
        self.patterns = {
            'group_buy_patterns': [],
            'announcement_patterns': [],
            'activity_cycles': {},
            'user_behavior_profiles': {}
        }
        
        # Load pre-trained models if they exist
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models from disk"""
        model_files = {
            'importance_classifier': 'importance_model.pkl',
            'event_predictor': 'event_model.pkl',
            'behavior_model': 'behavior_model.pkl',
            'anomaly_detector': 'anomaly_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
    
    def _save_model(self, model_name: str, model):
        """Save trained model to disk"""
        model_path = self.model_dir / f"{model_name}.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_name}")
        except Exception as e:
            logger.error(f"Failed to save {model_name}: {e}")
    
    def _get_messages_df(self, server_id: Optional[int] = None, days: int = 90) -> pd.DataFrame:
        """Fetch messages as pandas DataFrame with feature engineering"""
        conn = sqlite3.connect(self.db_path)
        
        # Calculate time threshold
        time_threshold = int((datetime.now() - timedelta(days=days)).timestamp())
        
        query = """
        SELECT m.id, m.server_id, m.channel_id, m.content, m.sent_at,
               s.name as server_name, c.name as channel_name
        FROM messages m
        JOIN servers s ON m.server_id = s.id
        JOIN channels c ON m.channel_id = c.id
        WHERE m.sent_at >= ?
        """
        params = [time_threshold]
        
        if server_id:
            query += " AND m.server_id = ?"
            params.append(server_id)
            
        query += " ORDER BY m.sent_at ASC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return df
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['sent_at'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Text features
        df['message_length'] = df['content'].str.len()
        df['word_count'] = df['content'].str.split().str.len()
        df['has_url'] = df['content'].str.contains(r'http[s]?://', regex=True, na=False)
        df['has_mention'] = df['content'].str.contains(r'@\w+', regex=True, na=False)
        df['has_emoji'] = df['content'].str.contains(r':[a-zA-Z0-9_]+:', regex=True, na=False)
        df['exclamation_count'] = df['content'].str.count('!')
        df['question_count'] = df['content'].str.count('\?')
        df['caps_ratio'] = df['content'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if x else 0)
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive features for predictive modeling"""
        if df.empty:
            return df
            
        features = []
        
        # Group by time windows for temporal features
        df_hourly = df.groupby([df['timestamp'].dt.floor('H'), 'channel_id']).agg({
            'id': 'count',
            'message_length': 'mean',
            'word_count': 'mean',
            'has_url': 'sum',
            'has_mention': 'sum',
            'exclamation_count': 'sum',
            'question_count': 'sum',
            'caps_ratio': 'mean'
        }).reset_index()
        df_hourly.columns = ['hour_window', 'channel_id', 'message_count', 'avg_length', 
                            'avg_words', 'url_count', 'mention_count', 'exclamation_total',
                            'question_total', 'avg_caps_ratio']
        
        return df_hourly
    
    # 1. EVENT PREDICTION
    def predict_group_buys(self, df: pd.DataFrame) -> List[EventPrediction]:
        """Predict when group buy events will occur"""
        predictions = []
        
        # Group buy indicators
        group_buy_keywords = [
            r'group\s?buy', r'gb\b', r'interest\s?check', r'ic\b',
            r'drop\s?date', r'preorder', r'limited\s?edition',
            r'restocking', r'available\s?now', r'price\s?drop'
        ]
        
        # Find historical group buy patterns
        pattern = '|'.join(group_buy_keywords)
        gb_messages = df[df['content'].str.contains(pattern, case=False, regex=True, na=False)]
        
        if not gb_messages.empty:
            # Analyze temporal patterns
            gb_messages['week'] = gb_messages['timestamp'].dt.isocalendar().week
            gb_messages['weekday'] = gb_messages['timestamp'].dt.dayofweek
            
            # Weekly patterns
            weekly_counts = gb_messages.groupby('week').size()
            if len(weekly_counts) > 2:
                # Predict next likely week
                recent_avg = weekly_counts.tail(4).mean()
                if recent_avg > weekly_counts.mean():
                    next_week_start = datetime.now() + timedelta(days=(7 - datetime.now().weekday()))
                    predictions.append(EventPrediction(
                        event_type='group_buy',
                        predicted_time=next_week_start,
                        confidence=min(0.8, recent_avg / weekly_counts.max()),
                        description=f"Group buy activity expected (avg {recent_avg:.1f} mentions/week recently)",
                        triggers=list(set(gb_messages['content'].str.extract(f'({pattern})')[0].dropna()))
                    ))
            
            # Daily patterns
            daily_pattern = gb_messages.groupby('weekday').size()
            most_active_day = daily_pattern.idxmax()
            if daily_pattern[most_active_day] > daily_pattern.mean() * 1.5:
                days_until = (most_active_day - datetime.now().weekday()) % 7
                next_active_day = datetime.now() + timedelta(days=days_until)
                predictions.append(EventPrediction(
                    event_type='group_buy',
                    predicted_time=next_active_day,
                    confidence=daily_pattern[most_active_day] / daily_pattern.sum(),
                    description=f"Group buys typically peak on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][most_active_day]}",
                    triggers=['weekly_pattern']
                ))
        
        return predictions
    
    def predict_announcements(self, df: pd.DataFrame) -> List[EventPrediction]:
        """Predict important announcements"""
        predictions = []
        
        # Announcement indicators
        announcement_keywords = [
            r'announcement', r'important', r'attention', r'notice',
            r'update', r'news', r'breaking', r'alert', r'reminder'
        ]
        
        pattern = '|'.join(announcement_keywords)
        announce_messages = df[df['content'].str.contains(pattern, case=False, regex=True, na=False)]
        
        if not announce_messages.empty:
            # Look for patterns in announcement timing
            announce_messages['hour'] = announce_messages['timestamp'].dt.hour
            announce_messages['day'] = announce_messages['timestamp'].dt.day
            
            # Peak announcement hours
            hourly_pattern = announce_messages.groupby('hour').size()
            if len(hourly_pattern) > 0:
                peak_hour = hourly_pattern.idxmax()
                today = datetime.now().date()
                next_peak = datetime.combine(today, datetime.min.time().replace(hour=peak_hour))
                if next_peak < datetime.now():
                    next_peak += timedelta(days=1)
                
                predictions.append(EventPrediction(
                    event_type='announcement',
                    predicted_time=next_peak,
                    confidence=hourly_pattern[peak_hour] / hourly_pattern.sum(),
                    description=f"Announcements typically occur around {peak_hour}:00",
                    triggers=['hourly_pattern']
                ))
        
        return predictions
    
    def predict_high_activity_periods(self, df: pd.DataFrame) -> List[EventPrediction]:
        """Predict when high activity periods will occur"""
        predictions = []
        
        if df.empty:
            return predictions
            
        # Hourly message counts
        hourly_activity = df.groupby(df['timestamp'].dt.floor('H')).size()
        activity_threshold = hourly_activity.quantile(0.8)  # Top 20% activity
        
        # Find patterns in high activity
        high_activity_hours = hourly_activity[hourly_activity >= activity_threshold]
        if not high_activity_hours.empty:
            # Convert to DataFrame for easier manipulation
            activity_df = high_activity_hours.reset_index()
            activity_df['hour'] = activity_df['timestamp'].dt.hour
            activity_df['weekday'] = activity_df['timestamp'].dt.dayofweek
            
            # Most active hours
            peak_hours = activity_df.groupby('hour').size().sort_values(ascending=False).head(3)
            for hour, count in peak_hours.items():
                next_occurrence = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
                if next_occurrence < datetime.now():
                    next_occurrence += timedelta(days=1)
                
                predictions.append(EventPrediction(
                    event_type='high_activity',
                    predicted_time=next_occurrence,
                    confidence=count / len(high_activity_hours),
                    description=f"High activity typically occurs at {hour}:00 ({count} occurrences)",
                    triggers=['hourly_activity_pattern']
                ))
        
        return predictions
    
    # 2. IMPORTANCE PREDICTION
    def train_importance_model(self, df: pd.DataFrame):
        """Train model to predict message importance"""
        if df.empty:
            return
            
        # Create importance labels based on engagement heuristics
        df['importance_score'] = (
            df['message_length'] / df['message_length'].max() * 0.3 +
            df['exclamation_count'] * 0.2 +
            df['has_url'].astype(int) * 0.3 +
            df['has_mention'].astype(int) * 0.2
        )
        
        # Binary classification: important (top 20%) vs not important
        importance_threshold = df['importance_score'].quantile(0.8)
        df['is_important'] = (df['importance_score'] >= importance_threshold).astype(int)
        
        # Features for model
        feature_columns = [
            'hour', 'day_of_week', 'message_length', 'word_count',
            'has_url', 'has_mention', 'has_emoji', 'exclamation_count',
            'question_count', 'caps_ratio'
        ]
        
        X = df[feature_columns].fillna(0)
        y = df['is_important']
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Store model
        self.models['importance_classifier'] = model
        self._save_model('importance_classifier', model)
        
        logger.info(f"Trained importance model on {len(df)} messages")
    
    def predict_message_importance(self, message_data: Dict) -> float:
        """Predict importance score for a new message"""
        if 'importance_classifier' not in self.models:
            return 0.5  # Default neutral importance
            
        # Extract features
        features = np.array([[
            message_data.get('hour', 12),
            message_data.get('day_of_week', 0),
            len(message_data.get('content', '')),
            len(message_data.get('content', '').split()),
            1 if re.search(r'http[s]?://', message_data.get('content', '')) else 0,
            1 if re.search(r'@\w+', message_data.get('content', '')) else 0,
            1 if re.search(r':[a-zA-Z0-9_]+:', message_data.get('content', '')) else 0,
            message_data.get('content', '').count('!'),
            message_data.get('content', '').count('?'),
            sum(1 for c in message_data.get('content', '') if c.isupper()) / max(len(message_data.get('content', '')), 1)
        ]])
        
        return self.models['importance_classifier'].predict_proba(features)[0][1]
    
    # 3. USER BEHAVIOR MODELING
    def analyze_user_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior patterns from message data"""
        patterns = {}
        
        if df.empty:
            return patterns
            
        # Activity by hour
        hourly_activity = df.groupby('hour').size()
        patterns['peak_hours'] = hourly_activity.nlargest(3).index.tolist()
        patterns['low_hours'] = hourly_activity.nsmallest(3).index.tolist()
        
        # Activity by day of week
        daily_activity = df.groupby('day_of_week').size()
        patterns['peak_days'] = daily_activity.nlargest(2).index.tolist()
        patterns['low_days'] = daily_activity.nsmallest(2).index.tolist()
        
        # Content preferences
        content_features = df.groupby('channel_name').agg({
            'has_url': 'sum',
            'message_length': 'mean',
            'word_count': 'mean'
        })
        
        patterns['preferred_channels'] = content_features.index.tolist()
        patterns['avg_message_length'] = df['message_length'].mean()
        patterns['activity_score'] = len(df) / max(df['timestamp'].max() - df['timestamp'].min(), timedelta(days=1)).days
        
        return patterns
    
    def predict_user_availability(self, current_time: datetime = None) -> Dict[str, float]:
        """Predict user availability/attention based on historical patterns"""
        if current_time is None:
            current_time = datetime.now()
            
        # Default patterns if no historical data
        default_patterns = {
            'weekday_morning': 0.7,  # 9-12
            'weekday_afternoon': 0.5,  # 12-17
            'weekday_evening': 0.8,   # 17-22
            'weekend_morning': 0.4,   # 9-12
            'weekend_afternoon': 0.6, # 12-17
            'weekend_evening': 0.7    # 17-22
        }
        
        hour = current_time.hour
        is_weekend = current_time.weekday() >= 5
        
        if is_weekend:
            if 9 <= hour < 12:
                return {'availability': default_patterns['weekend_morning'], 'period': 'weekend_morning'}
            elif 12 <= hour < 17:
                return {'availability': default_patterns['weekend_afternoon'], 'period': 'weekend_afternoon'}
            elif 17 <= hour < 22:
                return {'availability': default_patterns['weekend_evening'], 'period': 'weekend_evening'}
        else:
            if 9 <= hour < 12:
                return {'availability': default_patterns['weekday_morning'], 'period': 'weekday_morning'}
            elif 12 <= hour < 17:
                return {'availability': default_patterns['weekday_afternoon'], 'period': 'weekday_afternoon'}
            elif 17 <= hour < 22:
                return {'availability': default_patterns['weekday_evening'], 'period': 'weekday_evening'}
        
        return {'availability': 0.3, 'period': 'off_hours'}
    
    # 4. PATTERN DISCOVERY
    def discover_recurring_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Automatically discover recurring events and patterns"""
        patterns = []
        
        if df.empty:
            return patterns
            
        # Weekly recurring patterns
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['weekday'] = df['timestamp'].dt.day_name()
        
        # Find topics that appear regularly
        # Simple keyword extraction
        all_content = ' '.join(df['content'].dropna().str.lower())
        words = re.findall(r'\b\w{4,}\b', all_content)
        word_counts = Counter(words)
        
        # Topics that appear across multiple weeks
        for word, count in word_counts.most_common(20):
            if count < 5:  # Filter low frequency
                continue
                
            word_messages = df[df['content'].str.contains(word, case=False, na=False)]
            if len(word_messages) < 3:
                continue
                
            weeks_with_topic = word_messages['week'].nunique()
            total_weeks = df['week'].nunique()
            
            if weeks_with_topic >= max(2, total_weeks * 0.3):  # Appears in 30%+ of weeks
                # Find temporal pattern
                weekly_pattern = word_messages.groupby(['week', 'weekday']).size().reset_index()
                most_common_day = word_messages['weekday'].mode()
                
                if not most_common_day.empty:
                    patterns.append({
                        'type': 'recurring_topic',
                        'topic': word,
                        'frequency': count,
                        'weeks_active': weeks_with_topic,
                        'most_common_day': most_common_day.iloc[0],
                        'regularity_score': weeks_with_topic / total_weeks,
                        'description': f"Topic '{word}' appears regularly, especially on {most_common_day.iloc[0]}s"
                    })
        
        return patterns
    
    def detect_conversation_lifecycles(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect patterns in conversation lifecycles"""
        lifecycles = []
        
        if df.empty:
            return lifecycles
            
        # Group by day and channel to analyze conversation patterns
        daily_channel_activity = df.groupby([
            df['timestamp'].dt.date, 
            'channel_name'
        ]).agg({
            'id': 'count',
            'message_length': 'mean',
            'word_count': 'sum'
        }).reset_index()
        
        # Find channels with consistent conversation patterns
        for channel in df['channel_name'].unique():
            channel_data = daily_channel_activity[daily_channel_activity['channel_name'] == channel]
            if len(channel_data) < 7:  # Need at least a week of data
                continue
                
            # Calculate conversation intensity patterns
            avg_messages = channel_data['id'].mean()
            std_messages = channel_data['id'].std()
            
            # Days with high activity (potential conversation peaks)
            high_activity_days = channel_data[channel_data['id'] > avg_messages + std_messages]
            
            if not high_activity_days.empty:
                lifecycles.append({
                    'type': 'conversation_lifecycle',
                    'channel': channel,
                    'avg_daily_messages': avg_messages,
                    'peak_days': len(high_activity_days),
                    'intensity_pattern': 'variable' if std_messages > avg_messages * 0.5 else 'consistent',
                    'description': f"#{channel} has {len(high_activity_days)} peak conversation days"
                })
        
        return lifecycles
    
    # 5. ACTIONABLE INSIGHTS
    def generate_predictive_insights(self, server_id: Optional[int] = None) -> List[PredictiveInsight]:
        """Generate comprehensive predictive insights"""
        insights = []
        
        # Get recent data
        df = self._get_messages_df(server_id, days=30)
        
        if df.empty:
            return [PredictiveInsight(
                insight_type='system',
                message="Insufficient data for predictions. Need more message history.",
                confidence=1.0,
                timestamp=datetime.now(),
                metadata={'reason': 'no_data'}
            )]
        
        # Event predictions
        gb_predictions = self.predict_group_buys(df)
        for pred in gb_predictions:
            insights.append(PredictiveInsight(
                insight_type='event',
                message=f"{pred.description} (Expected: {pred.predicted_time.strftime('%A, %B %d')})",
                confidence=pred.confidence,
                timestamp=datetime.now(),
                metadata={'prediction': pred}
            ))
        
        # Announcement predictions
        announce_predictions = self.predict_announcements(df)
        for pred in announce_predictions:
            insights.append(PredictiveInsight(
                insight_type='event',
                message=f"{pred.description} (Expected: {pred.predicted_time.strftime('%A at %H:%M')})",
                confidence=pred.confidence,
                timestamp=datetime.now(),
                metadata={'prediction': pred}
            ))
        
        # User behavior insights
        user_patterns = self.analyze_user_patterns(df)
        if user_patterns.get('peak_hours'):
            peak_hours_str = ', '.join([f"{h}:00" for h in user_patterns['peak_hours']])
            insights.append(PredictiveInsight(
                insight_type='behavior',
                message=f"You're most active during: {peak_hours_str}",
                confidence=0.8,
                timestamp=datetime.now(),
                metadata={'patterns': user_patterns}
            ))
        
        # Current availability prediction
        availability = self.predict_user_availability()
        if availability['availability'] < 0.5:
            insights.append(PredictiveInsight(
                insight_type='behavior',
                message=f"Low attention period detected ({availability['period']}). Consider scheduling important notifications for later.",
                confidence=availability['availability'],
                timestamp=datetime.now(),
                metadata={'availability': availability}
            ))
        
        # Pattern discoveries
        recurring_events = self.discover_recurring_events(df)
        for event in recurring_events[:3]:  # Top 3 patterns
            insights.append(PredictiveInsight(
                insight_type='pattern',
                message=event['description'],
                confidence=event['regularity_score'],
                timestamp=datetime.now(),
                metadata={'pattern': event}
            ))
        
        return insights
    
    def get_smart_notifications(self, server_id: Optional[int] = None) -> List[str]:
        """Generate smart notification recommendations"""
        notifications = []
        
        insights = self.generate_predictive_insights(server_id)
        current_time = datetime.now()
        
        for insight in insights:
            if insight.confidence > 0.6:  # High confidence insights only
                if insight.insight_type == 'event':
                    pred = insight.metadata.get('prediction')
                    if pred and pred.predicted_time:
                        hours_until = (pred.predicted_time - current_time).total_seconds() / 3600
                        if 0 < hours_until <= 24:  # Within next 24 hours
                            notifications.append(
                                f"🔮 {pred.event_type.replace('_', ' ').title()} expected in {hours_until:.1f} hours"
                            )
                
                elif insight.insight_type == 'behavior' and 'Low attention period' in insight.message:
                    notifications.append("📱 Consider checking Discord later for better focus")
                
                elif insight.insight_type == 'pattern' and insight.confidence > 0.7:
                    notifications.append(f"📊 Pattern detected: {insight.message}")
        
        return notifications
    
    def train_all_models(self, server_id: Optional[int] = None):
        """Train all predictive models with available data"""
        logger.info("Training all predictive models...")
        
        # Get training data
        df = self._get_messages_df(server_id, days=90)  # 3 months of data
        
        if df.empty:
            logger.warning("No data available for training")
            return
        
        # Train importance model
        self.train_importance_model(df.copy())
        
        # Store patterns for future use
        self.patterns['user_behavior_profiles'] = self.analyze_user_patterns(df)
        self.patterns['recurring_events'] = self.discover_recurring_events(df)
        self.patterns['conversation_lifecycles'] = self.detect_conversation_lifecycles(df)
        
        logger.info("Model training completed")


# Helper functions for integration
def get_predictions_for_dashboard(db_path: str, server_id: Optional[int] = None) -> Dict[str, Any]:
    """Get predictions formatted for dashboard display"""
    engine = DiscordPredictiveEngine(db_path)
    insights = engine.generate_predictive_insights(server_id)
    notifications = engine.get_smart_notifications(server_id)
    
    return {
        'insights': [
            {
                'type': insight.insight_type,
                'message': insight.message,
                'confidence': f"{insight.confidence:.0%}",
                'timestamp': insight.timestamp.strftime("%Y-%m-%d %H:%M")
            }
            for insight in insights
        ],
        'notifications': notifications,
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def initialize_predictive_system(db_path: str) -> DiscordPredictiveEngine:
    """Initialize the predictive system and train initial models"""
    engine = DiscordPredictiveEngine(db_path)
    engine.train_all_models()  # Train on all available data
    return engine