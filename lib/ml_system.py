"""
ML System for Discord Message Intelligence
Production-ready ML system for importance scoring, pattern recognition, and personalization.
"""

import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
import logging
from dataclasses import dataclass
from enum import Enum

# ML dependencies
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re

logger = logging.getLogger('ml_system')

class ImportanceLevel(Enum):
    NOISE = 0      # Spam, irrelevant chatter
    NORMAL = 1     # Regular conversation
    IMPORTANT = 2  # Mentions user, group activities
    URGENT = 3     # Time-sensitive, consensus moments

@dataclass
class MessageFeatures:
    """Container for all message features"""
    # Text features
    text_length: int
    word_count: int
    exclamation_count: int
    question_count: int
    caps_ratio: float
    link_count: int
    emoji_count: int
    
    # Temporal features
    hour_of_day: int
    day_of_week: int
    time_since_last_message: float  # minutes
    
    # Social features
    mentions_count: int
    mentions_user: bool
    author_message_count_24h: int
    
    # Pattern features
    contains_money: bool
    contains_numbers: bool
    contains_event_keywords: bool
    contains_purchase_keywords: bool
    
    # TF-IDF features will be added separately as they're sparse
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.text_length, self.word_count, self.exclamation_count,
            self.question_count, self.caps_ratio, self.link_count,
            self.emoji_count, self.hour_of_day, self.day_of_week,
            self.time_since_last_message, self.mentions_count,
            int(self.mentions_user), self.author_message_count_24h,
            int(self.contains_money), int(self.contains_numbers),
            int(self.contains_event_keywords), int(self.contains_purchase_keywords)
        ])

class FeatureExtractor:
    """Extract comprehensive features from Discord messages"""
    
    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id
        
        # Pattern matching
        self.money_pattern = re.compile(r'\$\d+|\d+\s*(?:dollars?|bucks?|€|£|\d+\.\d+)')
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]')
        self.link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Event and purchase keywords
        self.event_keywords = {
            'meeting', 'event', 'party', 'hangout', 'gather', 'conference',
            'deadline', 'due', 'schedule', 'appointment', 'tonight', 'tomorrow',
            'today', 'urgent', 'asap', 'important', 'critical'
        }
        
        self.purchase_keywords = {
            'buy', 'purchase', 'order', 'shipping', 'delivery', 'sale',
            'discount', 'deal', 'price', 'cost', 'expensive', 'cheap',
            'amazon', 'ebay', 'store', 'shop', 'cart', 'checkout'
        }
    
    def extract_features(self, message_data: Dict[str, Any], 
                        context_messages: List[Dict[str, Any]] = None) -> MessageFeatures:
        """Extract all features from a message"""
        
        content = message_data.get('content', '')
        author_id = message_data.get('author_id', '')
        sent_at = message_data.get('sent_at', 0)
        
        # Text features
        text_length = len(content)
        words = content.split()
        word_count = len(words)
        exclamation_count = content.count('!')
        question_count = content.count('?')
        caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        
        # Pattern matching
        link_count = len(self.link_pattern.findall(content))
        emoji_count = len(self.emoji_pattern.findall(content))
        contains_money = bool(self.money_pattern.search(content.lower()))
        contains_numbers = bool(self.number_pattern.search(content))
        
        # Keyword matching
        content_lower = content.lower()
        contains_event_keywords = any(keyword in content_lower for keyword in self.event_keywords)
        contains_purchase_keywords = any(keyword in content_lower for keyword in self.purchase_keywords)
        
        # Temporal features
        dt = datetime.fromtimestamp(sent_at)
        hour_of_day = dt.hour
        day_of_week = dt.weekday()
        
        # Social features
        mentions_count = content.count('@')
        mentions_user = self.user_id and f'@{self.user_id}' in content
        
        # Context features
        time_since_last_message = 0.0
        author_message_count_24h = 0
        
        if context_messages:
            # Find time since last message
            sorted_messages = sorted(context_messages, key=lambda x: x.get('sent_at', 0))
            for i, msg in enumerate(sorted_messages):
                if msg.get('id') == message_data.get('id'):
                    if i > 0:
                        time_diff = sent_at - sorted_messages[i-1].get('sent_at', 0)
                        time_since_last_message = time_diff / 60.0  # Convert to minutes
                    break
            
            # Count author's messages in last 24 hours
            cutoff_time = sent_at - 86400  # 24 hours ago
            author_message_count_24h = sum(
                1 for msg in context_messages 
                if msg.get('author_id') == author_id and msg.get('sent_at', 0) > cutoff_time
            )
        
        return MessageFeatures(
            text_length=text_length,
            word_count=word_count,
            exclamation_count=exclamation_count,
            question_count=question_count,
            caps_ratio=caps_ratio,
            link_count=link_count,
            emoji_count=emoji_count,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            time_since_last_message=time_since_last_message,
            mentions_count=mentions_count,
            mentions_user=mentions_user,
            author_message_count_24h=author_message_count_24h,
            contains_money=contains_money,
            contains_numbers=contains_numbers,
            contains_event_keywords=contains_event_keywords,
            contains_purchase_keywords=contains_purchase_keywords
        )

class MessageImportanceModel:
    """Lightweight model for scoring message importance"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.classifier = None
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.feature_extractor = None
        
        # Model paths
        self.model_path = self.model_dir / "importance_model.pkl"
        self.tfidf_path = self.model_dir / "tfidf_vectorizer.pkl"
        self.scaler_path = self.model_dir / "feature_scaler.pkl"
        
        # Performance tracking
        self.performance_history = []
        
    def _create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline for incremental learning"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SGDClassifier(
                loss='log_loss',  # For probability estimates
                alpha=0.01,
                random_state=42,
                class_weight='balanced'
            ))
        ])
    
    def prepare_training_data(self, messages_df: pd.DataFrame, 
                            user_feedback: Dict[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from messages and user feedback"""
        
        features_list = []
        tfidf_features_list = []
        labels = []
        
        # Extract features for all messages
        for _, row in messages_df.iterrows():
            message_data = {
                'id': row['id'],
                'content': row['content'],
                'author_id': str(row.get('author_id', '')),
                'sent_at': row['sent_at']
            }
            
            # Get context messages (previous 50 messages for context)
            context_messages = []  # In production, fetch from DB
            
            features = self.feature_extractor.extract_features(message_data, context_messages)
            features_list.append(features.to_array())
            
            # TF-IDF features
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2
                )
                # Fit on all messages
                all_content = messages_df['content'].fillna('').tolist()
                self.tfidf_vectorizer.fit(all_content)
            
            tfidf_features = self.tfidf_vectorizer.transform([message_data['content']]).toarray()
            tfidf_features_list.append(tfidf_features[0])
            
            # Generate labels (bootstrap with heuristics, then use feedback)
            message_id = row['id']
            if user_feedback and message_id in user_feedback:
                labels.append(user_feedback[message_id])
            else:
                # Heuristic labeling for bootstrap
                label = self._heuristic_label(features, message_data['content'])
                labels.append(label)
        
        # Combine structured and TF-IDF features
        structured_features = np.array(features_list)
        tfidf_features = np.array(tfidf_features_list)
        combined_features = np.hstack([structured_features, tfidf_features])
        
        return combined_features, np.array(labels)
    
    def _heuristic_label(self, features: MessageFeatures, content: str) -> int:
        """Bootstrap labeling using heuristics"""
        
        # Noise indicators
        if features.text_length < 5 or features.word_count < 2:
            return ImportanceLevel.NOISE.value
        
        # Urgent indicators
        urgent_keywords = ['urgent', 'asap', 'emergency', 'critical', 'deadline']
        if any(keyword in content.lower() for keyword in urgent_keywords):
            return ImportanceLevel.URGENT.value
        
        # Important indicators
        if (features.mentions_user or 
            features.contains_event_keywords or 
            features.contains_purchase_keywords or
            features.exclamation_count >= 2):
            return ImportanceLevel.IMPORTANT.value
        
        # Default to normal
        return ImportanceLevel.NORMAL.value
    
    def train(self, messages_df: pd.DataFrame, user_feedback: Dict[int, int] = None):
        """Train the importance model"""
        
        logger.info(f"Training importance model with {len(messages_df)} messages")
        
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
        
        X, y = self.prepare_training_data(messages_df, user_feedback)
        
        if len(X) == 0:
            logger.warning("No training data available")
            return
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train pipeline
        self.classifier = self._create_pipeline()
        self.classifier.fit(X_train, y_train)
        
        # Validation
        val_predictions = self.classifier.predict(X_val)
        val_score = self.classifier.score(X_val, y_val)
        
        # Log performance
        performance = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': val_score,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'classification_report': classification_report(y_val, val_predictions, output_dict=True)
        }
        
        self.performance_history.append(performance)
        logger.info(f"Model trained with accuracy: {val_score:.3f}")
        
        # Save model
        self.save_model()
    
    def predict_importance(self, message_data: Dict[str, Any], 
                          context_messages: List[Dict[str, Any]] = None) -> Tuple[int, float]:
        """Predict importance level and confidence for a message"""
        
        if self.classifier is None:
            logger.warning("Model not trained. Using heuristic labeling.")
            features = self.feature_extractor.extract_features(message_data, context_messages)
            label = self._heuristic_label(features, message_data.get('content', ''))
            return label, 0.5  # Low confidence
        
        # Extract features
        features = self.feature_extractor.extract_features(message_data, context_messages)
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform([message_data.get('content', '')]).toarray()
        
        # Combine features
        combined_features = np.hstack([features.to_array().reshape(1, -1), tfidf_features])
        
        # Predict
        prediction = self.classifier.predict(combined_features)[0]
        probabilities = self.classifier.predict_proba(combined_features)[0]
        confidence = np.max(probabilities)
        
        return int(prediction), float(confidence)
    
    def partial_fit(self, new_messages_df: pd.DataFrame, user_feedback: Dict[int, int]):
        """Incrementally update the model with new data"""
        
        if self.classifier is None:
            logger.warning("Model not initialized. Training from scratch.")
            self.train(new_messages_df, user_feedback)
            return
        
        logger.info(f"Incrementally training with {len(new_messages_df)} new messages")
        
        X_new, y_new = self.prepare_training_data(new_messages_df, user_feedback)
        
        if len(X_new) > 0:
            # Get the SGDClassifier from the pipeline
            sgd_classifier = self.classifier.named_steps['classifier']
            scaler = self.classifier.named_steps['scaler']
            
            # Scale new features
            X_new_scaled = scaler.transform(X_new)
            
            # Partial fit
            sgd_classifier.partial_fit(X_new_scaled, y_new)
            
            logger.info("Incremental training completed")
            self.save_model()
    
    def save_model(self):
        """Save the trained model"""
        if self.classifier:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            
            with open(self.tfidf_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            # Save performance history
            performance_path = self.model_dir / "performance_history.json"
            with open(performance_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                
                with open(self.tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                
                # Load performance history
                performance_path = self.model_dir / "performance_history.json"
                if performance_path.exists():
                    with open(performance_path, 'r') as f:
                        self.performance_history = json.load(f)
                
                self.feature_extractor = FeatureExtractor()
                logger.info("Model loaded successfully")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.performance_history:
            return {"status": "not_trained"}
        
        latest_performance = self.performance_history[-1]
        return {
            "status": "trained",
            "accuracy": latest_performance.get("accuracy", 0),
            "training_samples": latest_performance.get("training_samples", 0),
            "last_updated": latest_performance.get("timestamp", "unknown"),
            "total_updates": len(self.performance_history)
        }