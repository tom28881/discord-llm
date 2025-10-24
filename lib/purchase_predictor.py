"""
Purchase Prediction Module for Discord Monitoring
Detects and predicts group purchases with Czech language support
"""

import re
import sqlite3
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class PurchasePredictor:
    """Predicts group purchases from Discord messages"""
    
    def __init__(self, db_path: str = 'data/db.sqlite'):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Czech and English purchase patterns
        self.purchase_patterns = {
            'direct_purchase': [
                r'\b(group\s*buy|společný\s*nákup|společnej\s*nákup)\b',
                r'\b(split\s*cost|rozdělíme\s*náklady|půjdem\s*na\s*půl)\b',
                r'\b(kdo\s*chce|who\'s\s*in|kdo\s*má\s*zájem|kdo\s*jde\s*do\s*toho)\b',
                r'\b(příspěvek|contribution|složíme\s*se|dáme\s*se\s*dohromady)\b',
            ],
            'purchase_intent': [
                r'\b(koupit|buy|nakoupit|pořídit|sehnat)\b',
                r'\b(objednat|order|objednáme|objednávka)\b',
                r'\b(zaplatit|pay|platit|platba|payment)\b',
                r'\b(cena|price|stojí|costs?|kolik|how\s*much)\b',
            ],
            'urgency': [
                r'\b(poslední\s*šance|last\s*chance|končí|ends?)\b',
                r'\b(rychle|hurry|pospěš|dnes|today|zítra|tomorrow)\b',
                r'\b(sleva|discount|akce|sale|výprodej)\b',
                r'\b(limitovan[éý]|limited|omezen[éý]|pouze)\b',
            ],
            'participation': [
                r'\b(já\s*jo|já\s*chci|count\s*me\s*in|já\s*beru)\b',
                r'\b(jsem\s*pro|i\'m\s*in|jdu\s*do\s*toho|beru\s*to)\b',
                r'\b(počítej\s*se\s*mnou|vezmu|bereš|berete)\b',
                r'\+1|\bjo+\b|\byes+\b|\btaky\b|\bme\s*too\b',
            ],
            'money': [
                r'\b\d+\s*(kč|czk|eur|euro|€|usd|\$|korun)\b',
                r'\b(korun|crowns|peněz|money|bucks)\b',
                r'\b\d+[kK]\b',  # 5k, 10K etc.
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for category, patterns in self.purchase_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.UNICODE) 
                for pattern in patterns
            ]
        
        # Load or train model
        self._load_or_train_model()
    
    def _load_or_train_model(self):
        """Load existing model or train new one"""
        # For now, we'll use rule-based detection
        # In production, this would load a trained model
        self.model_trained = False
    
    def predict_purchase(self, messages: List[Dict[str, Any]], 
                        channel_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict if messages indicate a group purchase
        
        Returns:
            Dict with probability, type, participants, and metadata
        """
        if not messages:
            return {
                'probability': 0.0,
                'prediction_type': 'no_data',
                'metadata': {}
            }
        
        # Extract features
        features = self._extract_features(messages)
        
        # Calculate probability
        probability = self._calculate_probability(features)
        
        # Detect purchase type
        purchase_type = self._detect_purchase_type(features)
        
        # Extract metadata
        metadata = self._extract_metadata(messages, features)
        
        # Save prediction if high probability
        if probability > 0.7 and messages:
            self._save_prediction(messages[-1], probability, purchase_type, metadata)
        
        return {
            'probability': probability,
            'prediction_type': purchase_type,
            'confidence': features['confidence'],
            'metadata': metadata,
            'features': features
        }
    
    def _extract_features(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from messages"""
        features = {
            'direct_purchase_count': 0,
            'purchase_intent_count': 0,
            'split_payment_count': 0,
            'pricing_count': 0,
            'money_count': 0,
            'urgency_count': 0,
            'participation_count': 0,
            'money_mentions': 0,
            'question_marks': 0,
            'exclamation_marks': 0,
            'message_count': len(messages),
            'unique_users': 0,
            'time_span_minutes': 0,
            'confidence': 0.5,
            'matched_patterns': []
        }
        
        # Combine all message content
        all_text = ' '.join([msg.get('content', '') for msg in messages])
        
        # Count pattern matches
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(all_text)
                if matches:
                    features[f'{category}_count'] += len(matches)
                    features['matched_patterns'].extend(matches[:3])  # Store first 3 matches
        
        # Count punctuation
        features['question_marks'] = all_text.count('?')
        features['exclamation_marks'] = all_text.count('!')
        
        # Calculate time span
        if messages:
            timestamps = [msg.get('sent_at', 0) for msg in messages]
            if timestamps:
                time_span = max(timestamps) - min(timestamps)
                features['time_span_minutes'] = time_span / 60
        
        # Calculate confidence based on feature strength
        confidence_score = 0
        if features['direct_purchase_count'] > 0:
            confidence_score += 0.4
        if features['money_mentions'] > 0:
            confidence_score += 0.2
        if features['urgency_count'] > 0:
            confidence_score += 0.15
        if features['participation_count'] >= 2:
            confidence_score += 0.15
        if features['purchase_intent_count'] > 0:
            confidence_score += 0.1
        
        features['confidence'] = min(confidence_score, 1.0)
        
        return features
    
    def _calculate_probability(self, features: Dict[str, Any]) -> float:
        """Calculate purchase probability from features"""
        score = 0.0
        
        # Strong indicators
        if features['direct_purchase_count'] > 0:
            score += 0.5
        
        # Medium indicators
        if features['money_mentions'] > 0:
            score += 0.25
        
        if features['urgency_count'] > 0:
            score += 0.15
        
        if features['participation_count'] >= 3:
            score += 0.2
        elif features['participation_count'] >= 1:
            score += 0.1
        
        # Weak indicators
        if features['purchase_intent_count'] > 0:
            score += 0.1
        
        if features['question_marks'] > 2:
            score += 0.05
        
        # Time-based adjustment
        if features['time_span_minutes'] < 30:  # Quick discussion
            score *= 1.1
        
        # Normalize to 0-1
        return min(score, 1.0)
    
    def _detect_purchase_type(self, features: Dict[str, Any]) -> str:
        """Detect the type of purchase"""
        if features['direct_purchase_count'] > 0:
            if features['urgency_count'] > 0:
                return 'urgent_group_buy'
            return 'group_buy'
        elif features['money_mentions'] > 0 and features['participation_count'] > 0:
            return 'cost_splitting'
        elif features['urgency_count'] > 0:
            return 'time_sensitive_opportunity'
        elif features['purchase_intent_count'] > 0:
            return 'purchase_discussion'
        else:
            return 'potential_purchase'
    
    def _extract_metadata(self, messages: List[Dict[str, Any]], 
                         features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata about the purchase"""
        metadata = {
            'price_mentions': [],
            'deadlines': [],
            'participants': [],
            'key_phrases': features.get('matched_patterns', [])[:5],
            'urgency_level': 0,
            'purchase_items': []
        }
        
        all_text = ' '.join([msg.get('content', '') for msg in messages])
        
        # Extract prices
        price_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*(kč|czk|eur|euro|€|usd|\$|korun)\b', re.IGNORECASE)
        prices = price_pattern.findall(all_text)
        metadata['price_mentions'] = [f"{amount} {currency}" for amount, currency in prices[:3]]
        
        # Extract deadlines
        deadline_patterns = [
            r'do\s+(\w+)',  # do pátku, do zítřka
            r'končí\s+(\w+)',  # končí dnes, končí zítra
            r'(dnes|zítra|pozítří|today|tomorrow)',
            r'(\d{1,2}\.\s?\d{1,2}\.?)',  # 15.3., 15. 3.
        ]
        
        for pattern in deadline_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            metadata['deadlines'].extend(matches[:2])
        
        # Calculate urgency level (0-5)
        urgency = min(features.get('urgency_count', 0) + 
                     len(metadata['deadlines']) + 
                     (features.get('exclamation_marks', 0) // 2), 5)
        metadata['urgency_level'] = urgency
        
        # Extract potential purchase items (nouns after buy/koupit)
        item_pattern = re.compile(r'(?:koupit|buy|nakoupit|objednat|order)\s+(\w+(?:\s+\w+)?)', re.IGNORECASE)
        items = item_pattern.findall(all_text)
        metadata['purchase_items'] = list(set(items[:3]))
        
        return metadata
    
    def _save_prediction(self, message: Dict[str, Any], probability: float, 
                        prediction_type: str, metadata: Dict[str, Any]):
        """Save prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO purchase_predictions 
                (message_id, channel_id, probability, prediction_type, metadata, predicted_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                message.get('id'),
                message.get('channel_id'),
                probability,
                prediction_type,
                json.dumps(metadata),
                int(datetime.now().timestamp())
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving prediction: {e}")
    
    def get_recent_predictions(self, hours: int = 24, min_probability: float = 0.7) -> List[Dict[str, Any]]:
        """Get recent high-probability predictions"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            time_threshold = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            cursor.execute('''
                SELECT pp.*, m.content, m.sent_at, c.name as channel_name
                FROM purchase_predictions pp
                JOIN messages m ON pp.message_id = m.id
                LEFT JOIN channels c ON pp.channel_id = c.id
                WHERE pp.predicted_at >= ? AND pp.probability >= ?
                ORDER BY pp.probability DESC
                LIMIT 20
            ''', (time_threshold, min_probability))
            
            predictions = []
            for row in cursor.fetchall():
                pred = dict(row)
                if pred.get('metadata'):
                    pred['metadata'] = json.loads(pred['metadata'])
                predictions.append(pred)
            
            conn.close()
            return predictions
            
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return []
    
    def analyze_channel_history(self, channel_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze channel history for purchase patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            time_threshold = int((datetime.now() - timedelta(days=days)).timestamp())
            
            # Get messages from channel
            cursor.execute('''
                SELECT * FROM messages
                WHERE channel_id = ? AND sent_at >= ?
                ORDER BY sent_at
            ''', (channel_id, time_threshold))
            
            messages = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            # Group messages into potential purchase discussions
            purchase_discussions = []
            current_group = []
            last_timestamp = 0
            
            for msg in messages:
                timestamp = msg.get('sent_at', 0)
                
                # If more than 30 minutes gap, start new group
                if timestamp - last_timestamp > 1800:  # 30 minutes
                    if current_group:
                        prediction = self.predict_purchase(current_group, channel_id)
                        if prediction['probability'] > 0.5:
                            purchase_discussions.append({
                                'messages': current_group,
                                'prediction': prediction
                            })
                    current_group = [msg]
                else:
                    current_group.append(msg)
                
                last_timestamp = timestamp
            
            # Check last group
            if current_group:
                prediction = self.predict_purchase(current_group, channel_id)
                if prediction['probability'] > 0.5:
                    purchase_discussions.append({
                        'messages': current_group,
                        'prediction': prediction
                    })
            
            return {
                'channel_id': channel_id,
                'total_messages': len(messages),
                'purchase_discussions_found': len(purchase_discussions),
                'discussions': purchase_discussions[:10]  # Limit to 10 most recent
            }
            
        except Exception as e:
            print(f"Error analyzing channel: {e}")
            return {
                'channel_id': channel_id,
                'error': str(e)
            }