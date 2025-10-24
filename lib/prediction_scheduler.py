"""
Prediction Scheduler and Background Processing System

This module handles scheduled prediction tasks, background model training,
and real-time prediction serving for the Discord monitoring system.

Author: Senior Backend Architect
"""

import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import json
from pathlib import Path
from dataclasses import asdict

from .predictive_engine import DiscordPredictiveEngine, get_predictions_for_dashboard
from .database import get_unique_server_ids

logger = logging.getLogger(__name__)

class PredictionScheduler:
    """Manages scheduled prediction tasks and background processing"""
    
    def __init__(self, db_path: str, cache_dir: str = "cache"):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.engine = DiscordPredictiveEngine(db_path)
        self.is_running = False
        self.scheduler_thread = None
        
        # Cache for predictions
        self.prediction_cache = {}
        self.cache_expiry = {}
        
        # Setup scheduled tasks
        self._setup_schedule()
        
    def _setup_schedule(self):
        """Setup scheduled prediction tasks"""
        # Model retraining - daily at 2 AM
        schedule.every().day.at("02:00").do(self._retrain_models)
        
        # Prediction updates - every 30 minutes during active hours (8 AM - 10 PM)
        for hour in range(8, 23):
            for minute in [0, 30]:
                schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(self._update_predictions)
        
        # Quick prediction refresh - every 5 minutes during peak hours (5 PM - 8 PM)
        schedule.every(5).minutes.do(self._quick_prediction_update)
        
        # Cache cleanup - every hour
        schedule.every().hour.do(self._cleanup_cache)
        
        logger.info("Prediction scheduler configured")
    
    def _retrain_models(self):
        """Retrain all models with latest data"""
        logger.info("Starting scheduled model retraining...")
        try:
            server_ids = get_unique_server_ids()
            for server_id in server_ids:
                self.engine.train_all_models(server_id)
            logger.info("Model retraining completed successfully")
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def _update_predictions(self):
        """Update predictions for all servers"""
        logger.info("Updating predictions for all servers...")
        try:
            server_ids = get_unique_server_ids()
            for server_id in server_ids:
                predictions = get_predictions_for_dashboard(self.db_path, server_id)
                self._cache_predictions(server_id, predictions)
            logger.info(f"Updated predictions for {len(server_ids)} servers")
        except Exception as e:
            logger.error(f"Prediction update failed: {e}")
    
    def _quick_prediction_update(self):
        """Quick prediction update during peak hours"""
        current_hour = datetime.now().hour
        if 17 <= current_hour <= 20:  # Peak hours 5-8 PM
            logger.debug("Quick prediction update during peak hours")
            self._update_predictions()
    
    def _cleanup_cache(self):
        """Clean expired cache entries"""
        current_time = datetime.now()
        expired_keys = [
            key for key, expiry in self.cache_expiry.items()
            if expiry < current_time
        ]
        
        for key in expired_keys:
            self.prediction_cache.pop(key, None)
            self.cache_expiry.pop(key, None)
        
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def _cache_predictions(self, server_id: int, predictions: Dict[str, Any], ttl_minutes: int = 30):
        """Cache predictions with TTL"""
        cache_key = f"predictions_{server_id}"
        self.prediction_cache[cache_key] = predictions
        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=ttl_minutes)
        
        # Also save to disk for persistence
        cache_file = self.cache_dir / f"predictions_{server_id}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(predictions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save predictions to disk: {e}")
    
    def get_cached_predictions(self, server_id: int) -> Optional[Dict[str, Any]]:
        """Get cached predictions if available and not expired"""
        cache_key = f"predictions_{server_id}"
        
        # Check memory cache first
        if cache_key in self.prediction_cache:
            expiry = self.cache_expiry.get(cache_key)
            if expiry and expiry > datetime.now():
                return self.prediction_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"predictions_{server_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    predictions = json.load(f)
                
                # Check if file is recent (within 1 hour)
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age < timedelta(hours=1):
                    # Restore to memory cache
                    self._cache_predictions(server_id, predictions, ttl_minutes=60)
                    return predictions
            except Exception as e:
                logger.error(f"Failed to load predictions from disk: {e}")
        
        return None
    
    def force_prediction_update(self, server_id: int) -> Dict[str, Any]:
        """Force immediate prediction update for a specific server"""
        logger.info(f"Force updating predictions for server {server_id}")
        predictions = get_predictions_for_dashboard(self.db_path, server_id)
        self._cache_predictions(server_id, predictions)
        return predictions
    
    def start(self):
        """Start the prediction scheduler in background thread"""
        if self.is_running:
            logger.warning("Prediction scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Prediction scheduler started")
    
    def stop(self):
        """Stop the prediction scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Prediction scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        logger.info("Prediction scheduler loop started")
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait longer on error


class RealTimePredictionAPI:
    """Real-time prediction API for immediate predictions"""
    
    def __init__(self, db_path: str, scheduler: PredictionScheduler = None):
        self.db_path = db_path
        self.engine = DiscordPredictiveEngine(db_path)
        self.scheduler = scheduler
    
    def predict_message_importance(self, message_content: str, channel_id: int, 
                                 timestamp: datetime = None) -> Dict[str, Any]:
        """Predict importance of a new message in real-time"""
        if timestamp is None:
            timestamp = datetime.now()
        
        message_data = {
            'content': message_content,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'channel_id': channel_id
        }
        
        importance_score = self.engine.predict_message_importance(message_data)
        
        return {
            'importance_score': importance_score,
            'is_important': importance_score > 0.7,
            'confidence': importance_score,
            'recommendation': self._get_importance_recommendation(importance_score),
            'timestamp': timestamp.isoformat()
        }
    
    def _get_importance_recommendation(self, score: float) -> str:
        """Get recommendation based on importance score"""
        if score > 0.9:
            return "🔴 Critical - Immediate attention required"
        elif score > 0.7:
            return "🟡 Important - Review soon"
        elif score > 0.5:
            return "🟢 Normal - Standard priority"
        else:
            return "⚪ Low priority"
    
    def get_current_insights(self, server_id: int) -> Dict[str, Any]:
        """Get current predictive insights"""
        # Try cache first
        if self.scheduler:
            cached = self.scheduler.get_cached_predictions(server_id)
            if cached:
                return cached
        
        # Generate fresh predictions
        return get_predictions_for_dashboard(self.db_path, server_id)
    
    def get_notification_recommendations(self, server_id: int) -> List[str]:
        """Get smart notification recommendations"""
        notifications = self.engine.get_smart_notifications(server_id)
        
        # Add real-time context
        current_time = datetime.now()
        availability = self.engine.predict_user_availability(current_time)
        
        if availability['availability'] < 0.4:
            notifications.append(
                f"⏰ Currently in {availability['period']} - consider scheduling notifications for later"
            )
        
        return notifications
    
    def analyze_conversation_urgency(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze urgency of an ongoing conversation"""
        if not messages:
            return {'urgency': 'low', 'score': 0.0}
        
        # Analyze recent message patterns
        recent_messages = messages[-5:]  # Last 5 messages
        
        urgency_indicators = {
            'rapid_fire': 0,  # Many messages in short time
            'keywords': 0,    # Urgent keywords
            'caps': 0,        # ALL CAPS usage
            'questions': 0,   # Question marks
            'mentions': 0     # User mentions
        }
        
        urgent_keywords = [
            'urgent', 'asap', 'important', 'help', 'problem', 'issue',
            'broken', 'down', 'error', 'critical', 'emergency'
        ]
        
        for msg in recent_messages:
            content = msg.get('content', '').lower()
            
            # Check for urgent keywords
            for keyword in urgent_keywords:
                if keyword in content:
                    urgency_indicators['keywords'] += 1
            
            # Check caps ratio
            original_content = msg.get('content', '')
            if original_content:
                caps_ratio = sum(1 for c in original_content if c.isupper()) / len(original_content)
                urgency_indicators['caps'] += caps_ratio
            
            # Count questions and mentions
            urgency_indicators['questions'] += original_content.count('?')
            urgency_indicators['mentions'] += len(re.findall(r'@\w+', original_content))
        
        # Calculate time-based urgency (messages per minute)
        if len(recent_messages) > 1:
            time_span = (datetime.fromisoformat(recent_messages[-1]['timestamp']) - 
                        datetime.fromisoformat(recent_messages[0]['timestamp'])).total_seconds() / 60
            if time_span > 0:
                urgency_indicators['rapid_fire'] = len(recent_messages) / time_span
        
        # Calculate overall urgency score
        urgency_score = (
            min(urgency_indicators['rapid_fire'] * 0.3, 1.0) +
            min(urgency_indicators['keywords'] * 0.4, 1.0) +
            min(urgency_indicators['caps'] * 0.2, 1.0) +
            min(urgency_indicators['questions'] * 0.05, 0.2) +
            min(urgency_indicators['mentions'] * 0.05, 0.2)
        ) / 1.2  # Normalize
        
        # Determine urgency level
        if urgency_score > 0.7:
            urgency_level = 'critical'
        elif urgency_score > 0.5:
            urgency_level = 'high'
        elif urgency_score > 0.3:
            urgency_level = 'medium'
        else:
            urgency_level = 'low'
        
        return {
            'urgency': urgency_level,
            'score': min(urgency_score, 1.0),
            'indicators': urgency_indicators,
            'recommendation': self._get_urgency_recommendation(urgency_level, urgency_score)
        }
    
    def _get_urgency_recommendation(self, level: str, score: float) -> str:
        """Get recommendation based on urgency level"""
        recommendations = {
            'critical': f"🚨 Critical conversation detected (confidence: {score:.0%}) - Immediate attention needed",
            'high': f"⚠️ High urgency conversation (confidence: {score:.0%}) - Check within 15 minutes",
            'medium': f"🟡 Moderate urgency (confidence: {score:.0%}) - Review when convenient",
            'low': f"🟢 Normal conversation pace (confidence: {score:.0%})"
        }
        return recommendations.get(level, "Unknown urgency level")


# Global scheduler instance
_scheduler_instance = None

def get_prediction_scheduler(db_path: str) -> PredictionScheduler:
    """Get global prediction scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = PredictionScheduler(db_path)
    return _scheduler_instance

def start_prediction_system(db_path: str) -> PredictionScheduler:
    """Initialize and start the prediction system"""
    scheduler = get_prediction_scheduler(db_path)
    scheduler.start()
    return scheduler

def get_prediction_api(db_path: str) -> RealTimePredictionAPI:
    """Get real-time prediction API instance"""
    scheduler = get_prediction_scheduler(db_path)
    return RealTimePredictionAPI(db_path, scheduler)