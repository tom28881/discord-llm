"""
ML Model Serving and Deployment Architecture
Production-ready serving system for ML models with versioning, monitoring, and A/B testing.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .ml_system import MessageImportanceModel, ImportanceLevel, FeatureExtractor
from .pattern_recognition import GroupActivityDetector, FOMODetector, ActivityCluster
from .personalization import UserPersonalizationEngine
from .database import DB_NAME

logger = logging.getLogger('ml_serving')

class ModelVersion(Enum):
    CHAMPION = "champion"  # Current production model
    CHALLENGER = "challenger"  # Model being A/B tested

@dataclass
class PredictionResult:
    """Container for ML prediction results"""
    message_id: int
    importance_level: int
    confidence: float
    personalized_score: float
    processing_time_ms: float
    model_version: str
    features_used: Dict[str, Any]
    timestamp: float

@dataclass 
class ActivityPrediction:
    """Container for activity detection results"""
    activities: List[ActivityCluster]
    fomo_moments: List[ActivityCluster]
    processing_time_ms: float
    timestamp: float

class ModelRegistry:
    """Registry for managing model versions and metadata"""
    
    def __init__(self, registry_dir: str = "data/model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "model_registry.json"
        self.models_metadata = self._load_registry()
        
        # Model instances cache
        self.model_cache: Dict[str, MessageImportanceModel] = {}
        
    def register_model(self, model: MessageImportanceModel, version: str, 
                      metadata: Dict[str, Any]) -> str:
        """Register a new model version"""
        
        model_id = f"importance_model_{version}_{int(time.time())}"
        
        # Calculate model hash for integrity checking
        model_hash = self._calculate_model_hash(model)
        
        # Save model
        model_path = self.registry_dir / f"{model_id}.pkl"
        model.model_path = model_path
        model.save_model()
        
        # Update registry
        self.models_metadata[model_id] = {
            "version": version,
            "model_path": str(model_path),
            "model_hash": model_hash,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata,
            "status": "registered"
        }
        
        self._save_registry()
        logger.info(f"Registered model {model_id} version {version}")
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[MessageImportanceModel]:
        """Get model instance by ID"""
        
        if model_id in self.model_cache:
            return self.model_cache[model_id]
        
        if model_id not in self.models_metadata:
            return None
        
        try:
            # Load model
            model_path = Path(self.models_metadata[model_id]["model_path"])
            model = MessageImportanceModel(model_dir=model_path.parent)
            
            if model.load_model():
                self.model_cache[model_id] = model
                return model
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
        
        return None
    
    def get_active_model(self, version_type: ModelVersion) -> Optional[str]:
        """Get the currently active model ID for a version type"""
        
        for model_id, metadata in self.models_metadata.items():
            if metadata.get("status") == version_type.value:
                return model_id
        
        return None
    
    def promote_model(self, model_id: str, version_type: ModelVersion):
        """Promote a model to champion or challenger status"""
        
        # Demote current model of this type
        current_model = self.get_active_model(version_type)
        if current_model:
            self.models_metadata[current_model]["status"] = "archived"
        
        # Promote new model
        if model_id in self.models_metadata:
            self.models_metadata[model_id]["status"] = version_type.value
            self.models_metadata[model_id]["promoted_at"] = datetime.now().isoformat()
            
            self._save_registry()
            logger.info(f"Promoted model {model_id} to {version_type.value}")
    
    def _calculate_model_hash(self, model: MessageImportanceModel) -> str:
        """Calculate hash of model for integrity checking"""
        # This is a simplified hash - in production, you'd hash the actual model parameters
        model_info = f"{model.model_path}_{time.time()}"
        return hashlib.md5(model_info.encode()).hexdigest()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.models_metadata, f, indent=2)

class PerformanceMonitor:
    """Monitor ML model performance and system health"""
    
    def __init__(self, monitoring_dir: str = "data/monitoring"):
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics storage
        self.metrics_file = self.monitoring_dir / "performance_metrics.json"
        self.metrics_history = self._load_metrics()
        
        # Real-time metrics
        self.current_metrics = {
            "predictions_count": 0,
            "avg_response_time": 0.0,
            "error_count": 0,
            "model_accuracy": 0.0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Alerts configuration
        self.alert_thresholds = {
            "max_response_time": 1000,  # ms
            "min_accuracy": 0.7,
            "max_error_rate": 0.05
        }
        
    def record_prediction(self, prediction: PredictionResult, actual_feedback: Optional[int] = None):
        """Record a prediction for monitoring"""
        
        timestamp = datetime.now().isoformat()
        
        # Update current metrics
        self.current_metrics["predictions_count"] += 1
        
        # Update response time (moving average)
        current_avg = self.current_metrics["avg_response_time"]
        n = self.current_metrics["predictions_count"]
        self.current_metrics["avg_response_time"] = (
            (current_avg * (n - 1) + prediction.processing_time_ms) / n
        )
        
        # Record detailed metrics
        metric_record = {
            "timestamp": timestamp,
            "message_id": prediction.message_id,
            "importance_level": prediction.importance_level,
            "confidence": prediction.confidence,
            "processing_time_ms": prediction.processing_time_ms,
            "model_version": prediction.model_version
        }
        
        if actual_feedback is not None:
            metric_record["actual_feedback"] = actual_feedback
            metric_record["prediction_error"] = abs(prediction.importance_level - actual_feedback)
        
        self.metrics_history.append(metric_record)
        
        # Check for alerts
        self._check_alerts()
        
        # Periodically save metrics
        if len(self.metrics_history) % 100 == 0:
            self._save_metrics()
    
    def get_model_performance(self, model_version: str, time_window_hours: int = 24) -> Dict[str, float]:
        """Get performance metrics for a specific model version"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        cutoff_str = cutoff_time.isoformat()
        
        # Filter metrics for this model and time window
        relevant_metrics = [
            m for m in self.metrics_history
            if m.get("model_version") == model_version and m.get("timestamp", "") > cutoff_str
        ]
        
        if not relevant_metrics:
            return {"error": "No data available"}
        
        # Calculate performance metrics
        processing_times = [m["processing_time_ms"] for m in relevant_metrics]
        confidences = [m["confidence"] for m in relevant_metrics]
        
        # Accuracy calculation (only for records with feedback)
        accuracy_records = [m for m in relevant_metrics if "actual_feedback" in m]
        accuracy = 0.0
        
        if accuracy_records:
            correct_predictions = sum(
                1 for m in accuracy_records 
                if abs(m["importance_level"] - m["actual_feedback"]) <= 1  # Allow 1 level difference
            )
            accuracy = correct_predictions / len(accuracy_records)
        
        return {
            "total_predictions": len(relevant_metrics),
            "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0,
            "p95_processing_time_ms": np.percentile(processing_times, 95) if processing_times else 0,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "accuracy": accuracy,
            "feedback_coverage": len(accuracy_records) / len(relevant_metrics) if relevant_metrics else 0
        }
    
    def _check_alerts(self):
        """Check if any alert thresholds are exceeded"""
        
        # Response time alert
        if self.current_metrics["avg_response_time"] > self.alert_thresholds["max_response_time"]:
            logger.warning(f"High response time: {self.current_metrics['avg_response_time']:.1f}ms")
        
        # Error rate alert (simplified - would need proper error tracking)
        error_rate = self.current_metrics["error_count"] / max(self.current_metrics["predictions_count"], 1)
        if error_rate > self.alert_thresholds["max_error_rate"]:
            logger.warning(f"High error rate: {error_rate:.2%}")
    
    def _load_metrics(self) -> List[Dict[str, Any]]:
        """Load metrics history from disk"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_metrics(self):
        """Save metrics history to disk"""
        # Keep only last 10,000 records to prevent unbounded growth
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-10000:]
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f)

class ABTestingFramework:
    """A/B testing framework for model evaluation"""
    
    def __init__(self, test_config_file: str = "data/ab_tests/test_config.json"):
        self.config_file = Path(test_config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.test_config = self._load_test_config()
        self.test_results: Dict[str, List[Dict[str, Any]]] = {}
        
    def start_ab_test(self, test_name: str, champion_model: str, challenger_model: str,
                     traffic_split: float = 0.1, duration_hours: int = 168):  # 1 week default
        """Start a new A/B test"""
        
        test_config = {
            "test_name": test_name,
            "champion_model": champion_model,
            "challenger_model": challenger_model,
            "traffic_split": traffic_split,  # Percentage going to challenger
            "start_time": datetime.now().isoformat(),
            "duration_hours": duration_hours,
            "status": "active",
            "results": {
                "champion": {"predictions": 0, "avg_confidence": 0.0, "feedback_count": 0, "accuracy": 0.0},
                "challenger": {"predictions": 0, "avg_confidence": 0.0, "feedback_count": 0, "accuracy": 0.0}
            }
        }
        
        self.test_config[test_name] = test_config
        self._save_test_config()
        
        logger.info(f"Started A/B test '{test_name}' with {traffic_split:.1%} traffic to challenger")
    
    def route_prediction_request(self, test_name: str, user_id: str) -> str:
        """Determine which model to use for a prediction request"""
        
        if test_name not in self.test_config:
            return "champion"  # Default to champion if no test
        
        test = self.test_config[test_name]
        
        if test["status"] != "active":
            return "champion"
        
        # Check if test has expired
        start_time = datetime.fromisoformat(test["start_time"])
        if datetime.now() > start_time + timedelta(hours=test["duration_hours"]):
            self.stop_ab_test(test_name)
            return "champion"
        
        # Use hash-based routing for consistent user experience
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        hash_value = int(user_hash[:8], 16) / (2**32)  # Convert to 0-1 range
        
        if hash_value < test["traffic_split"]:
            return "challenger"
        else:
            return "champion"
    
    def record_ab_result(self, test_name: str, model_type: str, prediction: PredictionResult,
                        actual_feedback: Optional[int] = None):
        """Record A/B test result"""
        
        if test_name not in self.test_config:
            return
        
        test = self.test_config[test_name]
        results = test["results"][model_type]
        
        # Update prediction count
        results["predictions"] += 1
        
        # Update confidence (moving average)
        n = results["predictions"]
        results["avg_confidence"] = (
            (results["avg_confidence"] * (n - 1) + prediction.confidence) / n
        )
        
        # Update accuracy if feedback is available
        if actual_feedback is not None:
            results["feedback_count"] += 1
            
            # Simple accuracy: allow 1 level difference
            correct = abs(prediction.importance_level - actual_feedback) <= 1
            
            current_accuracy = results["accuracy"]
            feedback_count = results["feedback_count"]
            results["accuracy"] = (
                (current_accuracy * (feedback_count - 1) + int(correct)) / feedback_count
            )
        
        self._save_test_config()
    
    def get_ab_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get current A/B test results"""
        
        if test_name not in self.test_config:
            return {"error": "Test not found"}
        
        test = self.test_config[test_name]
        
        # Calculate statistical significance (simplified)
        champion_results = test["results"]["champion"]
        challenger_results = test["results"]["challenger"]
        
        # Calculate confidence interval for accuracy difference
        if (champion_results["feedback_count"] > 20 and 
            challenger_results["feedback_count"] > 20):
            
            accuracy_diff = challenger_results["accuracy"] - champion_results["accuracy"]
            
            # Simplified significance test
            is_significant = abs(accuracy_diff) > 0.05 and min(
                champion_results["feedback_count"], 
                challenger_results["feedback_count"]
            ) > 50
            
            recommendation = "promote_challenger" if (accuracy_diff > 0.02 and is_significant) else "keep_champion"
        else:
            is_significant = False
            accuracy_diff = 0.0
            recommendation = "continue_test"
        
        return {
            "test_name": test_name,
            "status": test["status"],
            "start_time": test["start_time"],
            "duration_hours": test["duration_hours"],
            "results": test["results"],
            "accuracy_difference": accuracy_diff,
            "is_significant": is_significant,
            "recommendation": recommendation
        }
    
    def stop_ab_test(self, test_name: str):
        """Stop an A/B test"""
        
        if test_name in self.test_config:
            self.test_config[test_name]["status"] = "completed"
            self.test_config[test_name]["end_time"] = datetime.now().isoformat()
            self._save_test_config()
            
            logger.info(f"Stopped A/B test '{test_name}'")
    
    def _load_test_config(self) -> Dict[str, Any]:
        """Load A/B test configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_test_config(self):
        """Save A/B test configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f, indent=2)

class MLInferenceService:
    """Main ML inference service that orchestrates all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.model_registry = ModelRegistry()
        self.performance_monitor = PerformanceMonitor()
        self.ab_framework = ABTestingFramework()
        self.personalization_engine = UserPersonalizationEngine()
        
        # Activity detection components
        self.activity_detector = GroupActivityDetector()
        self.fomo_detector = FOMODetector()
        
        # Cache for frequent operations
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("ML Inference Service initialized")
    
    def predict_message_importance(self, user_id: str, message_data: Dict[str, Any],
                                 context_messages: List[Dict[str, Any]] = None) -> PredictionResult:
        """Main prediction endpoint"""
        
        start_time = time.time()
        
        try:
            # Determine which model to use (A/B testing)
            model_routing = self.ab_framework.route_prediction_request("main_test", user_id)
            
            # Get the appropriate model
            if model_routing == "challenger":
                model_id = self.model_registry.get_active_model(ModelVersion.CHALLENGER)
            else:
                model_id = self.model_registry.get_active_model(ModelVersion.CHAMPION)
            
            if not model_id:
                # Fallback to heuristic
                return self._fallback_prediction(message_data, start_time)
            
            model = self.model_registry.get_model(model_id)
            if not model:
                return self._fallback_prediction(message_data, start_time)
            
            # Make prediction
            importance_level, confidence = model.predict_importance(message_data, context_messages)
            
            # Apply personalization
            personalized_importance, personalized_confidence = \
                self.personalization_engine.get_personalized_importance_score(
                    user_id, message_data, importance_level / 3.0, confidence
                )
            
            # Create result
            processing_time = (time.time() - start_time) * 1000
            
            result = PredictionResult(
                message_id=message_data.get('id', 0),
                importance_level=importance_level,
                confidence=personalized_confidence,
                personalized_score=personalized_importance,
                processing_time_ms=processing_time,
                model_version=model_id,
                features_used={"model_routing": model_routing},
                timestamp=time.time()
            )
            
            # Record for monitoring and A/B testing
            self.performance_monitor.record_prediction(result)
            self.ab_framework.record_ab_result("main_test", model_routing, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return self._fallback_prediction(message_data, start_time, str(e))
    
    def detect_group_activities(self, server_id: int, time_window_hours: int = 24) -> ActivityPrediction:
        """Detect group activities and FOMO moments"""
        
        start_time = time.time()
        
        try:
            # Get recent messages from database
            conn = sqlite3.connect(DB_NAME)
            query = """
                SELECT id, server_id, channel_id, content, sent_at
                FROM messages 
                WHERE server_id = ? AND sent_at > ?
                ORDER BY sent_at DESC
            """
            
            cutoff_time = time.time() - (time_window_hours * 3600)
            messages_df = pd.read_sql_query(query, conn, params=[server_id, cutoff_time])
            conn.close()
            
            # Detect activities
            activities = self.activity_detector.detect_activities(messages_df, time_window_hours)
            
            # Detect FOMO moments
            fomo_moments = self.fomo_detector.detect_fomo_moments(activities)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ActivityPrediction(
                activities=activities,
                fomo_moments=fomo_moments,
                processing_time_ms=processing_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error in activity detection: {e}")
            return ActivityPrediction([], [], 0.0, time.time())
    
    def record_user_feedback(self, user_id: str, message_id: int, feedback_type: str,
                           explicit_rating: Optional[int] = None, dwell_time: float = 0.0):
        """Record user feedback for learning"""
        
        # Record interaction for personalization
        self.personalization_engine.record_interaction(
            user_id, message_id, feedback_type, dwell_time, explicit_rating
        )
        
        # Update A/B test results if we have the prediction
        # In a full implementation, you'd look up the original prediction
        
        logger.info(f"Recorded {feedback_type} feedback from user {user_id} for message {message_id}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        
        champion_model = self.model_registry.get_active_model(ModelVersion.CHAMPION)
        challenger_model = self.model_registry.get_active_model(ModelVersion.CHALLENGER)
        
        champion_perf = self.performance_monitor.get_model_performance(champion_model) if champion_model else {}
        challenger_perf = self.performance_monitor.get_model_performance(challenger_model) if challenger_model else {}
        
        ab_results = self.ab_framework.get_ab_test_results("main_test")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "models": {
                "champion": champion_model,
                "challenger": challenger_model
            },
            "performance": {
                "champion": champion_perf,
                "challenger": challenger_perf
            },
            "ab_testing": ab_results,
            "system_metrics": self.performance_monitor.current_metrics
        }
    
    def _fallback_prediction(self, message_data: Dict[str, Any], start_time: float, 
                           error: str = None) -> PredictionResult:
        """Fallback prediction when models fail"""
        
        # Simple heuristic-based prediction
        content = str(message_data.get('content', ''))
        
        # Basic importance scoring
        importance_score = 0.5  # Default normal
        
        if any(word in content.lower() for word in ['urgent', 'important', 'asap']):
            importance_score = 0.9
        elif len(content) < 10:
            importance_score = 0.2
        elif '@' in content:  # Mentions
            importance_score = 0.7
        
        importance_level = int(importance_score * 3)  # Convert to 0-3 scale
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResult(
            message_id=message_data.get('id', 0),
            importance_level=importance_level,
            confidence=0.3,  # Low confidence for fallback
            personalized_score=importance_score,
            processing_time_ms=processing_time,
            model_version="fallback",
            features_used={"error": error} if error else {},
            timestamp=time.time()
        )