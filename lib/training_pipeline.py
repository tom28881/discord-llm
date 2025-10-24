"""
ML Training Pipeline with Incremental Learning
Automated pipeline for model training, validation, and deployment with incremental updates.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from dataclasses import dataclass
import schedule
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.utils import resample

from .ml_system import MessageImportanceModel, ImportanceLevel
from .ml_serving import ModelRegistry, PerformanceMonitor, ModelVersion
from .personalization import UserPersonalizationEngine
from .database import DB_NAME

logger = logging.getLogger('training_pipeline')

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Data collection
    min_training_samples: int = 100
    max_training_samples: int = 10000
    data_refresh_hours: int = 168  # 1 week
    
    # Model training
    validation_split: float = 0.2
    min_accuracy_threshold: float = 0.65
    improvement_threshold: float = 0.02  # 2% improvement to deploy
    
    # Incremental learning
    batch_size: int = 50
    incremental_threshold: int = 20  # Min new samples for incremental update
    
    # Scheduling
    full_retrain_interval_hours: int = 168  # Weekly full retrain
    incremental_interval_hours: int = 24  # Daily incremental updates
    
    # Model promotion
    validation_period_hours: int = 72  # 3 days validation before promotion
    min_feedback_samples: int = 50

@dataclass
class TrainingJob:
    """Represents a training job"""
    job_id: str
    job_type: str  # 'full_retrain', 'incremental', 'validation'
    status: str  # 'pending', 'running', 'completed', 'failed'
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class DataCollector:
    """Collects and prepares training data from various sources"""
    
    def __init__(self):
        self.db_path = DB_NAME
    
    def collect_training_data(self, config: TrainingConfig) -> Tuple[pd.DataFrame, Dict[int, int]]:
        """Collect messages and feedback for training"""
        
        logger.info("Collecting training data from database")
        
        # Get messages from the last data_refresh_hours
        cutoff_time = time.time() - (config.data_refresh_hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        
        # Get messages with content
        messages_query = """
            SELECT m.id, m.server_id, m.channel_id, m.content, m.sent_at,
                   s.name as server_name, c.name as channel_name
            FROM messages m
            JOIN servers s ON m.server_id = s.id
            JOIN channels c ON m.channel_id = c.id
            WHERE m.sent_at > ? AND m.content IS NOT NULL AND length(m.content) > 5
            ORDER BY m.sent_at DESC
            LIMIT ?
        """
        
        messages_df = pd.read_sql_query(
            messages_query, 
            conn, 
            params=[cutoff_time, config.max_training_samples]
        )
        
        conn.close()
        
        logger.info(f"Collected {len(messages_df)} messages for training")
        
        # Load user feedback
        user_feedback = self._load_user_feedback()
        
        return messages_df, user_feedback
    
    def collect_incremental_data(self, since_timestamp: float) -> Tuple[pd.DataFrame, Dict[int, int]]:
        """Collect new data since last training"""
        
        conn = sqlite3.connect(self.db_path)
        
        incremental_query = """
            SELECT m.id, m.server_id, m.channel_id, m.content, m.sent_at,
                   s.name as server_name, c.name as channel_name
            FROM messages m
            JOIN servers s ON m.server_id = s.id
            JOIN channels c ON m.channel_id = c.id
            WHERE m.sent_at > ? AND m.content IS NOT NULL AND length(m.content) > 5
            ORDER BY m.sent_at DESC
        """
        
        messages_df = pd.read_sql_query(incremental_query, conn, params=[since_timestamp])
        conn.close()
        
        user_feedback = self._load_user_feedback(since_timestamp)
        
        logger.info(f"Collected {len(messages_df)} incremental messages")
        
        return messages_df, user_feedback
    
    def _load_user_feedback(self, since_timestamp: float = 0) -> Dict[int, int]:
        """Load user feedback from personalization engine"""
        
        # This would integrate with the personalization engine
        # For now, return empty dict as feedback collection depends on UI integration
        feedback = {}
        
        try:
            # In production, this would query the personalization system
            # for explicit ratings and implicit feedback
            personalization_engine = UserPersonalizationEngine()
            
            # Aggregate implicit feedback into importance scores
            for user_id, interactions in personalization_engine.interactions.items():
                for interaction in interactions:
                    if interaction.timestamp > since_timestamp:
                        if interaction.explicit_rating is not None:
                            feedback[interaction.message_id] = interaction.explicit_rating
                        elif interaction.interaction_type in ['react', 'reply', 'flag_important']:
                            feedback[interaction.message_id] = 4  # High importance
                        elif interaction.interaction_type == 'ignore':
                            feedback[interaction.message_id] = 1  # Low importance
                        else:
                            feedback[interaction.message_id] = 2  # Medium importance
            
        except Exception as e:
            logger.warning(f"Error loading user feedback: {e}")
        
        return feedback
    
    def prepare_training_features(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare additional features for training"""
        
        # Add temporal features
        messages_df['hour'] = pd.to_datetime(messages_df['sent_at'], unit='s').dt.hour
        messages_df['day_of_week'] = pd.to_datetime(messages_df['sent_at'], unit='s').dt.dayofweek
        messages_df['weekend'] = messages_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add text features
        messages_df['text_length'] = messages_df['content'].str.len()
        messages_df['word_count'] = messages_df['content'].str.split().str.len()
        messages_df['has_mention'] = messages_df['content'].str.contains('@').astype(int)
        messages_df['has_link'] = messages_df['content'].str.contains('http').astype(int)
        messages_df['exclamation_count'] = messages_df['content'].str.count('!')
        messages_df['question_count'] = messages_df['content'].str.count('\\?')
        
        # Add channel/server features
        messages_df['channel_activity'] = messages_df.groupby('channel_id')['id'].transform('count')
        messages_df['server_activity'] = messages_df.groupby('server_id')['id'].transform('count')
        
        return messages_df

class ModelTrainer:
    """Handles model training and validation"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.data_collector = DataCollector()
    
    def train_full_model(self, config: TrainingConfig) -> Tuple[MessageImportanceModel, Dict[str, Any]]:
        """Train a new model from scratch"""
        
        logger.info("Starting full model training")
        
        # Collect training data
        messages_df, user_feedback = self.data_collector.collect_training_data(config)
        
        if len(messages_df) < config.min_training_samples:
            raise ValueError(f"Insufficient training data: {len(messages_df)} < {config.min_training_samples}")
        
        # Prepare features
        messages_df = self.data_collector.prepare_training_features(messages_df)
        
        # Initialize and train model
        model = MessageImportanceModel()
        model.train(messages_df, user_feedback)
        
        # Validate model
        validation_metrics = self._validate_model(model, messages_df, user_feedback, config)
        
        logger.info(f"Model training completed with accuracy: {validation_metrics['accuracy']:.3f}")
        
        return model, validation_metrics
    
    def train_incremental_model(self, base_model: MessageImportanceModel, 
                              since_timestamp: float, config: TrainingConfig) -> Tuple[MessageImportanceModel, Dict[str, Any]]:
        """Update model incrementally with new data"""
        
        logger.info("Starting incremental model training")
        
        # Collect incremental data
        new_messages_df, new_feedback = self.data_collector.collect_incremental_data(since_timestamp)
        
        if len(new_messages_df) < config.incremental_threshold:
            raise ValueError(f"Insufficient new data for incremental training: {len(new_messages_df)}")
        
        # Prepare features
        new_messages_df = self.data_collector.prepare_training_features(new_messages_df)
        
        # Incremental training
        base_model.partial_fit(new_messages_df, new_feedback)
        
        # Quick validation on new data
        validation_metrics = self._validate_incremental(base_model, new_messages_df, new_feedback)
        
        logger.info(f"Incremental training completed with {len(new_messages_df)} new samples")
        
        return base_model, validation_metrics
    
    def _validate_model(self, model: MessageImportanceModel, messages_df: pd.DataFrame, 
                       user_feedback: Dict[int, int], config: TrainingConfig) -> Dict[str, Any]:
        """Comprehensive model validation"""
        
        # Prepare validation data
        X, y = model.prepare_training_data(messages_df, user_feedback)
        
        if len(X) < 20:  # Need minimum data for validation
            return {"accuracy": 0.0, "error": "Insufficient validation data"}
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.validation_split, random_state=42, stratify=y
        )
        
        # Re-train on training split
        model_copy = MessageImportanceModel()
        
        # Create temporary DataFrame for training split
        train_indices = train_test_split(
            range(len(messages_df)), test_size=config.validation_split, random_state=42
        )[0]
        
        train_df = messages_df.iloc[train_indices]
        train_feedback = {k: v for k, v in user_feedback.items() if k in train_df['id'].values}
        
        model_copy.train(train_df, train_feedback)
        
        # Predict on validation set
        val_predictions = []
        val_confidences = []
        
        val_indices = train_test_split(
            range(len(messages_df)), test_size=config.validation_split, random_state=42
        )[1]
        
        val_df = messages_df.iloc[val_indices]
        
        for _, row in val_df.iterrows():
            message_data = row.to_dict()
            pred, conf = model_copy.predict_importance(message_data)
            val_predictions.append(pred)
            val_confidences.append(conf)
        
        val_predictions = np.array(val_predictions)
        val_actual = np.array([user_feedback.get(msg_id, 2) for msg_id in val_df['id']])  # Default to medium
        
        # Calculate metrics
        accuracy = np.mean(np.abs(val_predictions - val_actual) <= 1)  # Allow 1 level difference
        mse = mean_squared_error(val_actual, val_predictions)
        avg_confidence = np.mean(val_confidences)
        
        # Class distribution
        class_distribution = {
            str(i): np.sum(val_actual == i) for i in range(4)
        }
        
        return {
            "accuracy": float(accuracy),
            "mse": float(mse),
            "avg_confidence": float(avg_confidence),
            "validation_samples": len(val_actual),
            "class_distribution": class_distribution,
            "feature_count": X.shape[1] if len(X.shape) > 1 else 0
        }
    
    def _validate_incremental(self, model: MessageImportanceModel, 
                            messages_df: pd.DataFrame, user_feedback: Dict[int, int]) -> Dict[str, Any]:
        """Quick validation for incremental updates"""
        
        predictions = []
        actual = []
        confidences = []
        
        for _, row in messages_df.iterrows():
            if row['id'] in user_feedback:
                message_data = row.to_dict()
                pred, conf = model.predict_importance(message_data)
                predictions.append(pred)
                confidences.append(conf)
                actual.append(user_feedback[row['id']])
        
        if len(predictions) < 5:
            return {"accuracy": 0.0, "error": "Insufficient feedback for validation"}
        
        predictions = np.array(predictions)
        actual = np.array(actual)
        
        accuracy = np.mean(np.abs(predictions - actual) <= 1)
        
        return {
            "accuracy": float(accuracy),
            "avg_confidence": float(np.mean(confidences)),
            "validation_samples": len(predictions)
        }

class TrainingScheduler:
    """Schedules and orchestrates training jobs"""
    
    def __init__(self, config: TrainingConfig, model_registry: ModelRegistry, 
                 performance_monitor: PerformanceMonitor):
        self.config = config
        self.model_registry = model_registry
        self.performance_monitor = performance_monitor
        self.trainer = ModelTrainer(model_registry)
        
        # Job management
        self.job_queue: List[TrainingJob] = []
        self.job_history: List[TrainingJob] = []
        self.current_job: Optional[TrainingJob] = None
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=1)  # Sequential training
        self.scheduler_thread = None
        self.is_running = False
        
        # State tracking
        self.last_full_training = 0.0
        self.last_incremental_training = 0.0
        
    def start(self):
        """Start the training scheduler"""
        
        self.is_running = True
        
        # Schedule periodic training
        schedule.every(self.config.full_retrain_interval_hours).hours.do(
            self._schedule_full_retrain
        )
        
        schedule.every(self.config.incremental_interval_hours).hours.do(
            self._schedule_incremental_training
        )
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Training scheduler started")
    
    def stop(self):
        """Stop the training scheduler"""
        
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Training scheduler stopped")
    
    def schedule_training_job(self, job_type: str, priority: bool = False) -> str:
        """Schedule a training job"""
        
        job_id = f"{job_type}_{int(time.time())}"
        
        job = TrainingJob(
            job_id=job_id,
            job_type=job_type,
            status="pending",
            created_at=time.time(),
            config=self.config.__dict__
        )
        
        if priority:
            self.job_queue.insert(0, job)  # Add to front
        else:
            self.job_queue.append(job)
        
        logger.info(f"Scheduled {job_type} training job: {job_id}")
        
        return job_id
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        
        while self.is_running:
            try:
                # Run scheduled jobs
                schedule.run_pending()
                
                # Process job queue
                if self.job_queue and not self.current_job:
                    self._process_next_job()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _process_next_job(self):
        """Process the next job in the queue"""
        
        if not self.job_queue:
            return
        
        job = self.job_queue.pop(0)
        self.current_job = job
        
        # Submit job to thread pool
        future = self.executor.submit(self._execute_training_job, job)
        
        # Move to history when complete (async)
        def job_complete(fut):
            self.current_job = None
            self.job_history.append(job)
            # Keep only last 100 jobs in history
            if len(self.job_history) > 100:
                self.job_history = self.job_history[-100:]
        
        future.add_done_callback(job_complete)
    
    def _execute_training_job(self, job: TrainingJob):
        """Execute a training job"""
        
        job.status = "running"
        job.started_at = time.time()
        
        try:
            if job.job_type == "full_retrain":
                self._execute_full_retrain(job)
            elif job.job_type == "incremental":
                self._execute_incremental_training(job)
            elif job.job_type == "validation":
                self._execute_validation(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            job.status = "completed"
            job.completed_at = time.time()
            
            logger.info(f"Training job {job.job_id} completed successfully")
            
        except Exception as e:
            job.status = "failed"
            job.completed_at = time.time()
            job.error_message = str(e)
            
            logger.error(f"Training job {job.job_id} failed: {e}")
            logger.error(traceback.format_exc())
    
    def _execute_full_retrain(self, job: TrainingJob):
        """Execute full model retraining"""
        
        logger.info("Executing full model retraining")
        
        # Train new model
        model, metrics = self.trainer.train_full_model(self.config)
        
        # Check if model meets quality threshold
        if metrics.get("accuracy", 0) < self.config.min_accuracy_threshold:
            raise ValueError(f"Model quality too low: {metrics.get('accuracy', 0):.3f} < {self.config.min_accuracy_threshold}")
        
        # Register as challenger model
        model_id = self.model_registry.register_model(
            model, 
            "challenger", 
            {
                "training_type": "full_retrain",
                "training_samples": metrics.get("validation_samples", 0),
                "accuracy": metrics.get("accuracy", 0),
                "training_job_id": job.job_id
            }
        )
        
        # Check if significantly better than champion
        champion_id = self.model_registry.get_active_model(ModelVersion.CHAMPION)
        
        if champion_id:
            champion_perf = self.performance_monitor.get_model_performance(champion_id)
            champion_accuracy = champion_perf.get("accuracy", 0)
            
            improvement = metrics.get("accuracy", 0) - champion_accuracy
            
            if improvement > self.config.improvement_threshold:
                # Promote to challenger for A/B testing
                self.model_registry.promote_model(model_id, ModelVersion.CHALLENGER)
                logger.info(f"New model shows {improvement:.3f} improvement, promoted to challenger")
            else:
                logger.info(f"New model improvement {improvement:.3f} below threshold {self.config.improvement_threshold}")
        else:
            # No champion exists, promote directly
            self.model_registry.promote_model(model_id, ModelVersion.CHAMPION)
            logger.info("No existing champion, promoted new model directly")
        
        job.metrics = metrics
        self.last_full_training = time.time()
    
    def _execute_incremental_training(self, job: TrainingJob):
        """Execute incremental model training"""
        
        logger.info("Executing incremental model training")
        
        # Get current champion model
        champion_id = self.model_registry.get_active_model(ModelVersion.CHAMPION)
        
        if not champion_id:
            logger.info("No champion model found, scheduling full retrain instead")
            self.schedule_training_job("full_retrain", priority=True)
            return
        
        champion_model = self.model_registry.get_model(champion_id)
        if not champion_model:
            logger.error("Could not load champion model")
            return
        
        # Perform incremental training
        updated_model, metrics = self.trainer.train_incremental_model(
            champion_model, 
            self.last_incremental_training, 
            self.config
        )
        
        # Register updated model
        model_id = self.model_registry.register_model(
            updated_model,
            "incremental",
            {
                "training_type": "incremental",
                "base_model_id": champion_id,
                "new_samples": metrics.get("validation_samples", 0),
                "accuracy": metrics.get("accuracy", 0),
                "training_job_id": job.job_id
            }
        )
        
        # If incremental training shows good results, consider promoting
        if metrics.get("accuracy", 0) > self.config.min_accuracy_threshold:
            self.model_registry.promote_model(model_id, ModelVersion.CHAMPION)
            logger.info("Incremental model promoted to champion")
        
        job.metrics = metrics
        self.last_incremental_training = time.time()
    
    def _execute_validation(self, job: TrainingJob):
        """Execute model validation job"""
        
        logger.info("Executing model validation")
        
        # Get models to validate
        champion_id = self.model_registry.get_active_model(ModelVersion.CHAMPION)
        challenger_id = self.model_registry.get_active_model(ModelVersion.CHALLENGER)
        
        validation_results = {}
        
        if champion_id:
            champion_perf = self.performance_monitor.get_model_performance(champion_id, 24)
            validation_results["champion"] = champion_perf
        
        if challenger_id:
            challenger_perf = self.performance_monitor.get_model_performance(challenger_id, 24)
            validation_results["challenger"] = challenger_perf
            
            # Check if challenger should be promoted
            if (validation_results.get("challenger", {}).get("accuracy", 0) > 
                validation_results.get("champion", {}).get("accuracy", 0) + self.config.improvement_threshold):
                
                # Promote challenger to champion
                self.model_registry.promote_model(challenger_id, ModelVersion.CHAMPION)
                logger.info("Challenger promoted to champion based on validation")
        
        job.metrics = validation_results
    
    def _schedule_full_retrain(self):
        """Schedule full retraining"""
        self.schedule_training_job("full_retrain")
    
    def _schedule_incremental_training(self):
        """Schedule incremental training"""
        self.schedule_training_job("incremental")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training pipeline status"""
        
        return {
            "is_running": self.is_running,
            "current_job": asdict(self.current_job) if self.current_job else None,
            "queue_length": len(self.job_queue),
            "last_full_training": datetime.fromtimestamp(self.last_full_training).isoformat() if self.last_full_training else None,
            "last_incremental_training": datetime.fromtimestamp(self.last_incremental_training).isoformat() if self.last_incremental_training else None,
            "recent_jobs": [asdict(job) for job in self.job_history[-5:]]  # Last 5 jobs
        }