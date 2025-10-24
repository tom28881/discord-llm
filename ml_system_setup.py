#!/usr/bin/env python3
"""
ML System Setup and Initialization Script
Sets up the production ML system for Discord message intelligence.
"""

import os
import sys
import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add the lib directory to the path
sys.path.append(str(Path(__file__).parent / 'lib'))

from ml_system import MessageImportanceModel, FeatureExtractor
from ml_serving import MLInferenceService, ModelRegistry, PerformanceMonitor
from training_pipeline import TrainingScheduler, TrainingConfig, DataCollector
from personalization import UserPersonalizationEngine
from database import DB_NAME, init_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ml_setup')

class MLSystemSetup:
    """Setup and initialize the ML system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / 'data'
        self.models_dir = self.data_dir / 'models'
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info("ML System Setup initialized")
    
    def check_requirements(self) -> bool:
        """Check if all required dependencies are installed"""
        
        required_packages = [
            'scikit-learn', 'numpy', 'pandas', 'networkx', 'schedule'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.info("Please install them with: pip install -r requirements.txt")
            return False
        
        logger.info("All required packages are installed")
        return True
    
    def check_database(self) -> bool:
        """Check if database exists and has data"""
        
        if not Path(DB_NAME).exists():
            logger.error(f"Database not found at {DB_NAME}")
            logger.info("Please run 'python load_messages.py' first to populate the database")
            return False
        
        try:
            conn = sqlite3.connect(DB_NAME)
            
            # Check for messages
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages")
            message_count = cursor.fetchone()[0]
            
            if message_count < 50:
                logger.warning(f"Only {message_count} messages in database. Recommend at least 100 for training.")
                return False
            
            logger.info(f"Database ready with {message_count} messages")
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False
    
    def bootstrap_training_data(self) -> pd.DataFrame:
        """Create bootstrap training data using heuristic labels"""
        
        logger.info("Creating bootstrap training data...")
        
        conn = sqlite3.connect(DB_NAME)
        
        # Get diverse sample of messages
        query = """
        SELECT id, server_id, channel_id, content, sent_at
        FROM messages 
        WHERE content IS NOT NULL 
        AND length(content) > 10 
        AND length(content) < 500
        ORDER BY RANDOM() 
        LIMIT 1000
        """
        
        messages_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(messages_df) < 50:
            raise ValueError("Insufficient messages for bootstrap training")
        
        logger.info(f"Bootstrap dataset created with {len(messages_df)} messages")
        return messages_df
    
    def train_initial_model(self, messages_df: pd.DataFrame) -> str:
        """Train the initial ML model"""
        
        logger.info("Training initial importance model...")
        
        # Initialize model
        model = MessageImportanceModel(model_dir=str(self.models_dir))
        
        # Create bootstrap labels using heuristics
        bootstrap_feedback = {}
        feature_extractor = FeatureExtractor()
        
        for _, row in messages_df.iterrows():
            message_data = {
                'id': row['id'],
                'content': row['content'],
                'sent_at': row['sent_at']
            }
            
            features = feature_extractor.extract_features(message_data)
            label = model._heuristic_label(features, row['content'])
            bootstrap_feedback[row['id']] = label
        
        # Train model
        model.train(messages_df, bootstrap_feedback)
        
        # Register model
        registry = ModelRegistry()
        model_id = registry.register_model(
            model, 
            "initial", 
            {
                "training_type": "bootstrap",
                "training_samples": len(messages_df),
                "bootstrap_labels": len(bootstrap_feedback)
            }
        )
        
        # Promote to champion
        registry.promote_model(model_id, "champion")
        
        logger.info(f"Initial model trained and registered as champion: {model_id}")
        return model_id
    
    def initialize_personalization(self):
        """Initialize personalization engine"""
        
        logger.info("Initializing personalization engine...")
        
        # Create personalization engine
        personalization = UserPersonalizationEngine()
        
        # Create sample user profile for demonstration
        personalization._create_user_profile("demo_user")
        personalization.save_user_data()
        
        logger.info("Personalization engine initialized")
    
    def setup_training_pipeline(self) -> bool:
        """Setup automated training pipeline"""
        
        logger.info("Setting up training pipeline...")
        
        try:
            # Create training configuration
            config = TrainingConfig(
                min_training_samples=50,
                full_retrain_interval_hours=168,  # Weekly
                incremental_interval_hours=24     # Daily
            )
            
            # Initialize components
            registry = ModelRegistry()
            monitor = PerformanceMonitor()
            
            # Create scheduler (don't start it here - that's done in production)
            scheduler = TrainingScheduler(config, registry, monitor)
            
            logger.info("Training pipeline configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up training pipeline: {e}")
            return False
    
    def validate_setup(self) -> bool:
        """Validate that the ML system is working correctly"""
        
        logger.info("Validating ML system setup...")
        
        try:
            # Test inference service
            ml_service = MLInferenceService()
            
            # Test with a sample message
            test_message = {
                'id': 1,
                'content': 'Hey everyone, important meeting tomorrow at 3 PM!',
                'server_id': 123,
                'channel_id': 456,
                'sent_at': datetime.now().timestamp()
            }
            
            # Test prediction
            result = ml_service.predict_message_importance("test_user", test_message)
            
            if result.importance_level >= 0 and result.confidence > 0:
                logger.info(f"✅ Prediction test passed: importance={result.importance_level}, confidence={result.confidence:.2f}")
            else:
                logger.error("❌ Prediction test failed")
                return False
            
            # Test activity detection
            activities = ml_service.detect_group_activities(123, 24)
            logger.info(f"✅ Activity detection test passed: {len(activities.activities)} activities detected")
            
            # Test system health
            health = ml_service.get_system_health()
            if health.get("models", {}).get("champion"):
                logger.info("✅ System health check passed")
            else:
                logger.warning("⚠️ No champion model found, using fallback")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False
    
    def run_full_setup(self):
        """Run complete ML system setup"""
        
        logger.info("🚀 Starting ML System Setup...")
        
        # Step 1: Check requirements
        if not self.check_requirements():
            logger.error("❌ Requirements check failed")
            return False
        
        # Step 2: Check database
        if not self.check_database():
            logger.error("❌ Database check failed")
            return False
        
        # Step 3: Initialize database schema extensions
        init_db()
        
        # Step 4: Bootstrap training data
        try:
            messages_df = self.bootstrap_training_data()
        except Exception as e:
            logger.error(f"❌ Bootstrap data creation failed: {e}")
            return False
        
        # Step 5: Train initial model
        try:
            model_id = self.train_initial_model(messages_df)
        except Exception as e:
            logger.error(f"❌ Initial model training failed: {e}")
            return False
        
        # Step 6: Initialize personalization
        try:
            self.initialize_personalization()
        except Exception as e:
            logger.error(f"❌ Personalization setup failed: {e}")
            return False
        
        # Step 7: Setup training pipeline
        if not self.setup_training_pipeline():
            logger.error("❌ Training pipeline setup failed")
            return False
        
        # Step 8: Validate setup
        if not self.validate_setup():
            logger.error("❌ System validation failed")
            return False
        
        logger.info("🎉 ML System Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run: streamlit run streamlit_app.py")
        logger.info("2. Navigate to 'ML Administration' page to manage the system")
        logger.info("3. Provide feedback on message importance to improve the model")
        logger.info("4. Monitor system health and performance metrics")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Setup ML System for Discord Message Intelligence")
    parser.add_argument(
        '--check-only', 
        action='store_true', 
        help='Only check requirements and database, do not setup'
    )
    parser.add_argument(
        '--validate-only', 
        action='store_true', 
        help='Only validate existing setup'
    )
    
    args = parser.parse_args()
    
    setup = MLSystemSetup()
    
    if args.check_only:
        logger.info("🔍 Checking system requirements...")
        req_ok = setup.check_requirements()
        db_ok = setup.check_database()
        
        if req_ok and db_ok:
            logger.info("✅ System ready for ML setup")
            return 0
        else:
            logger.error("❌ System not ready")
            return 1
    
    elif args.validate_only:
        logger.info("🔍 Validating existing setup...")
        if setup.validate_setup():
            logger.info("✅ System validation passed")
            return 0
        else:
            logger.error("❌ System validation failed")
            return 1
    
    else:
        # Full setup
        if setup.run_full_setup():
            return 0
        else:
            return 1

if __name__ == "__main__":
    sys.exit(main())