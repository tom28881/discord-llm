"""
Enhanced message loading with real-time intelligence processing
Drop-in replacement for load_messages.py with performance optimizations
"""
import os
import sys
import logging
import argparse
import time
from datetime import datetime
from typing import List, Tuple

# Import optimized database functions
from lib.database_integration import (
    init_db, save_messages, get_last_message_id, save_server, save_channel,
    run_background_processing, detect_group_activities
)
from lib.discord_client import Discord
from lib.config_manager import load_config, load_forbidden_channels, add_forbidden_channel
from sqlite_performance_config import SQLitePerformanceOptimizer, setup_wal_checkpoint_optimization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('discord_bot_optimized')

class OptimizedMessageProcessor:
    """Handles message processing with real-time intelligence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.optimizer = SQLitePerformanceOptimizer(db_path)
        self.batch_size = 100  # Process messages in batches
        self.importance_threshold = 0.7  # Log high importance messages
        
    def setup_database_optimization(self):
        """Setup database for high-performance processing"""
        logger.info("Setting up database optimizations...")
        
        # Setup WAL mode for concurrent processing
        setup_wal_checkpoint_optimization(self.db_path)
        
        # Create performance indexes
        with self.optimizer.get_connection() as conn:
            self.optimizer.create_performance_indexes(conn)
            
        logger.info("Database optimization setup completed")
    
    def process_messages_batch(self, messages_batch: List[Tuple], server_id: int, channel_name: str):
        """Process a batch of messages with intelligence analysis"""
        
        if not messages_batch:
            return
            
        # Save messages with intelligence processing
        save_messages(messages_batch)
        
        # Count high-importance messages in this batch
        high_importance_count = 0
        
        # Note: In a real implementation, you'd get the importance scores
        # from the save_messages_with_intelligence function
        logger.info(f"Processed batch of {len(messages_batch)} messages from #{channel_name}")
        
        if high_importance_count > 0:
            logger.info(f"⚠️  Found {high_importance_count} high-importance messages in #{channel_name}")

def fetch_and_store_messages_optimized(client: Discord, forbidden_channels: set, config: dict, 
                                     server_id: str, server_name: str, processor: OptimizedMessageProcessor):
    """Enhanced message fetching with real-time processing"""
    
    logger.info(f"Starting optimized message fetch for server: {server_name}")
    start_time = time.time()
    
    channel_info = client.get_channel_ids()
    total_new_messages = 0
    
    # Save server information
    save_server(server_id, server_name)
    
    for channel_id, channel_name in channel_info:
        if channel_id in forbidden_channels:
            logger.info(f"Skipping forbidden channel: #{channel_name} (ID: {channel_id})")
            continue
            
        if not channel_name:  # Skip channels with no name
            continue
            
        # Save channel information
        save_channel(channel_id, server_id, channel_name)
        
        last_message_id = get_last_message_id(server_id, channel_id)
        
        try:
            logger.info(f"Fetching messages from channel: #{channel_name} (ID: {channel_id})")
            new_messages = client.fetch_messages(channel_id, last_message_id, 5000)
            
            if new_messages:
                # Prepare messages for batch processing
                messages_to_save = [
                    (server_id, channel_id, message_id, content, sent_at)
                    for message_id, content, sent_at in new_messages
                ]
                
                # Process in batches for better performance
                batch_size = processor.batch_size
                for i in range(0, len(messages_to_save), batch_size):
                    batch = messages_to_save[i:i + batch_size]
                    processor.process_messages_batch(batch, server_id, channel_name)
                    
                    # Small delay to prevent overwhelming the database
                    if len(messages_to_save) > batch_size:
                        time.sleep(0.1)
                
                total_new_messages += len(new_messages)
                logger.info(f"✅ Stored {len(new_messages)} new messages from #{channel_name}")
            else:
                logger.info(f"No new messages in #{channel_name}")
                
        except Exception as e:
            if "403" in str(e):
                logger.warning(f"Access forbidden to channel #{channel_name} (ID: {channel_id}). Adding to forbidden list.")
                add_forbidden_channel(channel_id)
            else:
                logger.error(f"Error fetching messages from #{channel_name}: {e}")
                
        # Rate limiting - 1 second between channels
        time.sleep(1)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(f"✅ Completed message fetch for {server_name}:")
    logger.info(f"   • Total new messages: {total_new_messages}")
    logger.info(f"   • Processing time: {processing_time:.2f} seconds")
    logger.info(f"   • Messages per second: {total_new_messages / max(processing_time, 1):.1f}")
    
    # Run pattern detection for new messages
    if total_new_messages > 0:
        logger.info("🔍 Running pattern detection on new messages...")
        try:
            patterns = detect_group_activities(int(server_id), hours=1)  # Check last hour
            if patterns:
                high_confidence_patterns = [p for p in patterns if p['confidence'] > 0.7]
                if high_confidence_patterns:
                    logger.info(f"🎯 Detected {len(high_confidence_patterns)} high-confidence patterns")
                    for pattern in high_confidence_patterns:
                        logger.info(f"   • {pattern['type']}: confidence={pattern['confidence']:.2f}")
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")

def run_post_processing_analysis(server_id: str):
    """Run comprehensive analysis after message loading"""
    
    logger.info("🧠 Running post-processing intelligence analysis...")
    
    try:
        # Run background processing for pattern detection
        pattern_count = run_background_processing(int(server_id))
        logger.info(f"✅ Background processing completed: {pattern_count} patterns detected")
        
        # Get recent high-importance messages
        from lib.database_integration import get_high_priority_alerts
        alerts = get_high_priority_alerts(hours=24)
        
        if alerts:
            logger.info(f"⚠️  Found {len(alerts)} high-priority messages in last 24 hours:")
            for alert in alerts[:3]:  # Show top 3
                timestamp = datetime.fromtimestamp(alert['sent_at']).strftime('%H:%M')
                logger.info(f"   • [{timestamp}] {alert['server_name']} > #{alert['channel_name']}: {alert['content'][:60]}...")
        
    except Exception as e:
        logger.error(f"Error in post-processing analysis: {e}")

def main():
    parser = argparse.ArgumentParser(description="Optimized Discord Message Importer with Intelligence")
    parser.add_argument("--server_id", type=str, help="Specific Discord server ID to import")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip post-processing analysis")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for message processing")
    args = parser.parse_args()

    # Initialize optimized database
    init_db()
    
    # Setup message processor
    db_path = "data/db.sqlite"
    processor = OptimizedMessageProcessor(db_path)
    processor.batch_size = args.batch_size
    
    # Setup database optimizations
    processor.setup_database_optimization()
    
    # Load configuration
    config = load_config()
    forbidden_channels = load_forbidden_channels()
    
    if not config:
        logger.error("Failed to load configuration. Please check your config.json file.")
        return
    
    # Initialize Discord client
    discord_client = Discord(config)
    if not discord_client.connect():
        logger.error("Failed to connect to Discord. Please check your token and configuration.")
        return
    
    try:
        if args.server_id:
            # Process specific server
            server_name = discord_client.get_server_name(args.server_id)
            if server_name:
                logger.info(f"Processing server: {server_name} (ID: {args.server_id})")
                fetch_and_store_messages_optimized(
                    discord_client, forbidden_channels, config, 
                    args.server_id, server_name, processor
                )
                
                # Run post-processing analysis unless skipped
                if not args.skip_analysis:
                    run_post_processing_analysis(args.server_id)
            else:
                logger.error(f"Could not find server with ID: {args.server_id}")
        else:
            # Process all servers
            servers = discord_client.get_servers()
            if not servers:
                logger.error("No servers found. Make sure the bot is added to servers or check permissions.")
                return
                
            logger.info(f"Found {len(servers)} servers to process")
            
            for server_id, server_name in servers:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing server: {server_name} (ID: {server_id})")
                logger.info(f"{'='*60}")
                
                fetch_and_store_messages_optimized(
                    discord_client, forbidden_channels, config,
                    str(server_id), server_name, processor
                )
                
                # Run post-processing for each server unless skipped
                if not args.skip_analysis:
                    run_post_processing_analysis(str(server_id))
                
                # Rate limiting between servers - 10 seconds
                if len(servers) > 1:
                    logger.info("Waiting 10 seconds before processing next server...")
                    time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Close Discord client
        discord_client.close()
        logger.info("Discord client closed")

if __name__ == "__main__":
    main()