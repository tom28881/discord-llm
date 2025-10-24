#!/usr/bin/env python3
"""
Test the new ML features with existing messages
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from lib.purchase_predictor import PurchasePredictor
from lib.deadline_tracker import DeadlineTracker
from lib.sentiment_analyzer import SentimentAnalyzer
from lib.conversation_threader import ConversationThreader
from lib.database_optimized import OptimizedDatabase
from datetime import datetime, timedelta

def test_ml_features():
    """Test all ML features"""
    print("Testing ML Features...")
    print("=" * 60)
    
    # Initialize modules
    db = OptimizedDatabase()
    purchase_predictor = PurchasePredictor()
    deadline_tracker = DeadlineTracker()
    sentiment_analyzer = SentimentAnalyzer()
    conversation_threader = ConversationThreader()
    
    # Get a server ID first - use the server with messages
    servers = db.get_servers()
    if not servers:
        print("No servers found in database")
        return
    
    # Find server with messages
    server_id = None
    for server in servers:
        # Check if server has messages
        if server['id'] == 809811760273555506:  # Specified Server has messages
            server_id = server['id']
            print(f"Using server: {server['name']} (ID: {server_id})")
            break
    
    if not server_id:
        server_id = servers[0]['id']
        print(f"Using server: {servers[0]['name']} (ID: {server_id})")
    
    # Get recent messages to analyze (expanded time range for more data)
    messages = db.get_messages_by_importance(server_id, hours=720, min_importance=0)[:100]
    print(f"Found {len(messages)} messages to analyze")
    
    if messages:
        print("\n1. Testing Purchase Prediction...")
        print("-" * 40)
        # Analyze for purchases
        prediction = purchase_predictor.predict_purchase(messages[:20])
        print(f"Purchase probability: {prediction['probability']:.2%}")
        print(f"Type: {prediction['prediction_type']}")
        if prediction['metadata']:
            print(f"Metadata: {prediction['metadata']}")
        
        print("\n2. Testing Deadline Tracking...")
        print("-" * 40)
        # Extract deadlines
        deadlines = deadline_tracker.extract_deadlines(messages[:20])
        print(f"Found {len(deadlines)} potential deadlines")
        for deadline in deadlines[:3]:
            if deadline.get('deadline_date'):
                dt = datetime.fromtimestamp(deadline['deadline_date'])
                print(f"  - {deadline.get('deadline_text', 'Deadline')}: {dt.strftime('%Y-%m-%d %H:%M')}")
        
        print("\n3. Testing Sentiment Analysis...")
        print("-" * 40)
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze_sentiment(messages[:20])
        print(f"Overall sentiment: {sentiment['sentiment_type']}")
        print(f"Sentiment score: {sentiment['sentiment_score']:.2f}")
        print(f"Excitement level: {sentiment['excitement_level']:.2f}")
        print(f"Confidence: {sentiment['confidence']:.2f}")
        
        print("\n4. Testing Conversation Threading...")
        print("-" * 40)
        # Thread messages
        threads = conversation_threader.thread_messages(messages[:50])
        print(f"Found {len(threads)} conversation threads")
        for thread in threads[:3]:
            print(f"  - Thread type: {thread['thread_type']}, Messages: {len(thread['messages'])}")
    
    # Test getting predictions from database
    print("\n5. Testing Database Retrieval...")
    print("-" * 40)
    
    recent_predictions = purchase_predictor.get_recent_predictions(hours=720)
    print(f"Recent purchase predictions: {len(recent_predictions)}")
    
    upcoming_deadlines = deadline_tracker.get_upcoming_deadlines(hours_ahead=720)
    print(f"Upcoming deadlines: {len(upcoming_deadlines)}")
    
    excitement_peaks = sentiment_analyzer.get_excitement_peaks(hours=720)
    print(f"Excitement peaks: {len(excitement_peaks)}")
    
    active_threads = conversation_threader.get_active_threads(hours=720)
    print(f"Active threads: {len(active_threads)}")
    
    print("\n" + "=" * 60)
    print("✅ ML Features Test Complete!")

if __name__ == "__main__":
    test_ml_features()