# Predictive Analytics Integration Guide

This guide provides step-by-step instructions to integrate the predictive analytics system into your Discord monitoring assistant.

## Overview

The predictive system adds the following capabilities:
- **Event Prediction**: Anticipate group buys, announcements, and high-activity periods
- **Importance Prediction**: Score messages for importance before reading
- **User Behavior Modeling**: Understand and predict user activity patterns
- **Pattern Discovery**: Automatically detect recurring events and trends
- **FOMO Prevention**: Intelligent filtering to prevent missing important events

## Quick Start

### 1. Install Additional Dependencies

```bash
pip install scikit-learn pandas plotly numpy
```

### 2. Initialize Predictive System

Add to your existing `streamlit_app.py`:

```python
# Add at top of file
from predictive_streamlit_integration import PredictiveStreamlitInterface
from lib.prediction_scheduler import start_prediction_system

# Add to main() function
def main():
    # Existing code...
    
    # Initialize predictive system
    if 'prediction_system' not in st.session_state:
        start_prediction_system(str(DB_NAME))
        st.session_state.prediction_system = True
    
    # Add predictive toggle in sidebar
    if st.sidebar.checkbox("🔮 Predictive Analytics", value=False):
        predictive_interface = PredictiveStreamlitInterface(str(DB_NAME))
        predictive_interface.render_predictive_dashboard()
    else:
        # Your existing interface
        display_chat()
```

### 3. Run Enhanced Application

```bash
streamlit run streamlit_app.py
```

The predictive analytics will be available via the sidebar toggle.

## Core Components

### 1. Predictive Engine (`lib/predictive_engine.py`)
Main prediction logic with algorithms for:
- Group buy pattern detection
- Announcement timing prediction
- Activity cycle analysis
- Message importance scoring
- User behavior profiling

### 2. Prediction Scheduler (`lib/prediction_scheduler.py`)
Background processing system that:
- Runs periodic model training
- Updates predictions every 30 minutes
- Caches results for performance
- Provides real-time prediction API

### 3. Enhanced Database (`lib/enhanced_database.py`)
Extended database operations for:
- Temporal pattern analysis
- Keyword trend detection
- Channel activity comparison
- Prediction result storage

### 4. Advanced Algorithms (`lib/prediction_algorithms.py`)
Specialized algorithms including:
- Time series seasonal pattern detection
- NLP-based importance scoring
- Behavioral pattern recognition
- Anomaly detection for emerging trends
- FOMO risk assessment

### 5. Streamlit Integration (`predictive_streamlit_integration.py`)
Complete dashboard interface with:
- Real-time insights display
- Interactive visualizations
- Smart alert system
- Pattern discovery tools
- System health monitoring

## Detailed Features

### Event Prediction

**Group Buy Detection:**
- Analyzes historical group buy messages
- Detects weekly and daily patterns
- Predicts next likely group buy timing
- Confidence scoring based on pattern strength

**Example Output:**
```
🔮 Group buy activity expected (avg 3.2 mentions/week recently)
Expected: Next Tuesday
Confidence: 78%
```

**Announcement Prediction:**
- Identifies announcement timing patterns
- Predicts optimal announcement windows
- Tracks seasonal announcement cycles

### Importance Prediction

**Real-time Message Scoring:**
```python
# Example usage
message_data = {
    'content': 'URGENT: Group buy closing in 2 hours!',
    'hour': 14,
    'day_of_week': 2,
    'channel_id': 123456
}

importance = predict_message_importance(message_data)
# Returns: {'importance_score': 0.89, 'is_important': True}
```

**Features Analyzed:**
- Content length and complexity
- Urgency keywords (urgent, deadline, limited)
- Time sensitivity indicators
- Channel importance weighting
- Social engagement signals

### User Behavior Modeling

**Activity Pattern Analysis:**
- Peak activity hours detection
- Weekly activity cycles
- Consistency scoring
- Attention prediction

**Availability Prediction:**
```
Current attention score: 72%
Recommendation: Good time for important notifications
Period: weekday_evening
```

### Pattern Discovery

**Automatic Event Detection:**
- Recurring weekly events
- Monthly cycles (restocks, releases)
- Seasonal patterns
- Topic clustering

**Trending Analysis:**
- Keyword momentum tracking
- Emerging topic detection
- Conversation lifecycle analysis

### Smart Notifications

**Intelligent Filtering:**
```
🔴 High FOMO risk - Check immediately to avoid missing out
🟡 Group buy discussions trending - Review when convenient  
📱 Currently in low attention period - consider scheduling for later
```

**Notification Types:**
- Event predictions (group buys, announcements)
- Importance alerts (critical messages)
- Pattern notifications (unusual activity)
- Behavior insights (optimal check times)

## Advanced Configuration

### Prediction Confidence Thresholds

```python
# Configure in Streamlit sidebar
confidence_threshold = 0.7  # Only show 70%+ confidence predictions
prediction_horizon = "24 hours"  # How far ahead to predict
```

### Custom Keywords

Add domain-specific keywords for your community:

```python
importance_keywords = {
    'group buy': 2.8,
    'interest check': 2.5, 
    'gb': 2.8,
    'ic': 2.3,
    'artisan': 2.0,
    'keycap': 1.8,
    # Add your community's specific terms
}
```

### Training Schedule

```python
# Automatic model retraining
schedule.every().day.at("02:00").do(retrain_models)  # Daily at 2 AM
schedule.every(30).minutes.do(update_predictions)     # Every 30 min
```

## Performance Optimization

### Database Indexing
The system creates optimized indexes for:
- Message timestamps
- Server and channel filtering
- Prediction caching

### Caching Strategy
- In-memory prediction caching (30min TTL)
- Disk-based persistence
- Background refresh during low-activity periods

### Resource Usage
- Memory: ~50-100MB additional
- CPU: Background processing during off-peak hours
- Storage: ~10-20% database size increase for predictions

## Monitoring and Debugging

### System Health Dashboard
Access via "System Status" tab:
- Database statistics
- Model performance metrics
- Cache hit rates
- Prediction accuracy

### Logging
Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('predictive_system')
```

### Common Issues

**Issue: No predictions showing**
- Check database has sufficient message history (minimum 7 days)
- Verify confidence threshold not too high
- Check server_id is correctly set

**Issue: Poor prediction accuracy**
- Allow more training time (system improves with data)
- Adjust confidence thresholds
- Verify message content quality

**Issue: High memory usage**
- Reduce prediction cache TTL
- Limit number of concurrent predictions
- Optimize database queries

## API Usage Examples

### Real-time Message Importance

```python
from lib.prediction_scheduler import get_prediction_api

api = get_prediction_api(db_path)

# Check new message importance
result = api.predict_message_importance(
    message_content="Group buy for artisan keycaps closing soon!",
    channel_id=123456,
    timestamp=datetime.now()
)

print(f"Importance: {result['importance_score']:.0%}")
print(f"Recommendation: {result['recommendation']}")
```

### Conversation Urgency Analysis

```python
# Analyze ongoing conversation
conversation_messages = [
    {'content': 'Anyone interested in the new group buy?', 'timestamp': '2024-01-01T10:00:00'},
    {'content': 'YES! Been waiting for this!', 'timestamp': '2024-01-01T10:01:00'},
    {'content': 'Only 50 spots available', 'timestamp': '2024-01-01T10:02:00'}
]

urgency = api.analyze_conversation_urgency(conversation_messages)
print(f"Urgency Level: {urgency['urgency']}")
print(f"Recommendation: {urgency['recommendation']}")
```

### Batch Predictions

```python
# Get all current insights for server
insights = api.get_current_insights(server_id=123456)

for insight in insights['insights']:
    print(f"{insight['type']}: {insight['message']} ({insight['confidence']})")
```

## Integration with Existing Workflows

### Discord Bot Integration
```python
# Add to Discord bot message handler
@bot.event
async def on_message(message):
    # Existing message processing...
    
    # Check importance
    importance = predict_message_importance({
        'content': message.content,
        'channel_id': message.channel.id,
        'timestamp': message.created_at
    })
    
    if importance['is_important']:
        # Send notification or highlight message
        await send_important_message_alert(message, importance)
```

### Webhook Integration
```python
# REST API endpoint for external integrations
@app.route('/api/predictions/<int:server_id>')
def get_predictions(server_id):
    predictions = get_predictions_for_dashboard(db_path, server_id)
    return jsonify(predictions)
```

## Customization Options

### Community-Specific Tuning

1. **Keyword Customization**: Add your community's specific terms
2. **Timing Patterns**: Adjust for your timezone and activity patterns  
3. **Channel Weighting**: Configure channel importance hierarchy
4. **Alert Thresholds**: Fine-tune notification sensitivity

### Advanced Features

1. **Multi-Server Analytics**: Compare patterns across servers
2. **User-Specific Profiles**: Personalized prediction models
3. **Integration APIs**: Connect with external tools
4. **Custom Dashboards**: Build domain-specific interfaces

## Support and Troubleshooting

For issues or questions:
1. Check system logs for errors
2. Verify database connectivity
3. Ensure sufficient training data
4. Review configuration settings
5. Monitor resource usage

The predictive system is designed to improve over time as it learns from your Discord community's unique patterns and behaviors.