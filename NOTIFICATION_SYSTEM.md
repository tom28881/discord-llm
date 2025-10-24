# Discord Notification System

A comprehensive notification and integration system for Discord monitoring with smart routing, external tool integrations, and RESTful API access.

## Overview

This notification system transforms your Discord monitoring assistant from a passive web interface into an active notification hub that can:

- Send intelligent notifications through multiple channels (email, Telegram, desktop, Slack, Teams)
- Integrate with external tools (Calendar, Todo apps, Note-taking apps)
- Provide RESTful API access for third-party applications
- Generate RSS feeds for feed readers
- Support webhook integrations with IFTTT, Zapier, and custom services

## Architecture

The system consists of several key components:

### Core Components

1. **Notification Database** (`lib/notification_database.py`)
   - Extended SQLite schema for notification rules, channels, and queues
   - Manages notification rules, delivery logs, and API tokens

2. **Notification Channels** (`lib/notification_channels.py`)
   - Email, Telegram, Desktop, Slack, Teams, and custom webhook channels
   - Rate limiting and error handling for each channel type

3. **Notification Engine** (`lib/notification_engine.py`)
   - Smart notification processing with deduplication and batching
   - Do Not Disturb management and priority routing
   - Background processing with configurable intervals

4. **External Integrations** (`lib/external_integrations.py`)
   - Google Calendar, Todoist, Notion integrations
   - RSS feed generation and webhook services
   - IFTTT and Zapier webhook support

5. **API Server** (`lib/api_server.py`)
   - RESTful API with OAuth-style token authentication
   - Rate limiting and permission-based access control
   - Webhook endpoints for external services

6. **Configuration Manager** (`lib/notification_config.py`)
   - Centralized configuration and system initialization
   - Integration bridge for existing Discord bots

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize the System

```python
from lib.notification_config import initialize_notification_system

# Initialize with default configuration
system = initialize_notification_system()
```

### 3. Run the Setup Script

```bash
python notification_setup.py
```

This will:
- Create the database schema
- Set up default notification rules
- Start the API server on http://localhost:8000
- Create API tokens for external access

## Configuration

### Basic Configuration

The system uses `notification_config.json` for configuration:

```json
{
  "notification_engine": {
    "processing_interval_seconds": 10,
    "dedup_window_minutes": 30,
    "batching": {
      "enabled": true,
      "batch_size": 5,
      "timeout_minutes": 15,
      "batch_channels": ["email"]
    }
  },
  "api_server": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 8000
  },
  "notification_channels": {
    "email": {
      "enabled": true,
      "config": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password",
        "to_emails": ["alerts@yourcompany.com"]
      }
    }
  }
}
```

### Notification Channels

#### Email (SMTP)
```json
"email": {
  "enabled": true,
  "config": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "your-email@gmail.com",
    "password": "your-app-password",
    "to_emails": ["user1@example.com", "user2@example.com"],
    "from_email": "discord-alerts@yourcompany.com"
  }
}
```

#### Telegram Bot
```json
"telegram": {
  "enabled": true,
  "config": {
    "bot_token": "YOUR_BOT_TOKEN",
    "chat_ids": ["CHAT_ID_1", "CHAT_ID_2"]
  }
}
```

#### Slack Webhook
```json
"slack": {
  "enabled": true,
  "config": {
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    "channel": "#discord-alerts",
    "username": "Discord Monitor"
  }
}
```

#### Desktop Notifications
```json
"desktop": {
  "enabled": true,
  "config": {
    "app_name": "Discord Monitor",
    "icon_path": "/path/to/icon.png"
  }
}
```

### External Integrations

#### Google Calendar
```json
"google_calendar": {
  "enabled": true,
  "config": {
    "client_id": "your-client-id",
    "client_secret": "your-client-secret", 
    "access_token": "your-access-token",
    "refresh_token": "your-refresh-token",
    "calendar_id": "primary"
  }
}
```

#### Todoist
```json
"todoist": {
  "enabled": true,
  "config": {
    "api_token": "your-todoist-api-token",
    "project_id": null
  }
}
```

#### Notion
```json
"notion": {
  "enabled": true,
  "config": {
    "auth_token": "your-notion-integration-token",
    "database_id": "your-database-id"
  }
}
```

## Integration with Existing Discord Bot

### 1. Import the System

```python
from lib.notification_config import get_system_integrator

# Get the integrator instance
integrator = get_system_integrator()
```

### 2. Process Discord Messages

```python
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Process message for notifications
    integrator.process_discord_message(
        server_id=message.guild.id if message.guild else 0,
        channel_id=message.channel.id,
        message_id=message.id,
        content=message.content,
        author=str(message.author),
        timestamp=message.created_at,
        server_name=message.guild.name if message.guild else "DM",
        channel_name=message.channel.name if hasattr(message.channel, 'name') else "DM"
    )
```

### 3. Add Bot Commands

```python
@bot.command(name='notify')
async def create_notification(ctx, priority: int = 2, *, content):
    """Create a manual notification"""
    integrator.create_manual_notification(
        title=f"Manual notification from {ctx.author}",
        content=content,
        priority=priority,
        channels=["desktop", "email"],
        metadata={
            "server_id": ctx.guild.id if ctx.guild else 0,
            "channel_id": ctx.channel.id,
            "author": str(ctx.author)
        }
    )
    await ctx.send("✅ Notification created!")

@bot.command(name='dnd')
async def set_dnd(ctx, duration: int = 60):
    """Set Do Not Disturb mode"""
    from lib.notification_engine import get_notification_engine
    
    engine = get_notification_engine()
    engine.dnd_manager.set_global_dnd(True, duration)
    await ctx.send(f"🔕 Do Not Disturb enabled for {duration} minutes")
```

## API Usage

### Authentication

All API requests require a Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_TOKEN" \
     http://localhost:8000/health
```

### Endpoints

#### Get Recent Messages
```bash
curl -H "Authorization: Bearer TOKEN" \
     "http://localhost:8000/messages?server_id=123456&hours=24&keywords=urgent,important"
```

#### Create Notification Rule
```bash
curl -X POST -H "Authorization: Bearer TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "High Priority Keywords",
       "keywords": ["urgent", "critical", "emergency"],
       "priority": 4,
       "channels": ["email", "telegram", "desktop"],
       "conditions": {"min_length": 10}
     }' \
     http://localhost:8000/rules
```

#### Send Test Notification
```bash
curl -X POST -H "Authorization: Bearer TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "Test notification from API",
       "server_id": 123456,
       "channel_id": 789012,
       "author": "API Test"
     }' \
     http://localhost:8000/test/notification
```

#### Get System Statistics
```bash
curl -H "Authorization: Bearer TOKEN" \
     http://localhost:8000/stats
```

## Notification Rules

The system supports sophisticated notification rules with conditions:

### Rule Components

- **Name**: Descriptive name for the rule
- **Server ID**: Optional filter for specific Discord server
- **Channel ID**: Optional filter for specific channel
- **Keywords**: List of keywords to match in message content
- **Priority**: 1 (Low) to 4 (Urgent)
- **Channels**: List of notification channels to use
- **Conditions**: Additional filtering conditions

### Example Rules

#### High Priority Keywords
```json
{
  "name": "High Priority Keywords",
  "keywords": ["urgent", "critical", "emergency", "asap"],
  "priority": 4,
  "channels": ["email", "telegram", "desktop"],
  "conditions": {
    "min_length": 15,
    "require_mentions": false
  }
}
```

#### Group Purchases
```json
{
  "name": "Group Purchases",
  "keywords": ["group buy", "bulk order", "split cost"],
  "priority": 3,
  "channels": ["email", "desktop"],
  "conditions": {
    "time_range": {"start": 9, "end": 21},
    "min_length": 20
  }
}
```

#### Work Hours Only
```json
{
  "name": "Work Hours Notifications",
  "keywords": ["meeting", "deadline", "project"],
  "priority": 2,
  "channels": ["desktop"],
  "conditions": {
    "time_range": {"start": 9, "end": 17}
  }
}
```

## Smart Features

### Deduplication
Prevents duplicate notifications within a configurable time window (default: 30 minutes).

### Batching
Combines multiple low-priority notifications into digest emails to reduce spam.

### Do Not Disturb
- Global DND mode with optional duration
- Per-user DND schedules
- Priority override for urgent notifications

### Rate Limiting
- Per-channel rate limits (hourly/daily)
- Automatic backoff on channel failures
- API rate limiting with token-based quotas

## External Integrations

### Calendar Integration (Google Calendar)
Automatically creates calendar events when it detects:
- Meeting mentions with times
- Event planning messages
- Scheduled activities

### Task Management (Todoist)
Creates tasks when it detects:
- Action items and TODO mentions
- Task-related keywords
- Reminder requests

### Note Taking (Notion)
Saves important messages as Notion pages with:
- Automatic categorization
- Source attribution
- Searchable content

### IFTTT/Zapier Integration
Triggers webhooks for:
- Custom automation workflows
- Third-party service integration
- Complex business logic

### RSS Feed Generation
Creates RSS feeds for:
- Feed reader applications
- Website integration
- Content syndication

## Security Considerations

### API Security
- Token-based authentication
- Permission-based access control
- Rate limiting to prevent abuse
- Secure token storage with hashing

### Credential Management
- Environment variable configuration
- Encrypted credential storage
- Token refresh mechanisms
- Secure webhook validation

### Privacy Protection
- Message content filtering
- Sensitive data detection
- Configurable retention policies
- GDPR compliance features

## Monitoring and Maintenance

### Logging
The system provides comprehensive logging:
- Notification processing events
- Integration successes/failures
- API access logs
- Error tracking and debugging

### Statistics
Real-time statistics available via API:
- Messages processed
- Notifications sent/failed
- Channel performance
- Rule effectiveness

### Health Checks
- API health endpoint
- Integration connection tests
- Database connectivity
- Channel availability

## Troubleshooting

### Common Issues

#### Email Not Sending
- Check SMTP credentials
- Verify firewall/security settings
- Test with a simple email client
- Check rate limits

#### Telegram Bot Not Working
- Verify bot token
- Check chat ID format
- Ensure bot is added to groups
- Test with Telegram API directly

#### Desktop Notifications Not Showing
- Check OS notification permissions
- Verify notification system availability
- Test with system notification tools
- Check Do Not Disturb settings

#### API Access Denied
- Verify token format and validity
- Check token permissions
- Confirm rate limit status
- Review API logs

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger('notification_system').setLevel(logging.DEBUG)
```

View system status:
```python
from lib.notification_config import get_notification_system
system = get_notification_system()
status = system.get_system_status()
print(json.dumps(status, indent=2))
```

## Performance Optimization

### Database Optimization
- Regular cleanup of old notifications
- Index optimization for queries
- Connection pooling for high loads

### Memory Management
- Configurable buffer sizes
- Background processing optimization
- Resource cleanup and monitoring

### Network Optimization
- Connection pooling for external APIs
- Retry logic with exponential backoff
- Timeout configuration for reliability

## Future Enhancements

Planned features for future versions:
- Machine learning for smart notification prioritization
- Advanced analytics and reporting
- Mobile app integration
- Voice notification support
- Multi-language support
- Advanced security features

## Contributing

To contribute to the notification system:
1. Fork the repository
2. Create feature branches
3. Add comprehensive tests
4. Update documentation
5. Submit pull requests

## License

This notification system is part of the Discord Monitor project and follows the same licensing terms.