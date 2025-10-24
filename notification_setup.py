"""
Comprehensive setup and example usage script for the Discord notification system.
This script demonstrates how to integrate the notification system with your existing Discord bot.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the lib directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))

from lib.notification_config import initialize_notification_system, get_system_integrator, shutdown_notification_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('notification_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_notification_system():
    """Set up and configure the notification system."""
    logger.info("Setting up Discord notification system...")
    
    # Initialize the system
    system = initialize_notification_system(auto_start=True)
    
    # Get system status
    status = system.get_system_status()
    logger.info(f"System status: {status}")
    
    return system

def example_notification_integration():
    """Example of how to integrate notifications with your Discord message processing."""
    
    # Get the system integrator
    integrator = get_system_integrator()
    
    # Example: Process a Discord message that might trigger notifications
    integrator.process_discord_message(
        server_id=123456789012345678,
        channel_id=987654321098765432,
        message_id=555666777888999000,
        content="URGENT: Group buy for mechanical keyboards closes in 1 hour! Please vote on your preferred switches.",
        author="KeyboardEnthusiast#1234",
        timestamp=datetime.now(),
        server_name="Tech Community",
        channel_name="group-buys"
    )
    
    # Example: Create manual notification
    integrator.create_manual_notification(
        title="System Alert",
        content="The notification system has been successfully configured and is running.",
        priority=3,
        channels=["desktop", "email"],
        metadata={"source": "setup_script", "manual": True}
    )
    
    logger.info("Example notifications processed")

def configure_notification_channels():
    """Configure specific notification channels with your credentials."""
    
    print("\n=== Notification Channel Configuration ===")
    print("The system supports the following notification channels:")
    print("1. Email (SMTP)")
    print("2. Telegram Bot")
    print("3. Desktop Notifications")
    print("4. Slack Webhooks")
    print("5. Microsoft Teams Webhooks")
    print("6. Custom Webhooks")
    
    print("\nTo configure these channels, edit the notification_config.json file")
    print("or use the API endpoints to add channel configurations.")
    
    # Example configuration snippets
    print("\n=== Example Email Configuration ===")
    print("""
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
    """)
    
    print("\n=== Example Telegram Configuration ===")
    print("""
    "telegram": {
        "enabled": true,
        "config": {
            "bot_token": "YOUR_BOT_TOKEN",
            "chat_ids": ["YOUR_CHAT_ID"]
        }
    }
    """)

def configure_external_integrations():
    """Configure external tool integrations."""
    
    print("\n=== External Integration Configuration ===")
    print("The system supports integration with:")
    print("1. Google Calendar (auto-create events from Discord messages)")
    print("2. Todoist (create tasks from action items)")
    print("3. Notion (save important messages as pages)")
    print("4. IFTTT (trigger automation workflows)")
    print("5. Zapier (connect to hundreds of apps)")
    print("6. RSS Feed Generation")
    
    print("\n=== Example Google Calendar Configuration ===")
    print("""
    To set up Google Calendar integration:
    1. Go to Google Cloud Console
    2. Create OAuth2 credentials
    3. Authorize the calendar scope
    4. Add your tokens to the configuration:
    
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
    """)

def setup_api_access():
    """Set up API access for external applications."""
    
    print("\n=== API Access Setup ===")
    print("The notification system provides a RESTful API for external access.")
    print("API server runs on http://localhost:8000 by default.")
    
    # Import API functions
    from lib.api_server import create_api_token
    
    try:
        # Create example tokens
        admin_token = create_api_token(
            name="admin_access",
            permissions=["admin", "read", "write", "webhook"],
            rate_limit_per_hour=1000,
            rate_limit_per_day=10000
        )
        
        readonly_token = create_api_token(
            name="readonly_access",
            permissions=["read"],
            rate_limit_per_hour=100,
            rate_limit_per_day=1000
        )
        
        print(f"\nAdmin API Token: {admin_token}")
        print(f"Read-only API Token: {readonly_token}")
        
        print("\n=== API Usage Examples ===")
        print("# Get recent messages")
        print(f"curl -H 'Authorization: Bearer {readonly_token}' \\")
        print("     'http://localhost:8000/messages?server_id=123456789&hours=24'")
        
        print("\n# Create notification rule")
        print(f"curl -X POST -H 'Authorization: Bearer {admin_token}' \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"name\": \"Test Rule\", \"keywords\": [\"test\"], \"priority\": 2, \"channels\": [\"desktop\"]}' \\")
        print("     'http://localhost:8000/rules'")
        
        print("\n# Send test notification")
        print(f"curl -X POST -H 'Authorization: Bearer {admin_token}' \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"content\": \"Test notification from API\", \"server_id\": 0, \"channel_id\": 0}' \\")
        print("     'http://localhost:8000/test/notification'")
        
    except Exception as e:
        logger.error(f"Failed to create API tokens: {e}")

def integration_with_existing_bot():
    """Show how to integrate with your existing Discord bot."""
    
    print("\n=== Integration with Existing Discord Bot ===")
    print("To integrate the notification system with your existing Discord bot:")
    
    print("\n1. Import the notification system:")
    print("""
from lib.notification_config import get_system_integrator
integrator = get_system_integrator()
    """)
    
    print("\n2. Process messages in your on_message handler:")
    print("""
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
    
    # Your existing message processing logic
    await process_commands(message)
    """)
    
    print("\n3. Add notification commands to your bot:")
    print("""
@bot.command(name='notify')
async def create_notification(ctx, *, content):
    integrator.create_manual_notification(
        title=f"Manual notification from {ctx.author}",
        content=content,
        priority=2,
        channels=["desktop"],
        metadata={
            "server_id": ctx.guild.id if ctx.guild else 0,
            "channel_id": ctx.channel.id,
            "author": str(ctx.author)
        }
    )
    await ctx.send("Notification created!")
    """)

def show_advanced_features():
    """Show advanced notification features."""
    
    print("\n=== Advanced Features ===")
    
    print("\n1. Smart Notification Rules:")
    print("   - Time-based filtering (only during work hours)")
    print("   - Keyword matching with priority levels")
    print("   - Server and channel specific rules")
    print("   - Content length and mention requirements")
    
    print("\n2. Notification Batching:")
    print("   - Combine multiple low-priority notifications")
    print("   - Reduce notification spam")
    print("   - Configurable batch size and timeout")
    
    print("\n3. Do Not Disturb:")
    print("   - Global and per-channel DND modes")
    print("   - Time-based DND schedules")
    print("   - Priority override for urgent notifications")
    
    print("\n4. Deduplication:")
    print("   - Prevent duplicate notifications")
    print("   - Configurable time window")
    print("   - Smart content comparison")
    
    print("\n5. Rate Limiting:")
    print("   - Per-channel rate limits")
    print("   - Hourly and daily quotas")
    print("   - Automatic backoff on failures")
    
    print("\n6. External Integrations:")
    print("   - Automatic calendar event creation")
    print("   - Task creation in todo apps")
    print("   - Note saving in knowledge bases")
    print("   - Webhook triggers for automation")

def main():
    """Main setup function."""
    try:
        print("=" * 60)
        print("Discord Notification System Setup")
        print("=" * 60)
        
        # Set up the notification system
        system = setup_notification_system()
        
        # Show configuration options
        configure_notification_channels()
        configure_external_integrations()
        
        # Set up API access
        setup_api_access()
        
        # Show integration examples
        integration_with_existing_bot()
        
        # Show advanced features
        show_advanced_features()
        
        # Process example notifications
        example_notification_integration()
        
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nThe notification system is now running.")
        print("Check notification_system.log for detailed logs.")
        print("API server is available at http://localhost:8000")
        print("API documentation at http://localhost:8000/docs (if FastAPI is installed)")
        
        print("\nNext steps:")
        print("1. Configure your notification channels in notification_config.json")
        print("2. Set up external integrations with your API tokens")
        print("3. Create custom notification rules via the API")
        print("4. Integrate with your existing Discord bot")
        
        # Keep the system running
        print("\nPress Ctrl+C to stop the notification system...")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down notification system...")
            shutdown_notification_system()
            print("Notification system stopped.")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\nSetup failed: {e}")
        print("Check the logs for more details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())