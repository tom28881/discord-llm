"""
Built-in notification channel plugins.
These provide various ways to send notifications when important messages are detected.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os

from .base import INotificationChannelPlugin, PluginMetadata, PluginType
from ..domain.models import ImportanceLevel

logger = logging.getLogger(__name__)


class ConsoleNotificationPlugin(INotificationChannelPlugin):
    """Simple console notification plugin for development/testing."""
    
    def __init__(self):
        self._enabled = False
        self._config = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="console_notifications",
            version="1.0.0",
            description="Simple console output notifications",
            author="Discord Monitor",
            plugin_type=PluginType.NOTIFICATION_CHANNEL,
            config_schema={
                "enabled": {"type": "boolean", "default": True},
                "include_metadata": {"type": "boolean", "default": False},
                "color_output": {"type": "boolean", "default": True}
            }
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize console notifications."""
        try:
            self._config = config
            self._enabled = config.get("enabled", True)
            logger.info("Console notification plugin initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize console notifications: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown console notifications."""
        self._enabled = False
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check console notification health."""
        return {
            "status": "healthy" if self._enabled else "disabled",
            "config": self._config
        }
    
    async def send_notification(
        self, 
        message: str, 
        importance_level: ImportanceLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send notification to console."""
        if not self._enabled:
            return False
        
        try:
            # Color coding based on importance
            color_codes = {
                ImportanceLevel.CRITICAL: "\033[91m",  # Red
                ImportanceLevel.HIGH: "\033[93m",      # Yellow
                ImportanceLevel.MEDIUM: "\033[94m",    # Blue
                ImportanceLevel.LOW: "\033[92m",       # Green
                ImportanceLevel.NOISE: "\033[90m"      # Gray
            }
            reset_code = "\033[0m"
            
            use_colors = self._config.get("color_output", True)
            
            if use_colors:
                color = color_codes.get(importance_level, "")
                formatted_message = f"{color}[{importance_level.value.upper()}] {message}{reset_code}"
            else:
                formatted_message = f"[{importance_level.value.upper()}] {message}"
            
            print(formatted_message)
            
            # Include metadata if configured
            if self._config.get("include_metadata", False) and metadata:
                print(f"Metadata: {json.dumps(metadata, indent=2)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send console notification: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test console notification (always works)."""
        return self._enabled
    
    def get_supported_importance_levels(self) -> List[ImportanceLevel]:
        """Console supports all importance levels."""
        return list(ImportanceLevel)


class SlackWebhookNotificationPlugin(INotificationChannelPlugin):
    """Slack webhook notification plugin."""
    
    def __init__(self):
        self._enabled = False
        self._webhook_url = ""
        self._channel = ""
        self._username = "Discord Monitor"
        self._min_importance = ImportanceLevel.MEDIUM
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="slack_webhook_notifications",
            version="1.0.0",
            description="Send notifications to Slack via webhooks",
            author="Discord Monitor",
            plugin_type=PluginType.NOTIFICATION_CHANNEL,
            dependencies=["aiohttp"],
            config_schema={
                "webhook_url": {"type": "string", "required": True},
                "channel": {"type": "string", "default": "#general"},
                "username": {"type": "string", "default": "Discord Monitor"},
                "min_importance": {"type": "string", "default": "medium"},
                "enabled": {"type": "boolean", "default": True}
            }
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Slack webhook notifications."""
        try:
            self._webhook_url = config.get("webhook_url", "")
            if not self._webhook_url:
                logger.error("Slack webhook URL not provided")
                return False
            
            self._channel = config.get("channel", "#general")
            self._username = config.get("username", "Discord Monitor")
            
            # Parse minimum importance level
            min_importance_str = config.get("min_importance", "medium").lower()
            importance_mapping = {
                "critical": ImportanceLevel.CRITICAL,
                "high": ImportanceLevel.HIGH,
                "medium": ImportanceLevel.MEDIUM,
                "low": ImportanceLevel.LOW,
                "noise": ImportanceLevel.NOISE
            }
            self._min_importance = importance_mapping.get(min_importance_str, ImportanceLevel.MEDIUM)
            
            self._enabled = config.get("enabled", True)
            
            logger.info("Slack webhook notification plugin initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack notifications: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown Slack notifications."""
        self._enabled = False
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Slack notification health."""
        return {
            "status": "healthy" if self._enabled else "disabled",
            "webhook_configured": bool(self._webhook_url),
            "channel": self._channel,
            "min_importance": self._min_importance.value
        }
    
    async def send_notification(
        self, 
        message: str, 
        importance_level: ImportanceLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send notification to Slack."""
        if not self._enabled or not self._webhook_url:
            return False
        
        # Check if importance level meets minimum threshold
        importance_order = {
            ImportanceLevel.NOISE: 1,
            ImportanceLevel.LOW: 2,
            ImportanceLevel.MEDIUM: 3,
            ImportanceLevel.HIGH: 4,
            ImportanceLevel.CRITICAL: 5
        }
        
        if importance_order[importance_level] < importance_order[self._min_importance]:
            return True  # Skip notification but return success
        
        try:
            # Format message for Slack
            color_mapping = {
                ImportanceLevel.CRITICAL: "danger",
                ImportanceLevel.HIGH: "warning",
                ImportanceLevel.MEDIUM: "good",
                ImportanceLevel.LOW: "#36a64f",
                ImportanceLevel.NOISE: "#cccccc"
            }
            
            emoji_mapping = {
                ImportanceLevel.CRITICAL: ":red_circle:",
                ImportanceLevel.HIGH: ":warning:",
                ImportanceLevel.MEDIUM: ":large_blue_circle:",
                ImportanceLevel.LOW: ":green_heart:",
                ImportanceLevel.NOISE: ":white_circle:"
            }
            
            slack_payload = {
                "channel": self._channel,
                "username": self._username,
                "attachments": [{
                    "color": color_mapping.get(importance_level, "good"),
                    "title": f"{emoji_mapping.get(importance_level, '')} {importance_level.value.title()} Priority Message",
                    "text": message,
                    "ts": int(asyncio.get_event_loop().time())
                }]
            }
            
            # Add metadata as fields if provided
            if metadata:
                fields = []
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        fields.append({
                            "title": key.replace("_", " ").title(),
                            "value": str(value),
                            "short": True
                        })
                
                if fields:
                    slack_payload["attachments"][0]["fields"] = fields
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=slack_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Slack webhook failed with status {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Slack webhook connection."""
        if not self._enabled or not self._webhook_url:
            return False
        
        try:
            test_payload = {
                "channel": self._channel,
                "username": self._username,
                "text": "🧪 Discord Monitor connection test"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        
        except Exception as e:
            logger.error(f"Slack connection test failed: {e}")
            return False
    
    def get_supported_importance_levels(self) -> List[ImportanceLevel]:
        """Slack supports all importance levels."""
        return list(ImportanceLevel)


class EmailNotificationPlugin(INotificationChannelPlugin):
    """Email notification plugin using SMTP."""
    
    def __init__(self):
        self._enabled = False
        self._smtp_server = ""
        self._smtp_port = 587
        self._username = ""
        self._password = ""
        self._from_email = ""
        self._to_emails: List[str] = []
        self._use_tls = True
        self._min_importance = ImportanceLevel.HIGH
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="email_notifications",
            version="1.0.0",
            description="Send notifications via email using SMTP",
            author="Discord Monitor",
            plugin_type=PluginType.NOTIFICATION_CHANNEL,
            config_schema={
                "smtp_server": {"type": "string", "required": True},
                "smtp_port": {"type": "integer", "default": 587},
                "username": {"type": "string", "required": True},
                "password": {"type": "string", "required": True},
                "from_email": {"type": "string", "required": True},
                "to_emails": {"type": "array", "required": True},
                "use_tls": {"type": "boolean", "default": True},
                "min_importance": {"type": "string", "default": "high"},
                "enabled": {"type": "boolean", "default": True}
            }
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize email notifications."""
        try:
            required_fields = ["smtp_server", "username", "password", "from_email", "to_emails"]
            for field in required_fields:
                if not config.get(field):
                    logger.error(f"Email notification missing required field: {field}")
                    return False
            
            self._smtp_server = config["smtp_server"]
            self._smtp_port = config.get("smtp_port", 587)
            self._username = config["username"]
            self._password = config["password"]
            self._from_email = config["from_email"]
            self._to_emails = config["to_emails"]
            self._use_tls = config.get("use_tls", True)
            
            # Parse minimum importance level
            min_importance_str = config.get("min_importance", "high").lower()
            importance_mapping = {
                "critical": ImportanceLevel.CRITICAL,
                "high": ImportanceLevel.HIGH,
                "medium": ImportanceLevel.MEDIUM,
                "low": ImportanceLevel.LOW,
                "noise": ImportanceLevel.NOISE
            }
            self._min_importance = importance_mapping.get(min_importance_str, ImportanceLevel.HIGH)
            
            self._enabled = config.get("enabled", True)
            
            logger.info("Email notification plugin initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize email notifications: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown email notifications."""
        self._enabled = False
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check email notification health."""
        return {
            "status": "healthy" if self._enabled else "disabled",
            "smtp_server": self._smtp_server,
            "smtp_port": self._smtp_port,
            "from_email": self._from_email,
            "to_emails_count": len(self._to_emails),
            "min_importance": self._min_importance.value
        }
    
    async def send_notification(
        self, 
        message: str, 
        importance_level: ImportanceLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send email notification."""
        if not self._enabled:
            return False
        
        # Check if importance level meets minimum threshold
        importance_order = {
            ImportanceLevel.NOISE: 1,
            ImportanceLevel.LOW: 2,
            ImportanceLevel.MEDIUM: 3,
            ImportanceLevel.HIGH: 4,
            ImportanceLevel.CRITICAL: 5
        }
        
        if importance_order[importance_level] < importance_order[self._min_importance]:
            return True  # Skip notification but return success
        
        try:
            # Run email sending in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                self._send_email_sync, 
                message, 
                importance_level, 
                metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _send_email_sync(
        self, 
        message: str, 
        importance_level: ImportanceLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Synchronous email sending."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self._from_email
            msg['To'] = ', '.join(self._to_emails)
            
            # Create subject with importance indicator
            importance_indicators = {
                ImportanceLevel.CRITICAL: "🔴 CRITICAL",
                ImportanceLevel.HIGH: "🟠 HIGH",
                ImportanceLevel.MEDIUM: "🔵 MEDIUM",
                ImportanceLevel.LOW: "🟢 LOW",
                ImportanceLevel.NOISE: "⚪ NOISE"
            }
            
            subject = f"Discord Monitor Alert - {importance_indicators.get(importance_level, importance_level.value.upper())}"
            msg['Subject'] = subject
            
            # Create email body
            email_body = f"""
Discord Monitor Alert

Importance Level: {importance_level.value.upper()}

Message:
{message}

"""
            
            # Add metadata if provided
            if metadata:
                email_body += "\nAdditional Information:\n"
                for key, value in metadata.items():
                    email_body += f"• {key.replace('_', ' ').title()}: {value}\n"
            
            email_body += f"\nGenerated at: {asyncio.get_event_loop().time()}"
            
            # Attach body to email
            msg.attach(MIMEText(email_body, 'plain'))
            
            # Connect to server and send email
            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                if self._use_tls:
                    server.starttls()
                
                server.login(self._username, self._password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent successfully to {len(self._to_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test email SMTP connection."""
        if not self._enabled:
            return False
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._test_smtp_connection)
            
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False
    
    def _test_smtp_connection(self) -> bool:
        """Test SMTP connection synchronously."""
        try:
            with smtplib.SMTP(self._smtp_server, self._smtp_port, timeout=10) as server:
                if self._use_tls:
                    server.starttls()
                server.login(self._username, self._password)
            return True
        except Exception as e:
            logger.error(f"SMTP connection failed: {e}")
            return False
    
    def get_supported_importance_levels(self) -> List[ImportanceLevel]:
        """Email supports all importance levels."""
        return list(ImportanceLevel)


class WebhookNotificationPlugin(INotificationChannelPlugin):
    """Generic webhook notification plugin."""
    
    def __init__(self):
        self._enabled = False
        self._webhook_url = ""
        self._headers: Dict[str, str] = {}
        self._min_importance = ImportanceLevel.MEDIUM
        self._timeout = 10
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="webhook_notifications",
            version="1.0.0",
            description="Send notifications to generic webhook endpoints",
            author="Discord Monitor",
            plugin_type=PluginType.NOTIFICATION_CHANNEL,
            dependencies=["aiohttp"],
            config_schema={
                "webhook_url": {"type": "string", "required": True},
                "headers": {"type": "object", "default": {}},
                "min_importance": {"type": "string", "default": "medium"},
                "timeout": {"type": "integer", "default": 10},
                "enabled": {"type": "boolean", "default": True}
            }
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize webhook notifications."""
        try:
            self._webhook_url = config.get("webhook_url", "")
            if not self._webhook_url:
                logger.error("Webhook URL not provided")
                return False
            
            self._headers = config.get("headers", {})
            self._timeout = config.get("timeout", 10)
            
            # Parse minimum importance level
            min_importance_str = config.get("min_importance", "medium").lower()
            importance_mapping = {
                "critical": ImportanceLevel.CRITICAL,
                "high": ImportanceLevel.HIGH,
                "medium": ImportanceLevel.MEDIUM,
                "low": ImportanceLevel.LOW,
                "noise": ImportanceLevel.NOISE
            }
            self._min_importance = importance_mapping.get(min_importance_str, ImportanceLevel.MEDIUM)
            
            self._enabled = config.get("enabled", True)
            
            logger.info("Webhook notification plugin initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize webhook notifications: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown webhook notifications."""
        self._enabled = False
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check webhook notification health."""
        return {
            "status": "healthy" if self._enabled else "disabled",
            "webhook_url": self._webhook_url,
            "min_importance": self._min_importance.value,
            "timeout": self._timeout
        }
    
    async def send_notification(
        self, 
        message: str, 
        importance_level: ImportanceLevel,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send notification to webhook."""
        if not self._enabled or not self._webhook_url:
            return False
        
        # Check if importance level meets minimum threshold
        importance_order = {
            ImportanceLevel.NOISE: 1,
            ImportanceLevel.LOW: 2,
            ImportanceLevel.MEDIUM: 3,
            ImportanceLevel.HIGH: 4,
            ImportanceLevel.CRITICAL: 5
        }
        
        if importance_order[importance_level] < importance_order[self._min_importance]:
            return True  # Skip notification but return success
        
        try:
            # Create webhook payload
            payload = {
                "timestamp": asyncio.get_event_loop().time(),
                "importance_level": importance_level.value,
                "message": message,
                "metadata": metadata or {},
                "source": "discord_monitor"
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as response:
                    if 200 <= response.status < 300:
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False
        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test webhook connection."""
        if not self._enabled or not self._webhook_url:
            return False
        
        try:
            test_payload = {
                "test": True,
                "timestamp": asyncio.get_event_loop().time(),
                "message": "Discord Monitor connection test",
                "source": "discord_monitor"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=test_payload,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return 200 <= response.status < 300
        
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False
    
    def get_supported_importance_levels(self) -> List[ImportanceLevel]:
        """Webhook supports all importance levels."""
        return list(ImportanceLevel)