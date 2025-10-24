"""
Notification channel implementations for various delivery methods.
Supports email, Telegram, desktop notifications, Slack, Teams, and webhooks.
"""

import smtplib
import ssl
import logging
import requests
import json
import platform
import subprocess
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from abc import ABC, abstractmethod
import time
import threading
from pathlib import Path

logger = logging.getLogger('notification_channels')

class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limiter = RateLimiter(
            config.get('rate_limit_per_hour', 60),
            config.get('rate_limit_per_day', 500)
        )
    
    @abstractmethod
    def send(self, title: str, content: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Send notification. Returns (success, error_message)."""
        pass
    
    def can_send(self) -> bool:
        """Check if we can send a notification (rate limiting)."""
        return self.rate_limiter.can_send()

class RateLimiter:
    """Simple rate limiter for notifications."""
    
    def __init__(self, hourly_limit: int, daily_limit: int):
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.hourly_count = 0
        self.daily_count = 0
        self.last_hour_reset = datetime.now().hour
        self.last_day_reset = datetime.now().date()
        self._lock = threading.Lock()
    
    def can_send(self) -> bool:
        """Check if we can send based on rate limits."""
        with self._lock:
            self._reset_counters()
            return self.hourly_count < self.hourly_limit and self.daily_count < self.daily_limit
    
    def record_send(self):
        """Record that a message was sent."""
        with self._lock:
            self._reset_counters()
            self.hourly_count += 1
            self.daily_count += 1
    
    def _reset_counters(self):
        """Reset counters if time periods have passed."""
        now = datetime.now()
        if now.hour != self.last_hour_reset:
            self.hourly_count = 0
            self.last_hour_reset = now.hour
        
        if now.date() != self.last_day_reset:
            self.daily_count = 0
            self.last_day_reset = now.date()

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel using SMTP."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config['smtp_server']
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config['username']
        self.password = config['password']
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config['to_emails']  # List of recipient emails
        self.use_tls = config.get('use_tls', True)
    
    def send(self, title: str, content: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Send email notification."""
        if not self.can_send():
            return False, "Rate limit exceeded"
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = title
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            
            # Create HTML and plain text versions
            html_content = self._create_html_content(title, content, metadata)
            text_content = self._create_text_content(title, content, metadata)
            
            msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.rate_limiter.record_send()
            logger.info(f"Email sent successfully: {title}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _create_html_content(self, title: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Create HTML email content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #5865F2; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ padding: 20px; background-color: #f8f9fa; margin: 10px 0; border-radius: 5px; }}
                .metadata {{ font-size: 12px; color: #666; margin-top: 10px; }}
                .timestamp {{ font-style: italic; color: #999; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Discord Alert: {title}</h2>
            </div>
            <div class="content">
                <p>{content.replace(chr(10), '<br>')}</p>
            </div>
        """
        
        if metadata:
            html += '<div class="metadata">'
            if 'server_name' in metadata:
                html += f"<p><strong>Server:</strong> {metadata['server_name']}</p>"
            if 'channel_name' in metadata:
                html += f"<p><strong>Channel:</strong> #{metadata['channel_name']}</p>"
            if 'author' in metadata:
                html += f"<p><strong>Author:</strong> {metadata['author']}</p>"
            html += '</div>'
        
        html += f"""
            <div class="timestamp">
                <p>Sent at {timestamp}</p>
            </div>
        </body>
        </html>
        """
        return html
    
    def _create_text_content(self, title: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Create plain text email content."""
        text = f"Discord Alert: {title}\n\n{content}\n\n"
        
        if metadata:
            text += "Details:\n"
            if 'server_name' in metadata:
                text += f"Server: {metadata['server_name']}\n"
            if 'channel_name' in metadata:
                text += f"Channel: #{metadata['channel_name']}\n"
            if 'author' in metadata:
                text += f"Author: {metadata['author']}\n"
        
        text += f"\nSent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return text

class TelegramNotificationChannel(NotificationChannel):
    """Telegram bot notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bot_token = config['bot_token']
        self.chat_ids = config['chat_ids']  # List of chat IDs to send to
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send(self, title: str, content: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Send Telegram notification."""
        if not self.can_send():
            return False, "Rate limit exceeded"
        
        try:
            message = self._format_message(title, content, metadata)
            
            success_count = 0
            errors = []
            
            for chat_id in self.chat_ids:
                success, error = self._send_to_chat(chat_id, message)
                if success:
                    success_count += 1
                else:
                    errors.append(f"Chat {chat_id}: {error}")
            
            if success_count > 0:
                self.rate_limiter.record_send()
                logger.info(f"Telegram message sent to {success_count}/{len(self.chat_ids)} chats")
                
                if errors:
                    return True, f"Partial success. Errors: {'; '.join(errors)}"
                return True, ""
            else:
                error_msg = f"Failed to send to all chats: {'; '.join(errors)}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Telegram send error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _send_to_chat(self, chat_id: str, message: str) -> Tuple[bool, str]:
        """Send message to a specific chat."""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                },
                timeout=10
            )
            response.raise_for_status()
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def _format_message(self, title: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Format message for Telegram."""
        message = f"<b>🔔 {title}</b>\n\n{content}"
        
        if metadata:
            message += "\n\n<i>Details:</i>"
            if 'server_name' in metadata:
                message += f"\n🖥 <b>Server:</b> {metadata['server_name']}"
            if 'channel_name' in metadata:
                message += f"\n📢 <b>Channel:</b> #{metadata['channel_name']}"
            if 'author' in metadata:
                message += f"\n👤 <b>Author:</b> {metadata['author']}"
        
        # Telegram message limit is 4096 characters
        if len(message) > 4096:
            message = message[:4093] + "..."
        
        return message

class DesktopNotificationChannel(NotificationChannel):
    """Desktop notification channel (cross-platform)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.app_name = config.get('app_name', 'Discord Monitor')
        self.icon_path = config.get('icon_path')
        self._detect_notification_system()
    
    def _detect_notification_system(self):
        """Detect the available notification system."""
        self.system = platform.system().lower()
        
        if self.system == 'darwin':  # macOS
            self.method = 'osascript'
        elif self.system == 'linux':
            # Check if notify-send is available
            try:
                subprocess.run(['which', 'notify-send'], 
                             check=True, capture_output=True)
                self.method = 'notify-send'
            except subprocess.CalledProcessError:
                self.method = None
        elif self.system == 'windows':
            # Try to import win10toast
            try:
                import win10toast
                self.method = 'win10toast'
            except ImportError:
                self.method = None
        else:
            self.method = None
    
    def send(self, title: str, content: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Send desktop notification."""
        if not self.can_send():
            return False, "Rate limit exceeded"
        
        if not self.method:
            return False, "No desktop notification system available"
        
        try:
            if self.method == 'osascript':
                success, error = self._send_macos(title, content)
            elif self.method == 'notify-send':
                success, error = self._send_linux(title, content)
            elif self.method == 'win10toast':
                success, error = self._send_windows(title, content)
            else:
                return False, "Unsupported notification method"
            
            if success:
                self.rate_limiter.record_send()
                logger.info(f"Desktop notification sent: {title}")
            
            return success, error
            
        except Exception as e:
            error_msg = f"Desktop notification error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _send_macos(self, title: str, content: str) -> Tuple[bool, str]:
        """Send notification on macOS using osascript."""
        try:
            script = f'''
            display notification "{content.replace('"', '\\"')}" with title "{title.replace('"', '\\"')}"
            '''
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"macOS notification failed: {e.stderr.decode()}"
    
    def _send_linux(self, title: str, content: str) -> Tuple[bool, str]:
        """Send notification on Linux using notify-send."""
        try:
            cmd = ['notify-send', title, content]
            if self.icon_path and os.path.exists(self.icon_path):
                cmd.extend(['-i', self.icon_path])
            
            subprocess.run(cmd, check=True, capture_output=True)
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, f"Linux notification failed: {e.stderr.decode()}"
    
    def _send_windows(self, title: str, content: str) -> Tuple[bool, str]:
        """Send notification on Windows using win10toast."""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                title, content,
                icon_path=self.icon_path,
                duration=10,
                threaded=True
            )
            return True, ""
        except Exception as e:
            return False, f"Windows notification failed: {str(e)}"

class SlackWebhookChannel(NotificationChannel):
    """Slack webhook notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config['webhook_url']
        self.channel = config.get('channel', '#general')
        self.username = config.get('username', 'Discord Monitor')
        self.icon_emoji = config.get('icon_emoji', ':bell:')
    
    def send(self, title: str, content: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Send Slack notification."""
        if not self.can_send():
            return False, "Rate limit exceeded"
        
        try:
            payload = {
                'channel': self.channel,
                'username': self.username,
                'icon_emoji': self.icon_emoji,
                'attachments': [{
                    'color': 'good',
                    'title': title,
                    'text': content,
                    'footer': 'Discord Monitor',
                    'ts': int(datetime.now().timestamp())
                }]
            }
            
            if metadata:
                fields = []
                if 'server_name' in metadata:
                    fields.append({'title': 'Server', 'value': metadata['server_name'], 'short': True})
                if 'channel_name' in metadata:
                    fields.append({'title': 'Channel', 'value': f"#{metadata['channel_name']}", 'short': True})
                if 'author' in metadata:
                    fields.append({'title': 'Author', 'value': metadata['author'], 'short': True})
                
                if fields:
                    payload['attachments'][0]['fields'] = fields
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.rate_limiter.record_send()
            logger.info(f"Slack notification sent: {title}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Slack notification failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

class TeamsWebhookChannel(NotificationChannel):
    """Microsoft Teams webhook notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config['webhook_url']
    
    def send(self, title: str, content: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Send Teams notification."""
        if not self.can_send():
            return False, "Rate limit exceeded"
        
        try:
            payload = {
                '@type': 'MessageCard',
                '@context': 'https://schema.org/extensions',
                'summary': title,
                'themeColor': '5865F2',
                'sections': [{
                    'activityTitle': title,
                    'text': content,
                    'markdown': True
                }]
            }
            
            if metadata:
                facts = []
                if 'server_name' in metadata:
                    facts.append({'name': 'Server', 'value': metadata['server_name']})
                if 'channel_name' in metadata:
                    facts.append({'name': 'Channel', 'value': f"#{metadata['channel_name']}"})
                if 'author' in metadata:
                    facts.append({'name': 'Author', 'value': metadata['author']})
                
                if facts:
                    payload['sections'][0]['facts'] = facts
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.rate_limiter.record_send()
            logger.info(f"Teams notification sent: {title}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Teams notification failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

class CustomWebhookChannel(NotificationChannel):
    """Custom webhook notification channel for IFTTT, Zapier, etc."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config['webhook_url']
        self.method = config.get('method', 'POST').upper()
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.payload_template = config.get('payload_template', {})
    
    def send(self, title: str, content: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
        """Send custom webhook notification."""
        if not self.can_send():
            return False, "Rate limit exceeded"
        
        try:
            # Build payload from template
            payload = self.payload_template.copy()
            payload.update({
                'title': title,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })
            
            if self.method == 'GET':
                response = requests.get(self.webhook_url, params=payload, 
                                      headers=self.headers, timeout=10)
            else:
                response = requests.request(self.method, self.webhook_url, 
                                          json=payload, headers=self.headers, timeout=10)
            
            response.raise_for_status()
            
            self.rate_limiter.record_send()
            logger.info(f"Custom webhook notification sent: {title}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Custom webhook failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

class NotificationChannelFactory:
    """Factory for creating notification channel instances."""
    
    CHANNEL_CLASSES = {
        'email': EmailNotificationChannel,
        'telegram': TelegramNotificationChannel,
        'desktop': DesktopNotificationChannel,
        'slack': SlackWebhookChannel,
        'teams': TeamsWebhookChannel,
        'webhook': CustomWebhookChannel,
    }
    
    @classmethod
    def create_channel(cls, channel_type: str, config: Dict[str, Any]) -> NotificationChannel:
        """Create a notification channel instance."""
        if channel_type not in cls.CHANNEL_CLASSES:
            raise ValueError(f"Unknown notification channel type: {channel_type}")
        
        channel_class = cls.CHANNEL_CLASSES[channel_type]
        return channel_class(config)
    
    @classmethod
    def get_available_channel_types(cls) -> List[str]:
        """Get list of available channel types."""
        return list(cls.CHANNEL_CLASSES.keys())