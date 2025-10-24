"""
External tool integrations for Calendar, Todo lists, Note-taking apps,
IFTTT, Zapier, and RSS feed generation.
"""

import logging
import requests
import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import re
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlencode
import base64

from .notification_database import DB_NAME

logger = logging.getLogger('external_integrations')

class ExternalIntegration(ABC):
    """Abstract base class for external integrations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integration_id = config.get('integration_id')
        self.last_sync = None
        self.error_count = 0
    
    @abstractmethod
    def process_notification(self, notification: Dict[str, Any]) -> Tuple[bool, str]:
        """Process a notification through this integration."""
        pass
    
    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Test the integration connection."""
        pass
    
    def record_sync(self, success: bool, error_message: str = None):
        """Record sync attempt in database."""
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            now = int(datetime.now().timestamp())
            if success:
                self.error_count = 0
            else:
                self.error_count += 1
            
            cursor.execute('''
            UPDATE external_integrations 
            SET last_sync = ?, error_count = ?
            WHERE id = ?
            ''', (now, self.error_count, self.integration_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record sync for integration {self.integration_id}: {e}")

class GoogleCalendarIntegration(ExternalIntegration):
    """Google Calendar integration for creating events from Discord messages."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.access_token = config['access_token']
        self.refresh_token = config['refresh_token']
        self.calendar_id = config.get('calendar_id', 'primary')
        self.client_id = config['client_id']
        self.client_secret = config['client_secret']
        
        # Patterns to detect events in messages
        self.event_patterns = [
            r'(?:meeting|call|session|event)\s+(?:at|on)\s+(\d{1,2}:\d{2}(?:\s*[AP]M)?)',
            r'(?:schedule|plan|arrange)\s+.*?(?:for|at)\s+(\d{1,2}:\d{2}(?:\s*[AP]M)?)',
            r'(\d{1,2}/\d{1,2}(?:/\d{4})?)\s+(?:at\s+)?(\d{1,2}:\d{2}(?:\s*[AP]M)?)',
            r'(?:tomorrow|today)\s+(?:at\s+)?(\d{1,2}:\d{2}(?:\s*[AP]M)?)'
        ]
    
    def process_notification(self, notification: Dict[str, Any]) -> Tuple[bool, str]:
        """Process notification to create calendar events."""
        try:
            content = notification['content']
            metadata = notification.get('metadata', {})
            
            # Extract potential events from content
            events = self._extract_events(content, metadata)
            
            if not events:
                return True, "No events detected"
            
            # Create calendar events
            created_count = 0
            for event in events:
                success, error = self._create_calendar_event(event)
                if success:
                    created_count += 1
                else:
                    logger.warning(f"Failed to create calendar event: {error}")
            
            if created_count > 0:
                self.record_sync(True)
                return True, f"Created {created_count} calendar events"
            else:
                return False, "Failed to create any calendar events"
            
        except Exception as e:
            error_msg = f"Calendar integration error: {str(e)}"
            logger.error(error_msg)
            self.record_sync(False, error_msg)
            return False, error_msg
    
    def _extract_events(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract potential events from message content."""
        events = []
        
        # Look for time patterns
        for pattern in self.event_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                event = self._parse_event_from_match(match, content, metadata)
                if event:
                    events.append(event)
        
        # Look for explicit event keywords
        if any(keyword in content.lower() for keyword in ['meeting', 'call', 'event', 'session', 'appointment']):
            # Create a general event
            event = {
                'summary': self._extract_event_title(content),
                'description': f"From Discord: {content[:500]}",
                'source': {
                    'server_name': metadata.get('server_name', 'Unknown Server'),
                    'channel_name': metadata.get('channel_name', 'Unknown Channel'),
                    'author': metadata.get('author', 'Unknown Author'),
                    'message_id': metadata.get('message_id')
                }
            }
            
            # Try to extract time
            time_match = re.search(r'(\d{1,2}:\d{2}(?:\s*[AP]M)?)', content, re.IGNORECASE)
            if time_match:
                event['start_time'] = time_match.group(1)
                events.append(event)
        
        return events
    
    def _parse_event_from_match(self, match: re.Match, content: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse event details from regex match."""
        try:
            # Extract context around the match
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end]
            
            return {
                'summary': self._extract_event_title(context),
                'description': f"From Discord: {content[:500]}",
                'start_time': match.group(1) if match.groups() else None,
                'source': {
                    'server_name': metadata.get('server_name', 'Unknown Server'),
                    'channel_name': metadata.get('channel_name', 'Unknown Channel'),
                    'author': metadata.get('author', 'Unknown Author'),
                    'message_id': metadata.get('message_id')
                }
            }
        except Exception as e:
            logger.warning(f"Failed to parse event from match: {e}")
            return None
    
    def _extract_event_title(self, content: str) -> str:
        """Extract a suitable title for the event."""
        # Try to find a descriptive title
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in ['meeting', 'call', 'session', 'event']):
                return sentence[:100]  # Limit title length
        
        # Fallback to first sentence
        first_sentence = sentences[0].strip()
        return first_sentence[:100] if first_sentence else "Discord Event"
    
    def _create_calendar_event(self, event_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Create event in Google Calendar."""
        try:
            # Prepare event data
            start_time = datetime.now() + timedelta(hours=1)  # Default to 1 hour from now
            
            if event_data.get('start_time'):
                # Parse time from event
                time_str = event_data['start_time']
                # Simple time parsing - could be enhanced
                try:
                    parsed_time = datetime.strptime(time_str, '%H:%M')
                    start_time = start_time.replace(
                        hour=parsed_time.hour,
                        minute=parsed_time.minute,
                        second=0,
                        microsecond=0
                    )
                except ValueError:
                    pass  # Use default time
            
            end_time = start_time + timedelta(hours=1)  # Default 1-hour duration
            
            calendar_event = {
                'summary': event_data['summary'],
                'description': event_data['description'],
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'source': {
                    'title': f"Discord - {event_data['source']['channel_name']}",
                    'url': f"https://discord.com/channels/{event_data['source'].get('server_id', '')}/{event_data['source'].get('channel_id', '')}/{event_data['source'].get('message_id', '')}"
                }
            }
            
            # Make API request
            response = requests.post(
                f'https://www.googleapis.com/calendar/v3/calendars/{self.calendar_id}/events',
                headers={
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json'
                },
                json=calendar_event,
                timeout=10
            )
            
            if response.status_code == 401:
                # Try to refresh token
                if self._refresh_access_token():
                    # Retry with new token
                    response = requests.post(
                        f'https://www.googleapis.com/calendar/v3/calendars/{self.calendar_id}/events',
                        headers={
                            'Authorization': f'Bearer {self.access_token}',
                            'Content-Type': 'application/json'
                        },
                        json=calendar_event,
                        timeout=10
                    )
            
            response.raise_for_status()
            
            event_id = response.json().get('id')
            logger.info(f"Created calendar event: {event_id}")
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def _refresh_access_token(self) -> bool:
        """Refresh the Google access token."""
        try:
            response = requests.post(
                'https://oauth2.googleapis.com/token',
                data={
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'refresh_token': self.refresh_token,
                    'grant_type': 'refresh_token'
                },
                timeout=10
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            # Update token in database
            self._update_token_in_db()
            
            return True
        except Exception as e:
            logger.error(f"Failed to refresh Google token: {e}")
            return False
    
    def _update_token_in_db(self):
        """Update access token in database."""
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            config = self.config.copy()
            config['access_token'] = self.access_token
            
            cursor.execute('''
            UPDATE external_integrations 
            SET config = ?
            WHERE id = ?
            ''', (json.dumps(config), self.integration_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update token in database: {e}")
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test Google Calendar connection."""
        try:
            response = requests.get(
                f'https://www.googleapis.com/calendar/v3/calendars/{self.calendar_id}',
                headers={'Authorization': f'Bearer {self.access_token}'},
                timeout=10
            )
            
            if response.status_code == 401:
                if self._refresh_access_token():
                    response = requests.get(
                        f'https://www.googleapis.com/calendar/v3/calendars/{self.calendar_id}',
                        headers={'Authorization': f'Bearer {self.access_token}'},
                        timeout=10
                    )
            
            response.raise_for_status()
            return True, "Connection successful"
            
        except Exception as e:
            return False, str(e)

class TodoistIntegration(ExternalIntegration):
    """Todoist integration for creating tasks from Discord messages."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_token = config['api_token']
        self.project_id = config.get('project_id')  # Optional specific project
        
        # Patterns to detect tasks
        self.task_patterns = [
            r'(?:todo|task|reminder|need to|have to|must|should).*?[:.]?\s*(.+)',
            r'(?:don\'t forget|remember to)\s+(.+)',
            r'(?:action item|ai)[:.]?\s*(.+)',
            r'^\s*[-*•]\s*(.+)',  # Bullet points
        ]
    
    def process_notification(self, notification: Dict[str, Any]) -> Tuple[bool, str]:
        """Process notification to create Todoist tasks."""
        try:
            content = notification['content']
            metadata = notification.get('metadata', {})
            
            # Extract potential tasks
            tasks = self._extract_tasks(content, metadata)
            
            if not tasks:
                return True, "No tasks detected"
            
            # Create tasks
            created_count = 0
            for task in tasks:
                success, error = self._create_task(task)
                if success:
                    created_count += 1
                else:
                    logger.warning(f"Failed to create task: {error}")
            
            if created_count > 0:
                self.record_sync(True)
                return True, f"Created {created_count} tasks"
            else:
                return False, "Failed to create any tasks"
            
        except Exception as e:
            error_msg = f"Todoist integration error: {str(e)}"
            logger.error(error_msg)
            self.record_sync(False, error_msg)
            return False, error_msg
    
    def _extract_tasks(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract potential tasks from message content."""
        tasks = []
        
        for pattern in self.task_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                task_content = match.group(1).strip()
                if len(task_content) > 10:  # Filter out very short matches
                    task = {
                        'content': task_content,
                        'description': f"From Discord ({metadata.get('server_name', 'Unknown')}/#{metadata.get('channel_name', 'Unknown')}): {content[:200]}",
                        'labels': ['discord', 'imported'],
                        'source': metadata
                    }
                    tasks.append(task)
        
        return tasks
    
    def _create_task(self, task_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Create task in Todoist."""
        try:
            payload = {
                'content': task_data['content'],
                'description': task_data['description']
            }
            
            if self.project_id:
                payload['project_id'] = self.project_id
            
            # Add labels if supported
            if task_data.get('labels'):
                payload['labels'] = task_data['labels']
            
            response = requests.post(
                'https://api.todoist.com/rest/v2/tasks',
                headers={
                    'Authorization': f'Bearer {self.api_token}',
                    'Content-Type': 'application/json'
                },
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            task_id = response.json().get('id')
            logger.info(f"Created Todoist task: {task_id}")
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test Todoist connection."""
        try:
            response = requests.get(
                'https://api.todoist.com/rest/v2/projects',
                headers={'Authorization': f'Bearer {self.api_token}'},
                timeout=10
            )
            response.raise_for_status()
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)

class NotionIntegration(ExternalIntegration):
    """Notion integration for creating pages/database entries from Discord messages."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.auth_token = config['auth_token']
        self.database_id = config['database_id']
        self.page_properties = config.get('page_properties', {})
    
    def process_notification(self, notification: Dict[str, Any]) -> Tuple[bool, str]:
        """Process notification to create Notion page."""
        try:
            content = notification['content']
            metadata = notification.get('metadata', {})
            
            page_data = self._prepare_page_data(content, metadata)
            success, error = self._create_notion_page(page_data)
            
            if success:
                self.record_sync(True)
                return True, "Created Notion page"
            else:
                self.record_sync(False, error)
                return False, error
            
        except Exception as e:
            error_msg = f"Notion integration error: {str(e)}"
            logger.error(error_msg)
            self.record_sync(False, error_msg)
            return False, error_msg
    
    def _prepare_page_data(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare page data for Notion."""
        title = self._extract_title(content)
        
        # Basic page structure
        page_data = {
            'parent': {'database_id': self.database_id},
            'properties': {
                'Title': {
                    'title': [{'text': {'content': title}}]
                },
                'Content': {
                    'rich_text': [{'text': {'content': content[:2000]}}]  # Notion limit
                },
                'Source': {
                    'rich_text': [{'text': {'content': 'Discord'}}]
                },
                'Server': {
                    'rich_text': [{'text': {'content': metadata.get('server_name', 'Unknown')}}]
                },
                'Channel': {
                    'rich_text': [{'text': {'content': f"#{metadata.get('channel_name', 'Unknown')}"}}]
                },
                'Author': {
                    'rich_text': [{'text': {'content': metadata.get('author', 'Unknown')}}]
                },
                'Created': {
                    'date': {'start': datetime.now().isoformat()}
                }
            }
        }
        
        # Add custom properties from config
        for prop_name, prop_config in self.page_properties.items():
            if prop_config['type'] == 'select':
                page_data['properties'][prop_name] = {
                    'select': {'name': prop_config['value']}
                }
            elif prop_config['type'] == 'multi_select':
                page_data['properties'][prop_name] = {
                    'multi_select': [{'name': val} for val in prop_config['value']]
                }
        
        return page_data
    
    def _extract_title(self, content: str) -> str:
        """Extract a suitable title from content."""
        # Use first sentence or first 50 characters
        sentences = content.split('.')
        title = sentences[0].strip()
        return title[:100] if title else "Discord Message"
    
    def _create_notion_page(self, page_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Create page in Notion."""
        try:
            response = requests.post(
                'https://api.notion.com/v1/pages',
                headers={
                    'Authorization': f'Bearer {self.auth_token}',
                    'Content-Type': 'application/json',
                    'Notion-Version': '2022-06-28'
                },
                json=page_data,
                timeout=10
            )
            response.raise_for_status()
            
            page_id = response.json().get('id')
            logger.info(f"Created Notion page: {page_id}")
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test Notion connection."""
        try:
            response = requests.get(
                f'https://api.notion.com/v1/databases/{self.database_id}',
                headers={
                    'Authorization': f'Bearer {self.auth_token}',
                    'Notion-Version': '2022-06-28'
                },
                timeout=10
            )
            response.raise_for_status()
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)

class RSSFeedGenerator:
    """Generate RSS feeds from Discord notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feed_path = Path(config.get('feed_path', 'data/discord_feed.xml'))
        self.feed_title = config.get('feed_title', 'Discord Notifications')
        self.feed_description = config.get('feed_description', 'Latest Discord notifications')
        self.feed_link = config.get('feed_link', 'http://localhost:8501')
        self.max_items = config.get('max_items', 50)
    
    def update_feed(self, notifications: List[Dict[str, Any]]):
        """Update RSS feed with latest notifications."""
        try:
            # Create RSS structure
            rss = ET.Element('rss', version='2.0')
            channel = ET.SubElement(rss, 'channel')
            
            # Channel metadata
            ET.SubElement(channel, 'title').text = self.feed_title
            ET.SubElement(channel, 'description').text = self.feed_description
            ET.SubElement(channel, 'link').text = self.feed_link
            ET.SubElement(channel, 'lastBuildDate').text = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
            
            # Add notification items
            for notification in notifications[-self.max_items:]:
                item = ET.SubElement(channel, 'item')
                
                ET.SubElement(item, 'title').text = notification.get('title', 'Discord Notification')
                ET.SubElement(item, 'description').text = notification.get('content', '')
                ET.SubElement(item, 'link').text = self._generate_item_link(notification)
                ET.SubElement(item, 'guid').text = str(notification.get('id', ''))
                
                # Format publication date
                if 'created_at' in notification:
                    pub_date = datetime.fromtimestamp(notification['created_at'])
                    ET.SubElement(item, 'pubDate').text = pub_date.strftime('%a, %d %b %Y %H:%M:%S GMT')
                
                # Add category for server/channel
                metadata = notification.get('metadata', {})
                if metadata.get('server_name'):
                    ET.SubElement(item, 'category').text = metadata['server_name']
            
            # Write to file
            self.feed_path.parent.mkdir(exist_ok=True)
            tree = ET.ElementTree(rss)
            tree.write(self.feed_path, encoding='utf-8', xml_declaration=True)
            
            logger.info(f"Updated RSS feed with {len(notifications)} items")
            
        except Exception as e:
            logger.error(f"Failed to update RSS feed: {e}")
    
    def _generate_item_link(self, notification: Dict[str, Any]) -> str:
        """Generate link for RSS item."""
        metadata = notification.get('metadata', {})
        server_id = metadata.get('server_id', '')
        channel_id = metadata.get('channel_id', '')
        message_id = metadata.get('message_id', '')
        
        if server_id and channel_id and message_id:
            return f"https://discord.com/channels/{server_id}/{channel_id}/{message_id}"
        else:
            return self.feed_link

class IFTTTWebhookIntegration(ExternalIntegration):
    """IFTTT webhook integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_key = config['webhook_key']
        self.event_name = config.get('event_name', 'discord_notification')
    
    def process_notification(self, notification: Dict[str, Any]) -> Tuple[bool, str]:
        """Send notification to IFTTT."""
        try:
            payload = {
                'value1': notification.get('title', ''),
                'value2': notification.get('content', ''),
                'value3': json.dumps(notification.get('metadata', {}))
            }
            
            response = requests.post(
                f'https://maker.ifttt.com/trigger/{self.event_name}/with/key/{self.webhook_key}',
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            self.record_sync(True)
            return True, "Sent to IFTTT"
            
        except Exception as e:
            error_msg = f"IFTTT integration error: {str(e)}"
            logger.error(error_msg)
            self.record_sync(False, error_msg)
            return False, error_msg
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test IFTTT webhook."""
        try:
            test_payload = {
                'value1': 'Test notification',
                'value2': 'This is a test from Discord Monitor',
                'value3': '{"test": true}'
            }
            
            response = requests.post(
                f'https://maker.ifttt.com/trigger/{self.event_name}/with/key/{self.webhook_key}',
                json=test_payload,
                timeout=10
            )
            response.raise_for_status()
            return True, "Test successful"
        except Exception as e:
            return False, str(e)

class ZapierWebhookIntegration(ExternalIntegration):
    """Zapier webhook integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config['webhook_url']
    
    def process_notification(self, notification: Dict[str, Any]) -> Tuple[bool, str]:
        """Send notification to Zapier."""
        try:
            payload = {
                'title': notification.get('title', ''),
                'content': notification.get('content', ''),
                'priority': notification.get('priority', 2),
                'timestamp': datetime.now().isoformat(),
                'metadata': notification.get('metadata', {})
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            self.record_sync(True)
            return True, "Sent to Zapier"
            
        except Exception as e:
            error_msg = f"Zapier integration error: {str(e)}"
            logger.error(error_msg)
            self.record_sync(False, error_msg)
            return False, error_msg
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test Zapier webhook."""
        try:
            test_payload = {
                'title': 'Test notification',
                'content': 'This is a test from Discord Monitor',
                'priority': 1,
                'timestamp': datetime.now().isoformat(),
                'metadata': {'test': True}
            }
            
            response = requests.post(
                self.webhook_url,
                json=test_payload,
                timeout=10
            )
            response.raise_for_status()
            return True, "Test successful"
        except Exception as e:
            return False, str(e)

class ExternalIntegrationManager:
    """Manages all external integrations."""
    
    INTEGRATION_CLASSES = {
        'google_calendar': GoogleCalendarIntegration,
        'todoist': TodoistIntegration,
        'notion': NotionIntegration,
        'ifttt': IFTTTWebhookIntegration,
        'zapier': ZapierWebhookIntegration,
    }
    
    def __init__(self):
        self.integrations: Dict[str, ExternalIntegration] = {}
        self.rss_generator: Optional[RSSFeedGenerator] = None
        self._load_integrations()
    
    def _load_integrations(self):
        """Load integrations from database."""
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, integration_type, name, config
            FROM external_integrations
            WHERE is_active = 1
            ''')
            
            for row in cursor.fetchall():
                integration_id, integration_type, name, config_json = row
                try:
                    config = json.loads(config_json)
                    config['integration_id'] = integration_id
                    
                    if integration_type in self.INTEGRATION_CLASSES:
                        integration_class = self.INTEGRATION_CLASSES[integration_type]
                        integration = integration_class(config)
                        self.integrations[f"{integration_type}_{name}"] = integration
                        logger.info(f"Loaded integration: {integration_type}_{name}")
                    
                    elif integration_type == 'rss':
                        self.rss_generator = RSSFeedGenerator(config)
                        logger.info("Loaded RSS generator")
                    
                except Exception as e:
                    logger.error(f"Failed to load integration {integration_type}_{name}: {e}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to load integrations: {e}")
    
    def process_notification(self, notification: Dict[str, Any]):
        """Process notification through all active integrations."""
        results = {}
        
        for name, integration in self.integrations.items():
            try:
                success, message = integration.process_notification(notification)
                results[name] = {'success': success, 'message': message}
                
                if success:
                    logger.info(f"Integration {name} processed notification successfully")
                else:
                    logger.warning(f"Integration {name} failed: {message}")
                    
            except Exception as e:
                error_msg = f"Integration {name} error: {str(e)}"
                logger.error(error_msg)
                results[name] = {'success': False, 'message': error_msg}
        
        return results
    
    def update_rss_feed(self, notifications: List[Dict[str, Any]]):
        """Update RSS feed with notifications."""
        if self.rss_generator:
            try:
                self.rss_generator.update_feed(notifications)
            except Exception as e:
                logger.error(f"Failed to update RSS feed: {e}")
    
    def test_all_integrations(self) -> Dict[str, Dict[str, Any]]:
        """Test all integrations."""
        results = {}
        
        for name, integration in self.integrations.items():
            try:
                success, message = integration.test_connection()
                results[name] = {'success': success, 'message': message}
            except Exception as e:
                results[name] = {'success': False, 'message': str(e)}
        
        return results
    
    def reload_integrations(self):
        """Reload integrations from database."""
        self.integrations.clear()
        self.rss_generator = None
        self._load_integrations()
        logger.info("External integrations reloaded")

# Global integration manager instance
_integration_manager: Optional[ExternalIntegrationManager] = None

def get_integration_manager() -> ExternalIntegrationManager:
    """Get or create the global integration manager instance."""
    global _integration_manager
    
    if _integration_manager is None:
        _integration_manager = ExternalIntegrationManager()
    
    return _integration_manager