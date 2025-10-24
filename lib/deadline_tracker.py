"""
Deadline Tracker Module for Discord Monitoring
Extracts and tracks deadlines from messages with Czech language support
"""

import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta
import json

class DeadlineTracker:
    """Tracks and manages deadlines from Discord messages"""
    
    def __init__(self, db_path: str = 'data/db.sqlite'):
        self.db_path = db_path
        
        # Czech and English deadline patterns
        self.deadline_patterns = {
            'relative_days': {
                'patterns': [
                    (r'\b(dnes|today)\b', 0),
                    (r'\b(zítra|tomorrow)\b', 1),
                    (r'\b(pozítří|day after tomorrow)\b', 2),
                    (r'\b(včera|yesterday)\b', -1),
                ],
                'priority': 3
            },
            'relative_weeks': {
                'patterns': [
                    (r'\b(tento|this)\s+(týden|week)\b', 0),
                    (r'\b(příští|next)\s+(týden|week)\b', 7),
                    (r'\b(za\s+týden|in\s+a\s+week)\b', 7),
                    (r'\b(za\s+(\d+)\s+týdn[ůy]?|in\s+(\d+)\s+weeks?)\b', 'weeks'),
                ],
                'priority': 2
            },
            'weekdays': {
                'patterns': [
                    (r'\b(pondělí|monday)\b', 0),
                    (r'\b(úterý|tuesday)\b', 1),
                    (r'\b(středa|wednesday)\b', 2),
                    (r'\b(čtvrtek|thursday)\b', 3),
                    (r'\b(pátek|friday)\b', 4),
                    (r'\b(sobota|saturday)\b', 5),
                    (r'\b(neděle|sunday)\b', 6),
                ],
                'priority': 2
            },
            'specific_date': {
                'patterns': [
                    (r'\b(\d{1,2})\.\s?(\d{1,2})\.?\s?(\d{2,4})?\b', 'czech_date'),  # 15.3.2024, 15. 3.
                    (r'\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b', 'slash_date'),  # 15/3/2024
                    (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', 'iso_date'),  # 2024-03-15
                ],
                'priority': 4
            },
            'prepositions': {
                'patterns': [
                    (r'\bdo\s+(.+?)(?:\.|,|!|\?|$)', 'until'),  # do pátku, do 15.3.
                    (r'\bkončí\s+(.+?)(?:\.|,|!|\?|$)', 'ends'),  # končí zítra
                    (r'\bdeadline\s+(.+?)(?:\.|,|!|\?|$)', 'deadline'),
                    (r'\bnejpozději\s+(.+?)(?:\.|,|!|\?|$)', 'latest'),  # nejpozději do
                    (r'\bby\s+(.+?)(?:\.|,|!|\?|$)', 'by'),
                ],
                'priority': 1
            },
            'time_patterns': {
                'patterns': [
                    (r'\b(\d{1,2}):(\d{2})\b', 'time'),  # 14:30
                    (r'\b(\d{1,2})\s*(hodin[ay]?|hours?|h)\b', 'hour'),  # 14 hodin, 2 hours
                    (r'\bv\s+(\d{1,2})\b', 'at_hour'),  # v 15, at 3
                ],
                'priority': 1
            },
            'urgency_indicators': {
                'patterns': [
                    (r'\b(poslední\s+šance|last\s+chance)\b', 5),
                    (r'\b(urgentní|urgent|naléhavé)\b', 4),
                    (r'\b(rychle|hurry|pospěš|asap)\b', 3),
                    (r'\b(brzy|soon|zanedlouho)\b', 2),
                    (r'\b(časově\s+omezené|limited\s+time)\b', 3),
                ],
                'priority': 0
            }
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for category, data in self.deadline_patterns.items():
            self.compiled_patterns[category] = []
            for pattern_data in data['patterns']:
                if isinstance(pattern_data, tuple):
                    pattern, value = pattern_data
                    compiled = re.compile(pattern, re.IGNORECASE | re.UNICODE)
                    self.compiled_patterns[category].append((compiled, value))
    
    def extract_deadlines(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all deadlines from messages"""
        deadlines = []
        
        for message in messages:
            content = message.get('content', '')
            if not content:
                continue
            
            # Extract deadline from this message
            deadline_info = self._extract_deadline_from_text(content, message)
            
            if deadline_info:
                deadline_info['message_id'] = message.get('id')
                deadline_info['channel_id'] = message.get('channel_id')
                deadline_info['original_message'] = content[:200]  # Store first 200 chars
                deadlines.append(deadline_info)
                
                # Save to database
                self._save_deadline(deadline_info)
        
        return deadlines
    
    def _extract_deadline_from_text(self, text: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract deadline information from text"""
        deadline_info = {
            'deadline_date': None,
            'deadline_text': '',
            'urgency_level': 1,
            'confidence': 0.5,
            'extracted_patterns': []
        }
        
        message_timestamp = message.get('sent_at', datetime.now().timestamp())
        message_date = datetime.fromtimestamp(message_timestamp)
        
        # Check for preposition patterns first (they contain the main deadline info)
        for pattern, prep_type in self.compiled_patterns['prepositions']:
            match = pattern.search(text)
            if match:
                deadline_phrase = match.group(1)
                deadline_info['deadline_text'] = deadline_phrase
                deadline_info['extracted_patterns'].append(f"{prep_type}: {deadline_phrase}")
                
                # Try to parse the deadline phrase
                parsed_date = self._parse_deadline_phrase(deadline_phrase, message_date)
                if parsed_date:
                    deadline_info['deadline_date'] = int(parsed_date.timestamp())
                    deadline_info['confidence'] = 0.8
                    break
        
        # If no preposition pattern found, check for direct date mentions
        if not deadline_info['deadline_date']:
            # Check specific dates
            for pattern, date_type in self.compiled_patterns['specific_date']:
                match = pattern.search(text)
                if match:
                    parsed_date = self._parse_specific_date(match, date_type, message_date)
                    if parsed_date:
                        deadline_info['deadline_date'] = int(parsed_date.timestamp())
                        deadline_info['deadline_text'] = match.group(0)
                        deadline_info['confidence'] = 0.9
                        deadline_info['extracted_patterns'].append(f"date: {match.group(0)}")
                        break
            
            # Check relative days
            if not deadline_info['deadline_date']:
                for pattern, days_offset in self.compiled_patterns['relative_days']:
                    if pattern.search(text):
                        deadline_date = message_date + timedelta(days=days_offset)
                        deadline_info['deadline_date'] = int(deadline_date.timestamp())
                        deadline_info['deadline_text'] = pattern.pattern
                        deadline_info['confidence'] = 0.7
                        deadline_info['extracted_patterns'].append(f"relative: +{days_offset} days")
                        break
        
        # Check urgency indicators
        urgency_level = 1
        for pattern, urgency in self.compiled_patterns['urgency_indicators']:
            if pattern.search(text):
                urgency_level = max(urgency_level, urgency)
                deadline_info['extracted_patterns'].append(f"urgency: level {urgency}")
        
        deadline_info['urgency_level'] = urgency_level
        
        # Only return if we found a deadline
        if deadline_info['deadline_date'] or deadline_info['urgency_level'] >= 3:
            return deadline_info
        
        return None
    
    def _parse_deadline_phrase(self, phrase: str, reference_date: datetime) -> Optional[datetime]:
        """Parse a deadline phrase into a datetime"""
        # Try parsing weekdays
        weekday_map = {
            'pondělí': 0, 'monday': 0,
            'úterý': 1, 'tuesday': 1,
            'středa': 2, 'wednesday': 2,
            'čtvrtek': 3, 'thursday': 3,
            'pátek': 4, 'friday': 4,
            'sobota': 5, 'saturday': 5,
            'neděle': 6, 'sunday': 6
        }
        
        phrase_lower = phrase.lower()
        
        # Check for weekday
        for day_name, day_num in weekday_map.items():
            if day_name in phrase_lower:
                # Find next occurrence of this weekday
                days_ahead = day_num - reference_date.weekday()
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                return reference_date + timedelta(days_ahead)
        
        # Check for relative days
        if 'dnes' in phrase_lower or 'today' in phrase_lower:
            return reference_date
        elif 'zítra' in phrase_lower or 'tomorrow' in phrase_lower:
            return reference_date + timedelta(days=1)
        elif 'pozítří' in phrase_lower:
            return reference_date + timedelta(days=2)
        
        # Try parsing specific date
        date_match = re.search(r'(\d{1,2})\.\s?(\d{1,2})\.?', phrase)
        if date_match:
            day = int(date_match.group(1))
            month = int(date_match.group(2))
            year = reference_date.year
            
            try:
                deadline = datetime(year, month, day)
                # If date is in the past, assume next year
                if deadline < reference_date:
                    deadline = datetime(year + 1, month, day)
                return deadline
            except ValueError:
                pass
        
        # Try dateutil parser as fallback
        try:
            return parser.parse(phrase, default=reference_date, fuzzy=True)
        except:
            pass
        
        return None
    
    def _parse_specific_date(self, match: re.Match, date_type: str, 
                            reference_date: datetime) -> Optional[datetime]:
        """Parse specific date formats"""
        try:
            if date_type == 'czech_date':  # 15.3.2024 or 15. 3.
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3)) if match.group(3) else reference_date.year
                
                # Handle 2-digit years
                if year < 100:
                    year += 2000
                
                deadline = datetime(year, month, day)
                # If date is in the past and no year specified, assume next year
                if deadline < reference_date and not match.group(3):
                    deadline = datetime(year + 1, month, day)
                return deadline
                
            elif date_type == 'slash_date':  # 15/3/2024
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3)) if match.group(3) else reference_date.year
                
                if year < 100:
                    year += 2000
                    
                return datetime(year, month, day)
                
            elif date_type == 'iso_date':  # 2024-03-15
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                return datetime(year, month, day)
                
        except ValueError:
            pass
        
        return None
    
    def _save_deadline(self, deadline_info: Dict[str, Any]):
        """Save deadline to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO deadlines 
                (message_id, channel_id, deadline_date, deadline_text, 
                 urgency_level, reminder_sent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                deadline_info.get('message_id'),
                deadline_info.get('channel_id'),
                deadline_info.get('deadline_date'),
                deadline_info.get('deadline_text', '')[:200],
                deadline_info.get('urgency_level', 1),
                0,  # reminder_sent = False
                int(datetime.now().timestamp())
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving deadline: {e}")
    
    def get_upcoming_deadlines(self, hours_ahead: int = 48) -> List[Dict[str, Any]]:
        """Get deadlines coming up in the next N hours"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            now = int(datetime.now().timestamp())
            future_threshold = int((datetime.now() + timedelta(hours=hours_ahead)).timestamp())
            
            cursor.execute('''
                SELECT d.*, m.content, c.name as channel_name
                FROM deadlines d
                JOIN messages m ON d.message_id = m.id
                LEFT JOIN channels c ON d.channel_id = c.id
                WHERE d.deadline_date >= ? AND d.deadline_date <= ?
                    AND d.reminder_sent = 0
                ORDER BY d.deadline_date ASC, d.urgency_level DESC
                LIMIT 20
            ''', (now, future_threshold))
            
            deadlines = []
            for row in cursor.fetchall():
                deadline = dict(row)
                # Add human-readable time
                if deadline.get('deadline_date'):
                    deadline['deadline_datetime'] = datetime.fromtimestamp(
                        deadline['deadline_date']
                    ).strftime('%Y-%m-%d %H:%M')
                    
                    # Calculate time remaining
                    time_remaining = deadline['deadline_date'] - now
                    if time_remaining < 3600:  # Less than 1 hour
                        deadline['time_remaining'] = f"{time_remaining // 60} minut"
                    elif time_remaining < 86400:  # Less than 1 day
                        deadline['time_remaining'] = f"{time_remaining // 3600} hodin"
                    else:
                        deadline['time_remaining'] = f"{time_remaining // 86400} dní"
                
                deadlines.append(deadline)
            
            conn.close()
            return deadlines
            
        except Exception as e:
            print(f"Error getting deadlines: {e}")
            return []
    
    def mark_reminder_sent(self, deadline_id: int):
        """Mark a deadline reminder as sent"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE deadlines 
                SET reminder_sent = 1 
                WHERE id = ?
            ''', (deadline_id,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error marking reminder: {e}")
    
    def get_expired_deadlines(self, hours_past: int = 24) -> List[Dict[str, Any]]:
        """Get recently expired deadlines"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            now = int(datetime.now().timestamp())
            past_threshold = int((datetime.now() - timedelta(hours=hours_past)).timestamp())
            
            cursor.execute('''
                SELECT d.*, m.content, c.name as channel_name
                FROM deadlines d
                JOIN messages m ON d.message_id = m.id
                LEFT JOIN channels c ON d.channel_id = c.id
                WHERE d.deadline_date < ? AND d.deadline_date >= ?
                ORDER BY d.deadline_date DESC
                LIMIT 10
            ''', (now, past_threshold))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            print(f"Error getting expired deadlines: {e}")
            return []