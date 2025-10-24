"""
Pattern detection system for identifying group activities, FOMO moments, and important events.
Specialized for detecting group purchases, events, and coordinated activities.
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

logger = logging.getLogger('discord_bot')


@dataclass
class GroupActivity:
    """Represents a detected group activity."""
    activity_type: str  # 'purchase', 'event', 'decision', etc.
    confidence: float
    participants: List[str]
    start_time: int
    end_time: int
    channel_id: Optional[int]
    key_messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    @property
    def duration_minutes(self) -> float:
        """Get activity duration in minutes."""
        return (self.end_time - self.start_time) / 60
    
    @property
    def is_active(self) -> bool:
        """Check if activity is still active (within last hour)."""
        return (time.time() - self.end_time) < 3600
    
    @property
    def participant_count(self) -> int:
        """Get number of participants."""
        return len(self.participants)


class PatternDetector:
    """Detect patterns in Discord messages for group activities and FOMO moments."""
    
    def __init__(self):
        self._initialize_patterns()
        self._initialize_detectors()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for different activity types."""
        self.patterns = {
            'purchase': {
                'initiation': [
                    r'group\s*buy',
                    r'bulk\s*order',
                    r'split\s*(the\s*)?cost',
                    r'anyone\s*want\s*to\s*split',
                    r'going\s*in\s*on',
                ],
                'participation': [
                    r"(i'm|im|i\s*am)\s*in",
                    r'count\s*me\s*in',
                    r'sign\s*me\s*up',
                    r'put\s*me\s*down',
                    r'interested',
                    r'\+1',
                    r'yes\s*please',
                ],
                'details': [
                    r'\$\d+',
                    r'\d+\s*(usd|eur|gbp|czk)',
                    r'cost[s]?\s*\d+',
                    r'price[s]?\s*\d+',
                    r'total[s]?\s*\d+',
                    r'each',
                    r'per\s*person',
                ],
                'urgency': [
                    r'closing\s*soon',
                    r'last\s*chance',
                    r'ends?\s*(today|tonight|tomorrow)',
                    r'limited\s*(time|quantity|stock)',
                    r'only\s*\d+\s*left',
                ]
            },
            'event': {
                'planning': [
                    r'(let\'s|lets)\s*(meet|gather|hangout)',
                    r'planning\s*a\s*(meeting|event|party)',
                    r'when\s*(is|are)\s*(everyone|you)',
                    r'what\s*time\s*(works?|is\s*good)',
                    r'schedule\s*a',
                ],
                'confirmation': [
                    r'(i|we)\s*(can|will)\s*(make\s*it|be\s*there|attend|come)',
                    r'see\s*you\s*(there|then)',
                    r'confirmed',
                    r'rsvp',
                    r'attending',
                ],
                'logistics': [
                    r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                    r'\d{1,2}:\d{2}\s*(am|pm)?',
                    r'(morning|afternoon|evening|night)',
                    r'location|place|venue|where',
                    r'online|zoom|discord|meet',
                ]
            },
            'decision': {
                'voting': [
                    r'vote\s*for',
                    r'poll',
                    r'which\s*one',
                    r'option\s*[a-z1-9]',
                    r'choice\s*[a-z1-9]',
                    r'prefer',
                ],
                'consensus': [
                    r'(we|everyone)\s*(should|could|need)',
                    r'agree[d]?',
                    r'sounds\s*good',
                    r'works\s*for\s*me',
                    r'let\'s\s*go\s*with',
                    r'decided',
                ],
                'discussion': [
                    r'what\s*do\s*you\s*think',
                    r'thoughts?',
                    r'opinions?',
                    r'any\s*objections',
                    r'feedback',
                ]
            },
            'fomo': {
                'exclusive': [
                    r'exclusive',
                    r'limited\s*edition',
                    r'rare',
                    r'special\s*offer',
                    r'members?\s*only',
                ],
                'urgency': [
                    r'hurry',
                    r'quick',
                    r'fast',
                    r'don\'t\s*miss',
                    r'last\s*chance',
                    r'ending\s*soon',
                ],
                'social_proof': [
                    r'everyone\'s',
                    r'we\'re\s*all',
                    r'most\s*people',
                    r'already\s*\d+\s*people',
                    r'sold\s*out\s*fast',
                ]
            }
        }
    
    def _initialize_detectors(self):
        """Initialize specific activity detectors."""
        self.detectors = {
            'purchase': self._detect_purchase_activity,
            'event': self._detect_event_activity,
            'decision': self._detect_decision_activity,
            'fomo': self._detect_fomo_moment
        }
    
    def detect_activities(self, messages: List[Dict[str, Any]], 
                         time_window_minutes: int = 60) -> List[GroupActivity]:
        """
        Detect group activities in a list of messages.
        
        Args:
            messages: List of message dictionaries
            time_window_minutes: Time window to group related messages
            
        Returns:
            List of detected GroupActivity objects
        """
        if not messages:
            return []
        
        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.get('sent_at', 0))
        
        # Group messages into time windows
        message_groups = self._group_messages_by_time(sorted_messages, time_window_minutes)
        
        detected_activities = []
        
        for group in message_groups:
            if len(group) < 2:  # Need at least 2 messages for an activity
                continue
            
            # Try each detector
            for activity_type, detector in self.detectors.items():
                activity = detector(group)
                if activity and activity.confidence > 0.5:
                    detected_activities.append(activity)
        
        # Merge overlapping activities of the same type
        merged_activities = self._merge_overlapping_activities(detected_activities)
        
        # Sort by confidence and recency
        merged_activities.sort(key=lambda a: (a.confidence, a.end_time), reverse=True)
        
        return merged_activities
    
    def _group_messages_by_time(self, messages: List[Dict[str, Any]], 
                                window_minutes: int) -> List[List[Dict[str, Any]]]:
        """Group messages into time windows."""
        groups = []
        current_group = []
        last_timestamp = 0
        
        for msg in messages:
            timestamp = msg.get('sent_at', 0)
            
            if not current_group:
                current_group = [msg]
                last_timestamp = timestamp
            elif (timestamp - last_timestamp) <= (window_minutes * 60):
                current_group.append(msg)
                last_timestamp = timestamp
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [msg]
                last_timestamp = timestamp
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _detect_purchase_activity(self, messages: List[Dict[str, Any]]) -> Optional[GroupActivity]:
        """Detect group purchase activity."""
        pattern_matches = defaultdict(int)
        participants = set()
        key_messages = []
        
        for msg in messages:
            content = msg.get('content', '').lower()
            author = msg.get('author_name') or msg.get('author_id')
            
            if not content or not author:
                continue
            
            # Check purchase patterns
            for pattern_type, patterns in self.patterns['purchase'].items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        pattern_matches[pattern_type] += 1
                        participants.add(author)
                        
                        # Track key messages
                        if pattern_type in ['initiation', 'details', 'urgency']:
                            key_messages.append(msg)
                        
                        break
        
        # Calculate confidence based on pattern matches
        confidence = 0.0
        
        if pattern_matches['initiation'] > 0:
            confidence += 0.3
        if pattern_matches['participation'] >= 2:
            confidence += 0.3
        if pattern_matches['details'] > 0:
            confidence += 0.2
        if pattern_matches['urgency'] > 0:
            confidence += 0.2
        
        # Boost confidence based on participation
        if len(participants) >= 3:
            confidence *= 1.2
        elif len(participants) >= 5:
            confidence *= 1.5
        
        confidence = min(1.0, confidence)
        
        if confidence > 0.5:
            # Extract purchase details
            metadata = self._extract_purchase_metadata(messages)
            
            return GroupActivity(
                activity_type='group_purchase',
                confidence=confidence,
                participants=list(participants),
                start_time=messages[0].get('sent_at', 0),
                end_time=messages[-1].get('sent_at', 0),
                channel_id=messages[0].get('channel_id'),
                key_messages=key_messages[:5],  # Top 5 key messages
                metadata=metadata
            )
        
        return None
    
    def _detect_event_activity(self, messages: List[Dict[str, Any]]) -> Optional[GroupActivity]:
        """Detect event planning activity."""
        pattern_matches = defaultdict(int)
        participants = set()
        key_messages = []
        
        for msg in messages:
            content = msg.get('content', '').lower()
            author = msg.get('author_name') or msg.get('author_id')
            
            if not content or not author:
                continue
            
            # Check event patterns
            for pattern_type, patterns in self.patterns['event'].items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        pattern_matches[pattern_type] += 1
                        participants.add(author)
                        
                        if pattern_type in ['planning', 'logistics']:
                            key_messages.append(msg)
                        
                        break
        
        # Calculate confidence
        confidence = 0.0
        
        if pattern_matches['planning'] > 0:
            confidence += 0.35
        if pattern_matches['confirmation'] >= 2:
            confidence += 0.35
        if pattern_matches['logistics'] > 0:
            confidence += 0.3
        
        # Participation boost
        if len(participants) >= 3:
            confidence *= 1.15
        
        confidence = min(1.0, confidence)
        
        if confidence > 0.5:
            metadata = self._extract_event_metadata(messages)
            
            return GroupActivity(
                activity_type='event_planning',
                confidence=confidence,
                participants=list(participants),
                start_time=messages[0].get('sent_at', 0),
                end_time=messages[-1].get('sent_at', 0),
                channel_id=messages[0].get('channel_id'),
                key_messages=key_messages[:5],
                metadata=metadata
            )
        
        return None
    
    def _detect_decision_activity(self, messages: List[Dict[str, Any]]) -> Optional[GroupActivity]:
        """Detect group decision making."""
        pattern_matches = defaultdict(int)
        participants = set()
        key_messages = []
        
        for msg in messages:
            content = msg.get('content', '').lower()
            author = msg.get('author_name') or msg.get('author_id')
            
            if not content or not author:
                continue
            
            # Check decision patterns
            for pattern_type, patterns in self.patterns['decision'].items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        pattern_matches[pattern_type] += 1
                        participants.add(author)
                        
                        if pattern_type in ['voting', 'consensus']:
                            key_messages.append(msg)
                        
                        break
        
        # Calculate confidence
        confidence = 0.0
        
        if pattern_matches['voting'] > 0:
            confidence += 0.4
        if pattern_matches['consensus'] >= 2:
            confidence += 0.35
        if pattern_matches['discussion'] > 0:
            confidence += 0.25
        
        # Participation boost
        if len(participants) >= 3:
            confidence *= 1.1
        
        confidence = min(1.0, confidence)
        
        if confidence > 0.5:
            metadata = self._extract_decision_metadata(messages)
            
            return GroupActivity(
                activity_type='group_decision',
                confidence=confidence,
                participants=list(participants),
                start_time=messages[0].get('sent_at', 0),
                end_time=messages[-1].get('sent_at', 0),
                channel_id=messages[0].get('channel_id'),
                key_messages=key_messages[:5],
                metadata=metadata
            )
        
        return None
    
    def _detect_fomo_moment(self, messages: List[Dict[str, Any]]) -> Optional[GroupActivity]:
        """Detect FOMO (Fear of Missing Out) moments."""
        pattern_matches = defaultdict(int)
        participants = set()
        key_messages = []
        
        for msg in messages:
            content = msg.get('content', '').lower()
            author = msg.get('author_name') or msg.get('author_id')
            
            if not content or not author:
                continue
            
            # Check FOMO patterns
            for pattern_type, patterns in self.patterns['fomo'].items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        pattern_matches[pattern_type] += 1
                        participants.add(author)
                        key_messages.append(msg)
                        break
        
        # Calculate confidence
        confidence = 0.0
        
        if pattern_matches['exclusive'] > 0:
            confidence += 0.35
        if pattern_matches['urgency'] > 0:
            confidence += 0.35
        if pattern_matches['social_proof'] > 0:
            confidence += 0.3
        
        # FOMO is stronger with more participants
        if len(participants) >= 4:
            confidence *= 1.3
        
        confidence = min(1.0, confidence)
        
        if confidence > 0.5:
            return GroupActivity(
                activity_type='fomo_moment',
                confidence=confidence,
                participants=list(participants),
                start_time=messages[0].get('sent_at', 0),
                end_time=messages[-1].get('sent_at', 0),
                channel_id=messages[0].get('channel_id'),
                key_messages=key_messages[:5],
                metadata={'fomo_type': max(pattern_matches, key=pattern_matches.get)}
            )
        
        return None
    
    def _extract_purchase_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata about a purchase activity."""
        metadata = {}
        
        # Look for prices
        for msg in messages:
            content = msg.get('content', '')
            
            # Extract price
            price_match = re.search(r'\$(\d+(?:\.\d{2})?)', content)
            if price_match and 'price' not in metadata:
                metadata['price'] = price_match.group(1)
            
            # Extract item name (simple heuristic)
            if 'buying' in content.lower() or 'selling' in content.lower():
                # Try to extract what's being bought/sold
                words = content.split()
                for i, word in enumerate(words):
                    if word.lower() in ['buying', 'selling', 'group buy']:
                        if i + 1 < len(words):
                            metadata['item'] = ' '.join(words[i+1:i+4])
                            break
        
        return metadata
    
    def _extract_event_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata about an event."""
        metadata = {}
        
        for msg in messages:
            content = msg.get('content', '')
            
            # Extract time
            time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:am|pm)?)', content, re.IGNORECASE)
            if time_match and 'time' not in metadata:
                metadata['time'] = time_match.group(1)
            
            # Extract day
            days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for day in days:
                if day in content.lower() and 'day' not in metadata:
                    metadata['day'] = day.capitalize()
                    break
            
            # Extract location/platform
            if 'zoom' in content.lower():
                metadata['platform'] = 'Zoom'
            elif 'discord' in content.lower() and 'voice' in content.lower():
                metadata['platform'] = 'Discord Voice'
            elif 'location' in content.lower() or 'where' in content.lower():
                # Try to extract location after these keywords
                pass
        
        return metadata
    
    def _extract_decision_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata about a decision."""
        metadata = {}
        
        # Count votes/preferences
        options = defaultdict(int)
        
        for msg in messages:
            content = msg.get('content', '').lower()
            
            # Look for option indicators
            option_matches = re.findall(r'option\s*([a-z1-9])', content)
            for opt in option_matches:
                options[opt] += 1
            
            # Look for explicit votes
            if 'vote' in content:
                vote_match = re.search(r'vote\s*(?:for\s*)?([a-z1-9])', content)
                if vote_match:
                    options[vote_match.group(1)] += 1
        
        if options:
            metadata['vote_counts'] = dict(options)
            metadata['leading_option'] = max(options, key=options.get)
        
        return metadata
    
    def _merge_overlapping_activities(self, activities: List[GroupActivity]) -> List[GroupActivity]:
        """Merge overlapping activities of the same type."""
        if not activities:
            return []
        
        # Group by activity type
        by_type = defaultdict(list)
        for activity in activities:
            by_type[activity.activity_type].append(activity)
        
        merged = []
        
        for activity_type, type_activities in by_type.items():
            # Sort by start time
            type_activities.sort(key=lambda a: a.start_time)
            
            current = type_activities[0]
            
            for next_activity in type_activities[1:]:
                # Check if activities overlap or are very close (within 5 minutes)
                if next_activity.start_time <= current.end_time + 300:
                    # Merge activities
                    current = GroupActivity(
                        activity_type=activity_type,
                        confidence=max(current.confidence, next_activity.confidence),
                        participants=list(set(current.participants + next_activity.participants)),
                        start_time=min(current.start_time, next_activity.start_time),
                        end_time=max(current.end_time, next_activity.end_time),
                        channel_id=current.channel_id,
                        key_messages=current.key_messages + next_activity.key_messages,
                        metadata={**current.metadata, **next_activity.metadata}
                    )
                else:
                    merged.append(current)
                    current = next_activity
            
            merged.append(current)
        
        return merged
    
    def get_activity_summary(self, activity: GroupActivity) -> str:
        """Generate a human-readable summary of an activity."""
        summaries = {
            'group_purchase': f"🛒 Group purchase detected with {activity.participant_count} participants. "
                            f"Confidence: {activity.confidence:.0%}. "
                            f"{activity.metadata.get('item', 'Item')} for {activity.metadata.get('price', 'unknown price')}",
            
            'event_planning': f"📅 Event being planned with {activity.participant_count} participants. "
                            f"{activity.metadata.get('day', '')} {activity.metadata.get('time', '')} "
                            f"on {activity.metadata.get('platform', 'TBD')}",
            
            'group_decision': f"🗳️ Group decision in progress with {activity.participant_count} participants. "
                            f"Leading option: {activity.metadata.get('leading_option', 'undecided')}",
            
            'fomo_moment': f"⚡ FOMO moment detected! {activity.participant_count} people involved. "
                         f"Type: {activity.metadata.get('fomo_type', 'general')}"
        }
        
        return summaries.get(activity.activity_type, 
                            f"Group activity ({activity.activity_type}) with {activity.participant_count} participants")