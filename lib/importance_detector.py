"""
Message importance detection module for Discord monitoring.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from dataclasses import dataclass

logger = logging.getLogger('discord_bot')


@dataclass
class ImportanceResult:
    """Result of importance detection."""
    score: float
    confidence: float
    detected_patterns: List[str]
    keywords_matched: List[str]
    reasons: List[str]


class MessageImportanceDetector:
    """Detects importance of Discord messages using pattern matching and ML."""
    
    def __init__(self, patterns_config: Optional[Dict] = None):
        """Initialize the importance detector.
        
        Args:
            patterns_config: Dictionary of patterns for importance detection
        """
        self.patterns = patterns_config or self._get_default_patterns()
        self.keyword_weights = self._build_keyword_weights()
        
    def _get_default_patterns(self) -> Dict:
        """Get default importance patterns."""
        return {
            "urgent": {
                "keywords": ["urgent", "emergency", "asap", "immediately", "critical", "breaking"],
                "weight": 0.9,
                "context_keywords": ["down", "issue", "problem", "fix", "help", "needed"],
                "regex_patterns": [
                    r"\b(urgent|emergency|asap|critical)\b",
                    r"\b(server|service|system)\s+(down|offline|broken)\b",
                    r"\b(need|needs)\s+(help|assistance|support)\s+(asap|urgently|now)\b"
                ]
            },
            "group_buy": {
                "keywords": ["group buy", "gb", "interest check", "ic", "drop", "pre-order"],
                "weight": 0.8,
                "context_keywords": ["keyboard", "switches", "keycaps", "price", "shipping", "deadline"],
                "regex_patterns": [
                    r"\b(group\s*buy|gb)\b",
                    r"\b(interest\s*check|ic)\b",
                    r"\b(drop|pre-order)\s+(starting|available|live)\b"
                ]
            },
            "event": {
                "keywords": ["meeting", "event", "workshop", "conference", "deadline", "reminder"],
                "weight": 0.7,
                "context_keywords": ["tomorrow", "today", "schedule", "time", "date", "calendar"],
                "regex_patterns": [
                    r"\b(meeting|event|workshop)\s+(tomorrow|today|at)\b",
                    r"\b(deadline|due)\s+(tomorrow|today|in\s+\d+\s+days?)\b",
                    r"\b(reminder|don't\s+forget)\b"
                ]
            },
            "announcement": {
                "keywords": ["announcement", "news", "update", "release", "launch", "new"],
                "weight": 0.6,
                "context_keywords": ["version", "feature", "important", "everyone", "please"],
                "regex_patterns": [
                    r"\b(announcement|news|update)\b",
                    r"\b(new\s+version|release|launch)\b",
                    r"\b@(everyone|here)\b"
                ]
            },
            "question": {
                "keywords": ["help", "question", "how", "what", "where", "when", "why"],
                "weight": 0.4,
                "context_keywords": ["please", "anyone", "know", "explain"],
                "regex_patterns": [
                    r"\b(can\s+anyone|does\s+anyone)\s+(help|know)\b",
                    r"\b(how\s+do\s+i|what\s+is|where\s+can)\b",
                    r"\?.*\?",  # Multiple question marks
                ]
            },
            "discussion": {
                "keywords": ["think", "opinion", "thoughts", "discussion", "debate"],
                "weight": 0.3,
                "context_keywords": ["what", "do", "you", "about", "should"],
                "regex_patterns": [
                    r"\b(what\s+do\s+you\s+think|thoughts|opinions?)\b",
                    r"\b(should\s+we|let's\s+discuss)\b"
                ]
            }
        }
    
    def _build_keyword_weights(self) -> Dict[str, float]:
        """Build keyword to weight mapping from patterns."""
        weights = {}
        for pattern_name, pattern_data in self.patterns.items():
            pattern_weight = pattern_data["weight"]
            for keyword in pattern_data["keywords"]:
                weights[keyword.lower()] = max(weights.get(keyword.lower(), 0), pattern_weight)
        return weights
    
    def detect_importance(self, message_content: str, channel_name: str = "", 
                         timestamp: Optional[datetime] = None) -> ImportanceResult:
        """Detect importance of a message.
        
        Args:
            message_content: The message content to analyze
            channel_name: Name of the channel (provides context)
            timestamp: When the message was sent
            
        Returns:
            ImportanceResult with score, confidence, and details
        """
        if not message_content or not message_content.strip():
            return ImportanceResult(0.0, 1.0, [], [], ["Empty message"])
        
        content_lower = message_content.lower()
        detected_patterns = []
        keywords_matched = []
        reasons = []
        pattern_scores = []
        
        # Check each pattern
        for pattern_name, pattern_data in self.patterns.items():
            pattern_score, matched_keywords, pattern_reasons = self._check_pattern(
                content_lower, message_content, pattern_data, pattern_name
            )
            
            if pattern_score > 0:
                detected_patterns.append(pattern_name)
                keywords_matched.extend(matched_keywords)
                reasons.extend(pattern_reasons)
                pattern_scores.append(pattern_score)
        
        # Apply channel context modifiers
        channel_modifier = self._get_channel_modifier(channel_name)
        
        # Apply time-based modifiers
        time_modifier = self._get_time_modifier(timestamp)
        
        # Calculate final score
        base_score = max(pattern_scores) if pattern_scores else 0.0
        context_boost = self._calculate_context_boost(content_lower, detected_patterns)
        
        final_score = min(1.0, base_score * channel_modifier * time_modifier + context_boost)
        
        # Calculate confidence based on number of matching patterns and keywords
        confidence = self._calculate_confidence(detected_patterns, keywords_matched, message_content)
        
        # Add context reasons
        if channel_modifier != 1.0:
            reasons.append(f"Channel context modifier: {channel_modifier}")
        if time_modifier != 1.0:
            reasons.append(f"Time context modifier: {time_modifier}")
        if context_boost > 0:
            reasons.append(f"Context boost: +{context_boost:.2f}")
        
        return ImportanceResult(
            score=final_score,
            confidence=confidence,
            detected_patterns=list(set(detected_patterns)),
            keywords_matched=list(set(keywords_matched)),
            reasons=reasons
        )
    
    def _check_pattern(self, content_lower: str, original_content: str, 
                      pattern_data: Dict, pattern_name: str) -> Tuple[float, List[str], List[str]]:
        """Check if message matches a specific pattern."""
        matched_keywords = []
        reasons = []
        
        # Check keywords
        keyword_matches = 0
        for keyword in pattern_data["keywords"]:
            if keyword.lower() in content_lower:
                matched_keywords.append(keyword)
                keyword_matches += 1
        
        # Check regex patterns
        regex_matches = 0
        for regex_pattern in pattern_data.get("regex_patterns", []):
            if re.search(regex_pattern, content_lower, re.IGNORECASE):
                regex_matches += 1
        
        # Check context keywords for boost
        context_matches = 0
        for context_keyword in pattern_data.get("context_keywords", []):
            if context_keyword.lower() in content_lower:
                context_matches += 1
        
        # Calculate pattern score
        if keyword_matches > 0 or regex_matches > 0:
            base_score = pattern_data["weight"]
            
            # Boost for multiple keyword matches
            keyword_boost = min(0.1 * (keyword_matches - 1), 0.3)
            
            # Boost for regex matches
            regex_boost = min(0.05 * regex_matches, 0.2)
            
            # Boost for context matches
            context_boost = min(0.02 * context_matches, 0.1)
            
            total_score = min(1.0, base_score + keyword_boost + regex_boost + context_boost)
            
            reasons.append(f"{pattern_name} pattern (base: {base_score}, keywords: {keyword_matches}, regex: {regex_matches}, context: {context_matches})")
            
            return total_score, matched_keywords, reasons
        
        return 0.0, [], []
    
    def _get_channel_modifier(self, channel_name: str) -> float:
        """Get importance modifier based on channel name."""
        if not channel_name:
            return 1.0
        
        channel_lower = channel_name.lower()
        
        # High importance channels
        if any(word in channel_lower for word in ["announcement", "important", "critical", "urgent"]):
            return 1.3
        
        # Medium importance channels  
        if any(word in channel_lower for word in ["general", "main", "discussion", "news"]):
            return 1.1
        
        # Low importance channels
        if any(word in channel_lower for word in ["random", "off-topic", "meme", "spam", "bot"]):
            return 0.7
        
        return 1.0
    
    def _get_time_modifier(self, timestamp: Optional[datetime]) -> float:
        """Get importance modifier based on message timing."""
        if not timestamp:
            return 1.0
        
        now = datetime.now()
        age_hours = (now - timestamp).total_seconds() / 3600
        
        # Recent messages (last 2 hours) get boost
        if age_hours <= 2:
            return 1.2
        
        # Messages from last 24 hours get slight boost
        elif age_hours <= 24:
            return 1.05
        
        # Old messages get slight penalty
        elif age_hours > 168:  # 1 week
            return 0.9
        
        return 1.0
    
    def _calculate_context_boost(self, content_lower: str, detected_patterns: List[str]) -> float:
        """Calculate boost based on content context."""
        boost = 0.0
        
        # All caps boost (indicates urgency/importance)
        if re.search(r'\b[A-Z]{3,}\b', content_lower.upper()):
            boost += 0.05
        
        # Multiple exclamation marks
        if content_lower.count('!') >= 3:
            boost += 0.03
        
        # @mentions (especially @everyone, @here)
        if '@everyone' in content_lower or '@here' in content_lower:
            boost += 0.1
        elif '@' in content_lower:
            boost += 0.02
        
        # URLs (might be important links)
        if re.search(r'https?://', content_lower):
            boost += 0.02
        
        # Numbers/prices (important for group buys, deadlines)
        if re.search(r'\$\d+|\d+\s*(usd|eur|gbp|days?|hours?|minutes?)', content_lower):
            boost += 0.03
        
        # Multiple pattern matches
        if len(detected_patterns) > 1:
            boost += 0.05 * (len(detected_patterns) - 1)
        
        return min(boost, 0.3)  # Cap the boost
    
    def _calculate_confidence(self, detected_patterns: List[str], 
                            keywords_matched: List[str], message_content: str) -> float:
        """Calculate confidence in the importance score."""
        confidence = 0.5  # Base confidence
        
        # More patterns = higher confidence
        confidence += 0.1 * len(detected_patterns)
        
        # More keywords = higher confidence  
        confidence += 0.05 * len(keywords_matched)
        
        # Longer messages tend to be more reliable for analysis
        if len(message_content) > 50:
            confidence += 0.1
        elif len(message_content) < 10:
            confidence -= 0.2
        
        # Messages with clear structure (punctuation) are more reliable
        if re.search(r'[.!?]', message_content):
            confidence += 0.05
        
        return min(max(confidence, 0.1), 1.0)  # Keep between 0.1 and 1.0
    
    def get_pattern_stats(self) -> Dict:
        """Get statistics about loaded patterns."""
        stats = {
            "total_patterns": len(self.patterns),
            "total_keywords": sum(len(p["keywords"]) for p in self.patterns.values()),
            "patterns": {}
        }
        
        for name, pattern in self.patterns.items():
            stats["patterns"][name] = {
                "weight": pattern["weight"],
                "keywords_count": len(pattern["keywords"]),
                "context_keywords_count": len(pattern.get("context_keywords", [])),
                "regex_patterns_count": len(pattern.get("regex_patterns", []))
            }
        
        return stats
    
    def update_patterns(self, new_patterns: Dict):
        """Update importance detection patterns."""
        self.patterns.update(new_patterns)
        self.keyword_weights = self._build_keyword_weights()
        logger.info(f"Updated importance patterns: {list(new_patterns.keys())}")


def create_importance_detector(config_path: Optional[str] = None) -> MessageImportanceDetector:
    """Factory function to create importance detector."""
    patterns = None
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                patterns = json.load(f)
            logger.info(f"Loaded importance patterns from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load patterns from {config_path}: {e}")
    
    return MessageImportanceDetector(patterns)