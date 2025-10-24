"""
Message importance analysis service using strategy pattern.
This service analyzes messages and assigns importance scores based on configurable rules.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

from ..domain.models import Message, ImportanceResult, ImportanceLevel
from ..interfaces.services import IImportanceAnalyzer
from ..interfaces.repositories import IConfigRepository

logger = logging.getLogger(__name__)


class ImportanceStrategy(ABC):
    """Abstract base class for importance scoring strategies."""
    
    @abstractmethod
    async def calculate_score(self, message: Message, context: Dict[str, Any]) -> float:
        """Calculate importance score for a message."""
        pass
    
    @abstractmethod
    def get_reasons(self) -> List[str]:
        """Get list of reasons why this strategy was applied."""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the strategy."""
        pass


class KeywordStrategy(ImportanceStrategy):
    """Strategy based on keyword matching."""
    
    def __init__(self, keywords: Dict[str, float], case_sensitive: bool = False):
        self.keywords = keywords  # keyword -> weight mapping
        self.case_sensitive = case_sensitive
        self.matched_keywords: List[str] = []
    
    async def calculate_score(self, message: Message, context: Dict[str, Any]) -> float:
        content = message.content if self.case_sensitive else message.content.lower()
        score = 0.0
        self.matched_keywords = []
        
        for keyword, weight in self.keywords.items():
            search_keyword = keyword if self.case_sensitive else keyword.lower()
            if search_keyword in content:
                score += weight
                self.matched_keywords.append(keyword)
        
        # Normalize score to 0-1 range
        return min(score, 1.0)
    
    def get_reasons(self) -> List[str]:
        if self.matched_keywords:
            return [f"Matched keywords: {', '.join(self.matched_keywords)}"]
        return []
    
    @property
    def strategy_name(self) -> str:
        return "keyword_matching"


class MentionStrategy(ImportanceStrategy):
    """Strategy based on user mentions and role mentions."""
    
    def __init__(self, mention_weight: float = 0.3, role_mention_weight: float = 0.5):
        self.mention_weight = mention_weight
        self.role_mention_weight = role_mention_weight
        self.mention_count = 0
        self.role_mention_count = 0
    
    async def calculate_score(self, message: Message, context: Dict[str, Any]) -> float:
        content = message.content
        
        # Count user mentions (@user)
        user_mentions = re.findall(r'<@!?(\d+)>', content)
        self.mention_count = len(user_mentions)
        
        # Count role mentions (@role)
        role_mentions = re.findall(r'<@&(\d+)>', content)
        self.role_mention_count = len(role_mentions)
        
        score = (self.mention_count * self.mention_weight + 
                self.role_mention_count * self.role_mention_weight)
        
        return min(score, 1.0)
    
    def get_reasons(self) -> List[str]:
        reasons = []
        if self.mention_count > 0:
            reasons.append(f"{self.mention_count} user mention(s)")
        if self.role_mention_count > 0:
            reasons.append(f"{self.role_mention_count} role mention(s)")
        return reasons
    
    @property
    def strategy_name(self) -> str:
        return "mention_analysis"


class LengthStrategy(ImportanceStrategy):
    """Strategy based on message length."""
    
    def __init__(self, min_length: int = 50, max_score_length: int = 500):
        self.min_length = min_length
        self.max_score_length = max_score_length
        self.message_length = 0
    
    async def calculate_score(self, message: Message, context: Dict[str, Any]) -> float:
        self.message_length = len(message.content.strip())
        
        if self.message_length < self.min_length:
            return 0.0
        
        # Linear scaling from min_length to max_score_length
        score = min((self.message_length - self.min_length) / 
                   (self.max_score_length - self.min_length), 1.0)
        
        return score * 0.2  # Weight length lower than other factors
    
    def get_reasons(self) -> List[str]:
        if self.message_length >= self.min_length:
            return [f"Message length: {self.message_length} characters"]
        return []
    
    @property
    def strategy_name(self) -> str:
        return "message_length"


class TimeBasedStrategy(ImportanceStrategy):
    """Strategy based on message timing (recent messages are more important)."""
    
    def __init__(self, decay_hours: int = 24):
        self.decay_hours = decay_hours
        self.age_hours = 0.0
    
    async def calculate_score(self, message: Message, context: Dict[str, Any]) -> float:
        now = datetime.utcnow()
        age = now - message.sent_at
        self.age_hours = age.total_seconds() / 3600
        
        if self.age_hours > self.decay_hours:
            return 0.1  # Minimum score for old messages
        
        # Exponential decay
        decay_factor = self.age_hours / self.decay_hours
        score = 1.0 - (decay_factor ** 2)
        
        return max(score, 0.1)
    
    def get_reasons(self) -> List[str]:
        if self.age_hours < 1:
            return ["Recent message (< 1 hour)"]
        elif self.age_hours < 6:
            return [f"Recent message ({self.age_hours:.1f} hours ago)"]
        return []
    
    @property
    def strategy_name(self) -> str:
        return "time_based"


class ChannelStrategy(ImportanceStrategy):
    """Strategy based on channel importance."""
    
    def __init__(self, channel_weights: Dict[str, float]):
        self.channel_weights = channel_weights  # channel_id -> weight
        self.channel_weight = 0.5
    
    async def calculate_score(self, message: Message, context: Dict[str, Any]) -> float:
        self.channel_weight = self.channel_weights.get(message.channel_id, 0.5)
        return self.channel_weight
    
    def get_reasons(self) -> List[str]:
        if self.channel_weight > 0.7:
            return ["High-priority channel"]
        elif self.channel_weight > 0.5:
            return ["Medium-priority channel"]
        return []
    
    @property
    def strategy_name(self) -> str:
        return "channel_priority"


class ImportanceAnalyzerService(IImportanceAnalyzer):
    """
    Main importance analyzer service that orchestrates multiple strategies.
    """
    
    def __init__(self, config_repository: IConfigRepository):
        self.config_repository = config_repository
        self.strategies: List[ImportanceStrategy] = []
        self._initialize_strategies()
    
    async def _initialize_strategies(self) -> None:
        """Initialize default strategies with configuration."""
        try:
            # Load configuration from repository
            keyword_config = await self.config_repository.get_config("importance.keywords", {
                "urgent": 0.8,
                "important": 0.6,
                "critical": 0.9,
                "help": 0.4,
                "error": 0.7,
                "bug": 0.6,
                "issue": 0.5,
                "problem": 0.5,
                "alert": 0.8,
                "warning": 0.6
            })
            
            channel_weights = await self.config_repository.get_config("importance.channel_weights", {})
            
            # Initialize strategies
            self.strategies = [
                KeywordStrategy(keyword_config),
                MentionStrategy(),
                LengthStrategy(),
                TimeBasedStrategy(),
                ChannelStrategy(channel_weights)
            ]
            
            logger.info(f"Initialized {len(self.strategies)} importance strategies")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies: {e}")
            # Fallback to default strategies
            self.strategies = [
                KeywordStrategy({"urgent": 0.8, "important": 0.6}),
                MentionStrategy(),
                LengthStrategy(),
                TimeBasedStrategy()
            ]
    
    async def analyze_importance(self, message: Message) -> ImportanceResult:
        """
        Analyze message importance using all configured strategies.
        """
        try:
            context = {
                "timestamp": datetime.utcnow(),
                "message_id": message.id
            }
            
            scores: List[float] = []
            all_reasons: List[str] = []
            strategy_results: Dict[str, float] = {}
            
            # Run all strategies
            for strategy in self.strategies:
                try:
                    score = await strategy.calculate_score(message, context)
                    scores.append(score)
                    strategy_results[strategy.strategy_name] = score
                    
                    reasons = strategy.get_reasons()
                    all_reasons.extend(reasons)
                    
                except Exception as e:
                    logger.warning(f"Strategy {strategy.strategy_name} failed: {e}")
                    scores.append(0.0)
            
            # Calculate weighted final score
            if scores:
                # Use weighted average with emphasis on higher scores
                sorted_scores = sorted(scores, reverse=True)
                weights = [0.4, 0.3, 0.2, 0.1]  # Give more weight to top strategies
                
                final_score = sum(
                    score * weight 
                    for score, weight in zip(sorted_scores, weights[:len(sorted_scores)])
                )
                final_score = min(final_score, 1.0)
            else:
                final_score = 0.5  # Default score
            
            # Determine importance level
            importance_level = self._score_to_level(final_score)
            
            # Calculate confidence based on strategy agreement
            confidence = self._calculate_confidence(scores)
            
            return ImportanceResult(
                score=final_score,
                level=importance_level,
                reasons=all_reasons,
                confidence=confidence,
                metadata={
                    "strategy_scores": strategy_results,
                    "raw_scores": scores,
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze importance for message {message.id}: {e}")
            return ImportanceResult(
                score=0.5,
                level=ImportanceLevel.MEDIUM,
                reasons=[f"Analysis failed: {str(e)}"],
                confidence=0.0
            )
    
    def _score_to_level(self, score: float) -> ImportanceLevel:
        """Convert numeric score to importance level."""
        if score >= 0.9:
            return ImportanceLevel.CRITICAL
        elif score >= 0.7:
            return ImportanceLevel.HIGH
        elif score >= 0.4:
            return ImportanceLevel.MEDIUM
        elif score >= 0.2:
            return ImportanceLevel.LOW
        else:
            return ImportanceLevel.NOISE
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on how much strategies agree."""
        if not scores or len(scores) < 2:
            return 0.5
        
        # Calculate variance - lower variance means higher confidence
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Convert variance to confidence (inverse relationship)
        confidence = max(0.0, 1.0 - (variance * 4))  # Scale variance appropriately
        return min(confidence, 1.0)
    
    async def update_scoring_rules(self, rules: Dict[str, Any]) -> bool:
        """Update scoring rules and reinitialize strategies."""
        try:
            # Save new rules to configuration
            for key, value in rules.items():
                await self.config_repository.set_config(f"importance.{key}", value)
            
            # Reinitialize strategies with new config
            await self._initialize_strategies()
            
            logger.info("Successfully updated importance scoring rules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update scoring rules: {e}")
            return False
    
    async def get_scoring_explanation(self, message: Message) -> Dict[str, Any]:
        """Get detailed explanation of importance scoring."""
        try:
            result = await self.analyze_importance(message)
            
            explanation = {
                "message_id": message.id,
                "final_score": result.score,
                "importance_level": result.level.value,
                "confidence": result.confidence,
                "reasons": result.reasons,
                "strategy_breakdown": result.metadata.get("strategy_scores", {}),
                "scoring_factors": {
                    "content_length": len(message.content),
                    "mentions_found": len(message.mentions),
                    "keywords_matched": message.keywords_matched,
                    "message_age_minutes": (datetime.utcnow() - message.sent_at).total_seconds() / 60
                }
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to get scoring explanation: {e}")
            return {"error": str(e)}