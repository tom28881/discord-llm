"""
Message processing service that orchestrates the entire message processing pipeline.
This service coordinates between different components to process messages efficiently.
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

from ..domain.models import (
    Message, ProcessingResult, MonitoringEvent, EventType, 
    ImportanceLevel, MessageType
)
from ..interfaces.services import (
    IMessageProcessingService, IImportanceAnalyzer, 
    IEventService, ILLMService, INotificationService
)
from ..interfaces.repositories import IMessageRepository, IEventRepository

logger = logging.getLogger(__name__)


class MessageProcessingService(IMessageProcessingService):
    """
    Orchestrates the complete message processing pipeline.
    
    Pipeline stages:
    1. Message validation and enrichment
    2. Importance analysis
    3. Event triggering based on importance
    4. Notification sending if thresholds met
    5. Storage with processed metadata
    """
    
    def __init__(
        self,
        message_repository: IMessageRepository,
        event_repository: IEventRepository,
        importance_analyzer: IImportanceAnalyzer,
        event_service: IEventService,
        notification_service: INotificationService,
        llm_service: Optional[ILLMService] = None
    ):
        self.message_repository = message_repository
        self.event_repository = event_repository
        self.importance_analyzer = importance_analyzer
        self.event_service = event_service
        self.notification_service = notification_service
        self.llm_service = llm_service
        
        # Processing configuration
        self.notification_threshold = ImportanceLevel.HIGH
        self.batch_size = 50
        self.max_concurrent_processing = 10
    
    async def process_message(self, message: Message) -> ProcessingResult:
        """
        Process a single message through the complete pipeline.
        """
        start_time = time.time()
        processing_errors = []
        events_triggered = []
        
        try:
            logger.debug(f"Starting processing for message {message.id}")
            
            # Stage 1: Message validation and enrichment
            enriched_message = await self._enrich_message(message)
            if not enriched_message:
                return ProcessingResult(
                    message=message,
                    importance=None,
                    events_triggered=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error_message="Message validation failed"
                )
            
            # Stage 2: Importance analysis
            importance_result = await self.importance_analyzer.analyze_importance(enriched_message)
            
            # Update message with importance data
            processed_message = Message(
                id=enriched_message.id,
                server_id=enriched_message.server_id,
                channel_id=enriched_message.channel_id,
                user_id=enriched_message.user_id,
                content=enriched_message.content,
                message_type=enriched_message.message_type,
                sent_at=enriched_message.sent_at,
                importance_level=importance_result.level,
                importance_score=importance_result.score,
                keywords_matched=enriched_message.keywords_matched,
                mentions=enriched_message.mentions,
                attachments=enriched_message.attachments,
                metadata={
                    **enriched_message.metadata,
                    "importance_analysis": {
                        "reasons": importance_result.reasons,
                        "confidence": importance_result.confidence,
                        "analysis_metadata": importance_result.metadata
                    }
                },
                processed_at=datetime.utcnow()
            )
            
            # Stage 3: Event triggering
            try:
                triggered_events = await self._trigger_events(processed_message, importance_result)
                events_triggered.extend(triggered_events)
            except Exception as e:
                logger.error(f"Event triggering failed for message {message.id}: {e}")
                processing_errors.append(f"Event triggering failed: {str(e)}")
            
            # Stage 4: Notification handling
            try:
                if importance_result.level.value in [ImportanceLevel.CRITICAL.value, ImportanceLevel.HIGH.value]:
                    await self._handle_notifications(processed_message, importance_result)
            except Exception as e:
                logger.error(f"Notification handling failed for message {message.id}: {e}")
                processing_errors.append(f"Notification failed: {str(e)}")
            
            # Stage 5: Storage
            try:
                await self.message_repository.save(processed_message)
            except Exception as e:
                logger.error(f"Message storage failed for message {message.id}: {e}")
                processing_errors.append(f"Storage failed: {str(e)}")
            
            processing_time = (time.time() - start_time) * 1000
            success = len(processing_errors) == 0
            
            logger.info(
                f"Processed message {message.id} in {processing_time:.2f}ms "
                f"(importance: {importance_result.level.value}, score: {importance_result.score:.3f})"
            )
            
            return ProcessingResult(
                message=processed_message,
                importance=importance_result,
                events_triggered=events_triggered,
                processing_time_ms=processing_time,
                success=success,
                error_message="; ".join(processing_errors) if processing_errors else None
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Message processing completely failed for {message.id}: {e}")
            
            return ProcessingResult(
                message=message,
                importance=None,
                events_triggered=events_triggered,
                processing_time_ms=processing_time,
                success=False,
                error_message=f"Processing failed: {str(e)}"
            )
    
    async def process_message_batch(self, messages: List[Message]) -> List[ProcessingResult]:
        """
        Process multiple messages efficiently using concurrent processing.
        """
        if not messages:
            return []
        
        logger.info(f"Processing batch of {len(messages)} messages")
        
        # Split into smaller batches to avoid overwhelming the system
        results = []
        
        for i in range(0, len(messages), self.batch_size):
            batch = messages[i:i + self.batch_size]
            
            # Process batch concurrently with limited concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent_processing)
            
            async def process_with_semaphore(message):
                async with semaphore:
                    return await self.process_message(message)
            
            batch_results = await asyncio.gather(
                *[process_with_semaphore(msg) for msg in batch],
                return_exceptions=True
            )
            
            # Handle any exceptions from concurrent processing
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed for message {batch[i].id}: {result}")
                    results.append(ProcessingResult(
                        message=batch[i],
                        importance=None,
                        events_triggered=[],
                        processing_time_ms=0,
                        success=False,
                        error_message=f"Concurrent processing failed: {str(result)}"
                    ))
                else:
                    results.append(result)
        
        successful_count = sum(1 for r in results if r.success)
        logger.info(f"Batch processing complete: {successful_count}/{len(results)} successful")
        
        return results
    
    async def reprocess_message(self, message_id: str) -> ProcessingResult:
        """
        Reprocess an existing message with current rules and configuration.
        """
        try:
            # Retrieve the original message
            message = await self.message_repository.get_by_id(message_id)
            if not message:
                return ProcessingResult(
                    message=None,
                    importance=None,
                    events_triggered=[],
                    processing_time_ms=0,
                    success=False,
                    error_message=f"Message {message_id} not found"
                )
            
            # Reset processing metadata for reprocessing
            reset_message = Message(
                id=message.id,
                server_id=message.server_id,
                channel_id=message.channel_id,
                user_id=message.user_id,
                content=message.content,
                message_type=message.message_type,
                sent_at=message.sent_at,
                importance_level=ImportanceLevel.MEDIUM,  # Reset to default
                importance_score=0.5,  # Reset to default
                keywords_matched=[],  # Will be recalculated
                mentions=message.mentions,
                attachments=message.attachments,
                metadata={k: v for k, v in message.metadata.items() if not k.startswith('importance_')},
                processed_at=None  # Mark as unprocessed
            )
            
            # Reprocess with current configuration
            result = await self.process_message(reset_message)
            
            logger.info(f"Reprocessed message {message_id} with new importance: {result.importance.level.value if result.importance else 'unknown'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to reprocess message {message_id}: {e}")
            return ProcessingResult(
                message=None,
                importance=None,
                events_triggered=[],
                processing_time_ms=0,
                success=False,
                error_message=f"Reprocessing failed: {str(e)}"
            )
    
    async def _enrich_message(self, message: Message) -> Optional[Message]:
        """
        Enrich message with additional metadata and validation.
        """
        try:
            # Extract mentions from content
            mentions = self._extract_mentions(message.content)
            
            # Extract keywords that might match our importance rules
            keywords_matched = await self._extract_matching_keywords(message.content)
            
            # Determine message type based on content
            message_type = self._determine_message_type(message)
            
            # Add enrichment metadata
            enrichment_metadata = {
                "enriched_at": datetime.utcnow().isoformat(),
                "content_length": len(message.content),
                "word_count": len(message.content.split()),
                "has_urls": bool(self._extract_urls(message.content)),
                "has_code_blocks": "```" in message.content,
                "has_emojis": self._has_emojis(message.content)
            }
            
            return Message(
                id=message.id,
                server_id=message.server_id,
                channel_id=message.channel_id,
                user_id=message.user_id,
                content=message.content,
                message_type=message_type,
                sent_at=message.sent_at,
                importance_level=message.importance_level,
                importance_score=message.importance_score,
                keywords_matched=keywords_matched,
                mentions=mentions,
                attachments=message.attachments,
                metadata={**message.metadata, **enrichment_metadata},
                processed_at=message.processed_at
            )
            
        except Exception as e:
            logger.error(f"Failed to enrich message {message.id}: {e}")
            return None
    
    def _extract_mentions(self, content: str) -> List[str]:
        """Extract user and role mentions from message content."""
        import re
        
        mentions = []
        
        # User mentions: <@!123456789> or <@123456789>
        user_mentions = re.findall(r'<@!?(\d+)>', content)
        mentions.extend([f"user:{uid}" for uid in user_mentions])
        
        # Role mentions: <@&123456789>
        role_mentions = re.findall(r'<@&(\d+)>', content)
        mentions.extend([f"role:{rid}" for rid in role_mentions])
        
        # Channel mentions: <#123456789>
        channel_mentions = re.findall(r'<#(\d+)>', content)
        mentions.extend([f"channel:{cid}" for cid in channel_mentions])
        
        return mentions
    
    async def _extract_matching_keywords(self, content: str) -> List[str]:
        """Extract keywords from content that match our importance rules."""
        # This would typically load keyword rules from configuration
        # For now, using a simple approach
        important_keywords = [
            "urgent", "important", "critical", "help", "error", "bug", 
            "issue", "problem", "alert", "warning", "asap", "emergency"
        ]
        
        content_lower = content.lower()
        matched = [keyword for keyword in important_keywords if keyword in content_lower]
        
        return matched
    
    def _determine_message_type(self, message: Message) -> MessageType:
        """Determine message type based on content and metadata."""
        if not message.content.strip():
            if message.attachments:
                return MessageType.ATTACHMENT
            else:
                return MessageType.SYSTEM
        elif message.content.startswith("```") or "```" in message.content:
            return MessageType.TEXT  # Code blocks are still text
        else:
            return MessageType.TEXT
    
    def _extract_urls(self, content: str) -> List[str]:
        """Extract URLs from message content."""
        import re
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        return re.findall(url_pattern, content)
    
    def _has_emojis(self, content: str) -> bool:
        """Check if content contains emojis (basic check)."""
        # Discord custom emojis: <:name:id> or <a:name:id>
        import re
        return bool(re.search(r'<a?:\w+:\d+>', content))
    
    async def _trigger_events(self, message: Message, importance_result) -> List[MonitoringEvent]:
        """Trigger appropriate events based on message importance and content."""
        events = []
        
        try:
            # Always trigger message received event
            message_event = MonitoringEvent(
                event_type=EventType.MESSAGE_RECEIVED,
                message=message,
                trigger_rules=["message_received"],
                metadata={
                    "importance_score": importance_result.score,
                    "importance_level": importance_result.level.value
                }
            )
            
            await self.event_service.trigger_event(message_event)
            events.append(message_event)
            
            # Trigger importance threshold event if needed
            if importance_result.level in [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH]:
                threshold_event = MonitoringEvent(
                    event_type=EventType.IMPORTANCE_THRESHOLD_EXCEEDED,
                    message=message,
                    trigger_rules=[f"importance_threshold_{importance_result.level.value}"],
                    metadata={
                        "threshold_exceeded": importance_result.level.value,
                        "score": importance_result.score,
                        "reasons": importance_result.reasons
                    }
                )
                
                await self.event_service.trigger_event(threshold_event)
                events.append(threshold_event)
            
            # Trigger keyword events
            if message.keywords_matched:
                keyword_event = MonitoringEvent(
                    event_type=EventType.KEYWORD_TRIGGERED,
                    message=message,
                    trigger_rules=[f"keyword_{kw}" for kw in message.keywords_matched],
                    metadata={
                        "keywords": message.keywords_matched
                    }
                )
                
                await self.event_service.trigger_event(keyword_event)
                events.append(keyword_event)
            
            # Trigger mention events
            if message.mentions:
                mention_event = MonitoringEvent(
                    event_type=EventType.USER_MENTIONED,
                    message=message,
                    trigger_rules=["user_mentioned"],
                    metadata={
                        "mentions": message.mentions
                    }
                )
                
                await self.event_service.trigger_event(mention_event)
                events.append(mention_event)
            
        except Exception as e:
            logger.error(f"Failed to trigger events for message {message.id}: {e}")
        
        return events
    
    async def _handle_notifications(self, message: Message, importance_result) -> None:
        """Handle sending notifications for important messages."""
        try:
            # Create notification message
            notification_text = self._format_notification(message, importance_result)
            
            # Send notification
            await self.notification_service.send_notification(
                message=notification_text,
                importance_level=importance_result.level,
                metadata={
                    "message_id": message.id,
                    "server_id": message.server_id,
                    "channel_id": message.channel_id,
                    "importance_score": importance_result.score
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send notification for message {message.id}: {e}")
    
    def _format_notification(self, message: Message, importance_result) -> str:
        """Format a notification message."""
        content_preview = message.content[:100] + "..." if len(message.content) > 100 else message.content
        
        return (
            f"🔔 {importance_result.level.value.upper()} message detected\n"
            f"Score: {importance_result.score:.2f}\n"
            f"Content: {content_preview}\n"
            f"Reasons: {', '.join(importance_result.reasons)}"
        )