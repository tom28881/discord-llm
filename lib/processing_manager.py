"""
Real-time processing manager with comprehensive error handling and recovery.

Manages message processing queues, handles pipeline failures, detects deadlocks,
prevents memory leaks, and ensures continuous operation for 24/7 reliability.
"""

import threading
import queue
import time
import logging
import gc
import psutil
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from collections import deque, defaultdict
import weakref

from .exceptions import (
    ProcessingError, QueueOverflowError, MemoryError as CustomMemoryError,
    DeadlockError, DiscordMonitorException
)
from .monitoring import get_monitoring_system


class ProcessingStatus(Enum):
    """Processing pipeline status."""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ProcessingTask:
    """Container for processing tasks."""
    id: str
    data: Any
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    retries: int = 0
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes default
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProcessingStats:
    """Statistics for processing pipeline."""
    tasks_processed: int = 0
    tasks_failed: int = 0
    tasks_retried: int = 0
    average_processing_time: float = 0.0
    queue_size: int = 0
    active_workers: int = 0
    memory_usage_mb: float = 0.0
    uptime_seconds: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


class DeadlockDetector:
    """Detects and resolves deadlock situations."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.task_states = {}  # task_id -> {'worker_id': str, 'start_time': datetime}
        self.worker_states = {}  # worker_id -> {'task_id': str, 'start_time': datetime}
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.logger = logging.getLogger(__name__ + '.deadlock')
    
    def start(self):
        """Start deadlock detection."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        self.logger.info("Deadlock detector started")
    
    def stop(self):
        """Stop deadlock detection."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def register_task_start(self, task_id: str, worker_id: str):
        """Register that a task has started processing."""
        with self.lock:
            self.task_states[task_id] = {
                'worker_id': worker_id,
                'start_time': datetime.now()
            }
            self.worker_states[worker_id] = {
                'task_id': task_id,
                'start_time': datetime.now()
            }
    
    def register_task_complete(self, task_id: str, worker_id: str):
        """Register that a task has completed."""
        with self.lock:
            self.task_states.pop(task_id, None)
            self.worker_states.pop(worker_id, None)
    
    def _detection_loop(self):
        """Main deadlock detection loop."""
        while self.running:
            try:
                self._check_for_deadlocks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Deadlock detection error: {e}")
                time.sleep(5)
    
    def _check_for_deadlocks(self):
        """Check for potential deadlock situations."""
        current_time = datetime.now()
        stalled_tasks = []
        
        with self.lock:
            for task_id, state in self.task_states.items():
                # Task running for more than 10 minutes is considered stalled
                if (current_time - state['start_time']).total_seconds() > 600:
                    stalled_tasks.append({
                        'task_id': task_id,
                        'worker_id': state['worker_id'],
                        'duration': current_time - state['start_time']
                    })
        
        if stalled_tasks:
            self.logger.warning(f"Detected {len(stalled_tasks)} stalled tasks")
            
            # Group by worker to identify potential deadlocks
            worker_groups = defaultdict(list)
            for task in stalled_tasks:
                worker_groups[task['worker_id']].append(task)
            
            # Check for circular dependencies (simplified detection)
            if len(worker_groups) > 1:
                worker_ids = list(worker_groups.keys())
                raise DeadlockError(
                    f"Potential deadlock detected involving {len(worker_groups)} workers",
                    involved_components=worker_ids
                )


class MemoryManager:
    """Monitors and manages memory usage."""
    
    def __init__(self, max_memory_mb: float = 1024, check_interval: int = 60):
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.logger = logging.getLogger(__name__ + '.memory')
        self.cleanup_callbacks = []
    
    def start(self):
        """Start memory monitoring."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        self.logger.info("Memory manager started")
    
    def stop(self):
        """Stop memory monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def add_cleanup_callback(self, callback: Callable[[], None]):
        """Add callback to be called during memory cleanup."""
        self.cleanup_callbacks.append(callback)
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _monitoring_loop(self):
        """Memory monitoring loop."""
        while self.running:
            try:
                current_usage = self.get_memory_usage_mb()
                
                if current_usage > self.max_memory_mb:
                    self.logger.warning(f"High memory usage: {current_usage:.1f}MB > {self.max_memory_mb}MB")
                    self._trigger_cleanup()
                
                # Record metrics
                monitoring = get_monitoring_system()
                monitoring.metrics.gauge("process.memory_manager.usage_mb", current_usage)
                monitoring.metrics.gauge("process.memory_manager.limit_mb", self.max_memory_mb)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(10)
    
    def _trigger_cleanup(self):
        """Trigger memory cleanup procedures."""
        initial_usage = self.get_memory_usage_mb()
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        collected = gc.collect()
        
        final_usage = self.get_memory_usage_mb()
        freed_mb = initial_usage - final_usage
        
        self.logger.info(
            f"Memory cleanup: {initial_usage:.1f}MB -> {final_usage:.1f}MB "
            f"(freed {freed_mb:.1f}MB, GC collected {collected} objects)"
        )
        
        # If still over limit, raise error
        if final_usage > self.max_memory_mb:
            raise CustomMemoryError(
                f"Memory usage still high after cleanup: {final_usage:.1f}MB",
                current_usage=final_usage,
                max_usage=self.max_memory_mb
            )


class ProcessingPipeline:
    """Processing pipeline with error handling and recovery."""
    
    def __init__(self, name: str, processor_func: Callable[[ProcessingTask], Any],
                 max_queue_size: int = 10000, worker_count: int = 4):
        self.name = name
        self.processor_func = processor_func
        self.max_queue_size = max_queue_size
        self.worker_count = worker_count
        
        # Queue and workers
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"{name}-worker")
        
        # State management
        self.status = ProcessingStatus.IDLE
        self.stats = ProcessingStats()
        self.start_time = datetime.now()
        
        # Error handling
        self.active_tasks = {}  # Future -> ProcessingTask
        self.failed_tasks = deque(maxlen=1000)  # Keep recent failures
        
        # Components
        self.deadlock_detector = DeadlockDetector()
        self.memory_manager = MemoryManager()
        
        # Control
        self.running = False
        self.shutdown_event = threading.Event()
        self.stats_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__ + f'.pipeline.{name}')
        
        # Monitoring
        self.monitoring = get_monitoring_system()
        
        # Setup memory cleanup
        self.memory_manager.add_cleanup_callback(self._cleanup_completed_tasks)
    
    def start(self):
        """Start the processing pipeline."""
        if self.running:
            return
        
        self.running = True
        self.status = ProcessingStatus.RUNNING
        self.start_time = datetime.now()
        
        # Start components
        self.deadlock_detector.start()
        self.memory_manager.start()
        
        # Start processing loop
        self._start_processing_loop()
        
        self.logger.info(f"Processing pipeline '{self.name}' started with {self.worker_count} workers")
    
    def stop(self, timeout: float = 30.0):
        """Stop the processing pipeline."""
        if not self.running:
            return
        
        self.logger.info(f"Stopping processing pipeline '{self.name}'...")
        self.status = ProcessingStatus.SHUTTING_DOWN
        self.running = False
        self.shutdown_event.set()
        
        # Stop accepting new tasks
        # (Current tasks will finish)
        
        # Wait for completion or timeout
        self.executor.shutdown(wait=True, timeout=timeout)
        
        # Stop components
        self.deadlock_detector.stop()
        self.memory_manager.stop()
        
        self.status = ProcessingStatus.IDLE
        self.logger.info(f"Processing pipeline '{self.name}' stopped")
    
    def submit_task(self, task: ProcessingTask) -> bool:
        """Submit a task for processing."""
        if not self.running or self.status != ProcessingStatus.RUNNING:
            raise ProcessingError(f"Pipeline '{self.name}' not running")
        
        try:
            # Check queue size
            if self.task_queue.qsize() >= self.max_queue_size:
                raise QueueOverflowError(
                    f"Task queue full for pipeline '{self.name}': {self.task_queue.qsize()}/{self.max_queue_size}"
                )
            
            # Add to queue (priority queue uses (priority, item) tuples)
            queue_item = (-task.priority, task.created_at, task)  # Negative for max-heap behavior
            self.task_queue.put(queue_item, timeout=1.0)
            
            self.monitoring.metrics.increment(f"pipeline.{self.name}.tasks_queued")
            return True
            
        except queue.Full:
            raise QueueOverflowError(f"Task queue timeout for pipeline '{self.name}'")
    
    def _start_processing_loop(self):
        """Start the main processing loop."""
        def processing_loop():
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Get task from queue
                    try:
                        priority, timestamp, task = self.task_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    
                    # Submit to thread pool
                    future = self.executor.submit(self._process_task_wrapper, task)
                    self.active_tasks[future] = task
                    
                    # Update stats
                    with self.stats_lock:
                        self.stats.queue_size = self.task_queue.qsize()
                        self.stats.active_workers = len(self.active_tasks)
                    
                    self.task_queue.task_done()
                    
                except Exception as e:
                    self.logger.error(f"Processing loop error: {e}")
                    time.sleep(1)
        
        # Start processing thread
        threading.Thread(target=processing_loop, daemon=True, name=f"{self.name}-processor").start()
        
        # Start result collection thread
        threading.Thread(target=self._collect_results, daemon=True, name=f"{self.name}-collector").start()
    
    def _process_task_wrapper(self, task: ProcessingTask) -> Tuple[ProcessingTask, Any, Optional[Exception]]:
        """Wrapper for task processing with error handling."""
        worker_id = threading.current_thread().name
        start_time = time.time()
        
        try:
            # Register with deadlock detector
            self.deadlock_detector.register_task_start(task.id, worker_id)
            
            # Process the task
            result = self.processor_func(task)
            
            # Success
            processing_time = time.time() - start_time
            self.monitoring.metrics.timing(f"pipeline.{self.name}.task_duration", processing_time)
            
            return (task, result, None)
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Task {task.id} failed after {processing_time:.2f}s: {e}")
            
            # Record error metrics
            self.monitoring.metrics.increment(f"pipeline.{self.name}.task_errors", {"error_type": type(e).__name__})
            
            return (task, None, e)
            
        finally:
            # Unregister from deadlock detector
            self.deadlock_detector.register_task_complete(task.id, worker_id)
    
    def _collect_results(self):
        """Collect results from completed tasks."""
        while self.running or self.active_tasks:
            try:
                # Check completed futures
                completed_futures = []
                for future in list(self.active_tasks.keys()):
                    if future.done():
                        completed_futures.append(future)
                
                # Process completed tasks
                for future in completed_futures:
                    task = self.active_tasks.pop(future)
                    
                    try:
                        task_result, result, error = future.result()
                        
                        if error is None:
                            # Success
                            self._handle_task_success(task, result)
                        else:
                            # Error - decide whether to retry
                            self._handle_task_error(task, error)
                            
                    except Exception as e:
                        self.logger.error(f"Error collecting result for task {task.id}: {e}")
                        self._handle_task_error(task, e)
                
                time.sleep(0.1)  # Small delay to prevent busy-waiting
                
            except Exception as e:
                self.logger.error(f"Result collection error: {e}")
                time.sleep(1)
    
    def _handle_task_success(self, task: ProcessingTask, result: Any):
        """Handle successful task completion."""
        with self.stats_lock:
            self.stats.tasks_processed += 1
            
            # Update average processing time
            total_time = self.stats.average_processing_time * (self.stats.tasks_processed - 1)
            processing_time = (datetime.now() - task.created_at).total_seconds()
            self.stats.average_processing_time = (total_time + processing_time) / self.stats.tasks_processed
        
        self.monitoring.metrics.increment(f"pipeline.{self.name}.tasks_completed")
        self.result_queue.put((task, result))
    
    def _handle_task_error(self, task: ProcessingTask, error: Exception):
        """Handle task processing error."""
        task.retries += 1
        
        with self.stats_lock:
            self.stats.tasks_failed += 1
            self.stats.last_error = str(error)
            self.stats.last_error_time = datetime.now()
        
        # Decide whether to retry
        if task.retries <= task.max_retries and self._should_retry_task(task, error):
            self.logger.info(f"Retrying task {task.id} (attempt {task.retries}/{task.max_retries})")
            
            # Add delay before retry
            time.sleep(min(2 ** task.retries, 30))  # Exponential backoff, max 30s
            
            try:
                self.submit_task(task)
                with self.stats_lock:
                    self.stats.tasks_retried += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to requeue task {task.id}: {e}")
                self._handle_task_final_failure(task, error)
        else:
            self._handle_task_final_failure(task, error)
    
    def _should_retry_task(self, task: ProcessingTask, error: Exception) -> bool:
        """Determine if a task should be retried based on the error type."""
        # Don't retry certain types of errors
        non_retryable_errors = (
            ValueError,  # Invalid data
            TypeError,   # Programming errors
            KeyError,    # Missing data
        )
        
        if isinstance(error, non_retryable_errors):
            return False
        
        # Don't retry if it's a memory error
        if isinstance(error, CustomMemoryError):
            return False
        
        return True
    
    def _handle_task_final_failure(self, task: ProcessingTask, error: Exception):
        """Handle final task failure (no more retries)."""
        self.failed_tasks.append({
            'task_id': task.id,
            'error': str(error),
            'error_type': type(error).__name__,
            'retries': task.retries,
            'final_failure_time': datetime.now().isoformat()
        })
        
        self.logger.error(f"Task {task.id} failed permanently after {task.retries} retries: {error}")
        self.monitoring.metrics.increment(f"pipeline.{self.name}.tasks_failed_permanently")
    
    def _cleanup_completed_tasks(self):
        """Cleanup completed tasks to free memory."""
        # Remove completed futures
        completed_count = 0
        for future in list(self.active_tasks.keys()):
            if future.done():
                self.active_tasks.pop(future, None)
                completed_count += 1
        
        if completed_count > 0:
            self.logger.debug(f"Cleaned up {completed_count} completed tasks")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        with self.stats_lock:
            stats = self.stats
            stats.uptime_seconds = uptime
            stats.memory_usage_mb = self.memory_manager.get_memory_usage_mb()
            stats.queue_size = self.task_queue.qsize()
            stats.active_workers = len(self.active_tasks)
        
        return {
            'name': self.name,
            'status': self.status.value,
            'running': self.running,
            'stats': stats.__dict__,
            'failed_tasks_recent': list(self.failed_tasks),
            'worker_count': self.worker_count,
            'max_queue_size': self.max_queue_size
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information."""
        status = self.get_status()
        
        # Determine health
        healthy = (
            self.running and
            self.status == ProcessingStatus.RUNNING and
            len(self.active_tasks) < self.worker_count * 2 and  # Not too many active tasks
            self.task_queue.qsize() < self.max_queue_size * 0.8  # Queue not too full
        )
        
        return {
            'healthy': healthy,
            'status': status,
            'issues': self._identify_issues()
        }
    
    def _identify_issues(self) -> List[str]:
        """Identify current issues with the pipeline."""
        issues = []
        
        if not self.running:
            issues.append("Pipeline not running")
        
        if self.status != ProcessingStatus.RUNNING:
            issues.append(f"Pipeline status: {self.status.value}")
        
        if self.task_queue.qsize() > self.max_queue_size * 0.9:
            issues.append("Task queue nearly full")
        
        if len(self.active_tasks) == 0 and self.task_queue.qsize() > 0:
            issues.append("No active workers but tasks queued")
        
        with self.stats_lock:
            if self.stats.last_error_time:
                time_since_error = (datetime.now() - self.stats.last_error_time).total_seconds()
                if time_since_error < 300:  # Error in last 5 minutes
                    issues.append(f"Recent error: {self.stats.last_error}")
        
        return issues


class ProcessingManager:
    """Manages multiple processing pipelines."""
    
    def __init__(self):
        self.pipelines = {}
        self.global_stats = {
            'pipelines_count': 0,
            'total_tasks_processed': 0,
            'total_tasks_failed': 0,
            'start_time': datetime.now()
        }
        self.logger = logging.getLogger(__name__ + '.manager')
        self.monitoring = get_monitoring_system()
    
    def add_pipeline(self, pipeline: ProcessingPipeline) -> bool:
        """Add a processing pipeline."""
        if pipeline.name in self.pipelines:
            self.logger.warning(f"Pipeline '{pipeline.name}' already exists")
            return False
        
        self.pipelines[pipeline.name] = pipeline
        self.global_stats['pipelines_count'] += 1
        
        self.logger.info(f"Added pipeline: {pipeline.name}")
        return True
    
    def remove_pipeline(self, name: str) -> bool:
        """Remove and stop a processing pipeline."""
        if name not in self.pipelines:
            return False
        
        pipeline = self.pipelines.pop(name)
        pipeline.stop()
        
        self.global_stats['pipelines_count'] -= 1
        self.logger.info(f"Removed pipeline: {name}")
        return True
    
    def start_all(self):
        """Start all pipelines."""
        for pipeline in self.pipelines.values():
            try:
                pipeline.start()
            except Exception as e:
                self.logger.error(f"Failed to start pipeline {pipeline.name}: {e}")
    
    def stop_all(self, timeout: float = 30.0):
        """Stop all pipelines."""
        for pipeline in self.pipelines.values():
            try:
                pipeline.stop(timeout)
            except Exception as e:
                self.logger.error(f"Failed to stop pipeline {pipeline.name}: {e}")
    
    def get_pipeline(self, name: str) -> Optional[ProcessingPipeline]:
        """Get a pipeline by name."""
        return self.pipelines.get(name)
    
    def submit_task(self, pipeline_name: str, task: ProcessingTask) -> bool:
        """Submit a task to a specific pipeline."""
        pipeline = self.pipelines.get(pipeline_name)
        if not pipeline:
            raise ProcessingError(f"Pipeline '{pipeline_name}' not found")
        
        return pipeline.submit_task(task)
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get global status across all pipelines."""
        total_processed = 0
        total_failed = 0
        total_queue_size = 0
        total_active_workers = 0
        
        pipeline_statuses = {}
        for name, pipeline in self.pipelines.items():
            status = pipeline.get_status()
            pipeline_statuses[name] = status
            
            total_processed += status['stats']['tasks_processed']
            total_failed += status['stats']['tasks_failed']
            total_queue_size += status['stats']['queue_size']
            total_active_workers += status['stats']['active_workers']
        
        uptime = (datetime.now() - self.global_stats['start_time']).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'pipelines': pipeline_statuses,
            'totals': {
                'pipelines_count': len(self.pipelines),
                'tasks_processed': total_processed,
                'tasks_failed': total_failed,
                'queue_size': total_queue_size,
                'active_workers': total_active_workers
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all pipelines."""
        healthy_pipelines = 0
        pipeline_health = {}
        issues = []
        
        for name, pipeline in self.pipelines.items():
            health = pipeline.get_health_check()
            pipeline_health[name] = health
            
            if health['healthy']:
                healthy_pipelines += 1
            else:
                issues.extend([f"{name}: {issue}" for issue in health['issues']])
        
        overall_healthy = healthy_pipelines == len(self.pipelines)
        
        return {
            'overall_healthy': overall_healthy,
            'healthy_pipelines': healthy_pipelines,
            'total_pipelines': len(self.pipelines),
            'pipeline_health': pipeline_health,
            'issues': issues
        }