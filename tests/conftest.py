"""
Global pytest configuration and fixtures for Discord LLM testing.
"""
import os
import sys
import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Generator, Dict, Any, List

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.database import init_db, DB_NAME, DATA_DIR
from lib.discord_client import Discord
from lib.llm import get_completion


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="function")
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp_file:
        temp_db_path = tmp_file.name
    
    # Monkey patch the DB_NAME for this test
    original_db_name = DB_NAME
    original_data_dir = DATA_DIR
    
    import lib.database
    lib.database.DB_NAME = temp_db_path
    lib.database.DATA_DIR = Path(temp_db_path).parent
    
    # Initialize the temporary database
    init_db()
    
    yield temp_db_path
    
    # Cleanup
    lib.database.DB_NAME = original_db_name
    lib.database.DATA_DIR = original_data_dir
    if os.path.exists(temp_db_path):
        os.unlink(temp_db_path)


@pytest.fixture(scope="function")
def mock_discord_client():
    """Mock Discord client for testing."""
    client = Mock(spec=Discord)
    client.server_id = "123456789012345678"
    client.headers = {"Authorization": "mock_token"}
    
    # Mock methods
    client.get_server_ids.return_value = [
        (123456789012345678, "Test Server"),
        (987654321098765432, "Another Server")
    ]
    
    client.get_channel_ids.return_value = [
        (111111111111111111, "general"),
        (222222222222222222, "random"),
        (333333333333333333, "announcements")
    ]
    
    client.fetch_messages.return_value = [
        ("1001", "Hello world!", 1703030400),
        ("1002", "How are you?", 1703030500),
        ("1003", "Great day today!", 1703030600)
    ]
    
    return client


@pytest.fixture(scope="function")
def mock_llm_client():
    """Mock LLM client for testing."""
    with patch('lib.llm.get_completion') as mock_completion:
        mock_completion.return_value = "This is a mock LLM response for testing purposes."
        yield mock_completion


@pytest.fixture(scope="function")
def sample_messages():
    """Sample Discord messages for testing."""
    return [
        {
            "id": "1001",
            "content": "Hey everyone, there's a group buy for mechanical keyboards starting tomorrow!",
            "timestamp": "2024-01-15T10:00:00Z",
            "channel_id": "111111111111111111",
            "server_id": "123456789012345678",
            "importance_score": 0.9
        },
        {
            "id": "1002", 
            "content": "Just saying hi to everyone",
            "timestamp": "2024-01-15T10:05:00Z",
            "channel_id": "111111111111111111",
            "server_id": "123456789012345678",
            "importance_score": 0.2
        },
        {
            "id": "1003",
            "content": "URGENT: Server maintenance in 1 hour, please save your work",
            "timestamp": "2024-01-15T11:00:00Z", 
            "channel_id": "333333333333333333",
            "server_id": "123456789012345678",
            "importance_score": 0.95
        },
        {
            "id": "1004",
            "content": "Meeting cancelled for today",
            "timestamp": "2024-01-15T09:00:00Z",
            "channel_id": "333333333333333333", 
            "server_id": "123456789012345678",
            "importance_score": 0.7
        }
    ]


@pytest.fixture(scope="function")
def sample_patterns():
    """Sample patterns for importance detection."""
    return {
        "group_buy": {
            "keywords": ["group buy", "gb", "interest check", "ic", "drop"],
            "weight": 0.8,
            "context_keywords": ["keyboard", "switches", "keycaps", "price", "shipping"]
        },
        "urgent": {
            "keywords": ["urgent", "emergency", "asap", "immediately", "critical"],
            "weight": 0.9,
            "context_keywords": ["down", "issue", "problem", "fix", "help"]
        },
        "event": {
            "keywords": ["meeting", "event", "workshop", "conference", "deadline"],
            "weight": 0.7,
            "context_keywords": ["tomorrow", "today", "schedule", "time", "date"]
        },
        "announcement": {
            "keywords": ["announcement", "news", "update", "release", "launch"],
            "weight": 0.6,
            "context_keywords": ["new", "version", "feature", "important"]
        }
    }


@pytest.fixture(scope="function")
def mock_environment():
    """Mock environment variables for testing."""
    env_vars = {
        "DISCORD_TOKEN": "mock_discord_token_12345",
        "GOOGLE_API_KEY": "mock_google_api_key_67890",
        "OPENAI_API_KEY": "mock_openai_key_abcdef"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture(scope="function")
def mock_requests():
    """Mock HTTP requests for Discord API testing."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status.return_value = None
        
        mock_get.return_value = mock_response
        mock_post.return_value = mock_response
        
        yield {"get": mock_get, "post": mock_post, "response": mock_response}


@pytest.fixture(scope="function")
def performance_baseline():
    """Performance baseline metrics for comparison."""
    return {
        "message_import_rate": 100,  # messages per second
        "query_response_time": 0.5,  # seconds
        "database_write_time": 0.1,  # seconds per message
        "llm_response_time": 2.0,    # seconds
        "memory_usage_mb": 150,      # MB
        "importance_scoring_time": 0.05  # seconds per message
    }


@pytest.fixture(scope="function")
def quality_metrics():
    """Quality metrics thresholds."""
    return {
        "importance_accuracy": 0.85,
        "false_positive_rate": 0.15,
        "false_negative_rate": 0.10,
        "notification_relevance": 0.80,
        "system_uptime": 0.99,
        "test_coverage": 0.80
    }


@pytest.fixture
def database_with_sample_data(temp_db, sample_messages):
    """Database populated with sample data for testing."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    # Insert sample servers
    cursor.execute("INSERT INTO servers (id, name) VALUES (?, ?)", 
                   (123456789012345678, "Test Server"))
    
    # Insert sample channels
    cursor.execute("INSERT INTO channels (id, server_id, name) VALUES (?, ?, ?)",
                   (111111111111111111, 123456789012345678, "general"))
    cursor.execute("INSERT INTO channels (id, server_id, name) VALUES (?, ?, ?)",
                   (333333333333333333, 123456789012345678, "announcements"))
    
    # Insert sample messages
    for msg in sample_messages:
        cursor.execute("""
            INSERT INTO messages (id, server_id, channel_id, content, sent_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            int(msg["id"]),
            int(msg["server_id"]),
            int(msg["channel_id"]),
            msg["content"],
            int(datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00")).timestamp())
        ))
    
    conn.commit()
    conn.close()
    
    return temp_db


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "ml" in str(item.fspath):
            item.add_marker(pytest.mark.ml)


class TestMetrics:
    """Test metrics collection utility."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_execution_time(self, test_name: str, execution_time: float):
        """Record test execution time."""
        if "execution_times" not in self.metrics:
            self.metrics["execution_times"] = {}
        self.metrics["execution_times"][test_name] = execution_time
    
    def record_memory_usage(self, test_name: str, memory_mb: float):
        """Record memory usage during test."""
        if "memory_usage" not in self.metrics:
            self.metrics["memory_usage"] = {}
        self.metrics["memory_usage"][test_name] = memory_mb
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics


@pytest.fixture(scope="session")
def test_metrics():
    """Global test metrics collector."""
    return TestMetrics()