"""
Fixtures for real-data integration tests.
These tests use actual Discord API and real databases (isolated test DBs).
"""
import os
import pytest
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.discord_client import Discord
from lib.database import init_db
import lib.database as db_module


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "real: Real API/database tests (opt-in, requires credentials)")
    config.addinivalue_line("markers", "skip_ci: Skip in CI/CD environments")


def pytest_collection_modifyitems(config, items):
    """Skip real tests by default unless explicitly requested."""
    if config.getoption("-m") != "real":
        skip_real = pytest.mark.skip(reason="Real tests require -m real flag")
        for item in items:
            if "real" in item.keywords:
                item.add_marker(skip_real)


@pytest.fixture(scope="session")
def check_real_tests_enabled():
    """Check if real tests are explicitly enabled."""
    enabled = os.getenv("ENABLE_REAL_TESTS", "0") == "1"
    if not enabled:
        pytest.skip(
            "Real tests not enabled. Set ENABLE_REAL_TESTS=1 in .env to run these tests."
        )
    return True


@pytest.fixture(scope="session")
def real_discord_token(check_real_tests_enabled):
    """Get real Discord token from environment."""
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        pytest.skip("DISCORD_TOKEN not found in environment. Add it to .env to run real tests.")
    return token


@pytest.fixture(scope="session")
def test_server_id(check_real_tests_enabled):
    """Get test server ID from environment."""
    server_id = os.getenv("TEST_SERVER_ID")
    if not server_id:
        pytest.skip(
            "TEST_SERVER_ID not set. Add TEST_SERVER_ID=your_server_id to .env "
            "to run real tests. Use a test server, not production!"
        )
    return server_id


@pytest.fixture(scope="session")
def test_channel_id():
    """Get optional test channel ID from environment."""
    return os.getenv("TEST_CHANNEL_ID")  # Optional


@pytest.fixture(scope="function")
def test_database() -> Generator[str, None, None]:
    """
    Create an isolated test database for real tests.
    This database is separate from production data.
    """
    # Create test database in data directory
    test_db_path = PROJECT_ROOT / "data" / "test_real_db.sqlite"
    
    # Backup original DB_NAME
    original_db_name = db_module.DB_NAME
    original_data_dir = db_module.DATA_DIR
    
    try:
        # Point to test database
        db_module.DB_NAME = test_db_path
        db_module.DATA_DIR = test_db_path.parent
        
        # Ensure data directory exists
        test_db_path.parent.mkdir(exist_ok=True)
        
        # Initialize test database
        init_db()
        
        yield str(test_db_path)
        
    finally:
        # Restore original paths
        db_module.DB_NAME = original_db_name
        db_module.DATA_DIR = original_data_dir
        
        # Note: We don't delete the test DB to allow inspection
        # User can manually delete data/test_real_db.sqlite if needed


@pytest.fixture(scope="function")
def real_discord_client(real_discord_token, test_server_id):
    """Create a real Discord client."""
    client = Discord(token=real_discord_token, server_id=test_server_id)
    return client


@pytest.fixture(scope="function")
def cleanup_test_db(test_database):
    """Optional fixture to clean up test database after test."""
    yield test_database
    
    # Clean up if requested
    if os.getenv("CLEANUP_TEST_DB", "0") == "1":
        if os.path.exists(test_database):
            os.remove(test_database)


@pytest.fixture
def test_config():
    """Test configuration from environment."""
    return {
        "max_messages": int(os.getenv("TEST_MAX_MESSAGES", "100")),
        "hours_back": int(os.getenv("TEST_HOURS_BACK", "1")),
        "rate_limit_sleep": float(os.getenv("TEST_RATE_LIMIT_SLEEP", "1.0")),
    }
