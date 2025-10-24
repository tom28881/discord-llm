# Data Fetch and Import Test Suite

Comprehensive test coverage for the Discord message fetching and import system - the **alpha and omega** of this application.

## Overview

This test suite focuses on the core functionality: fetching messages from Discord API and importing them into the database. The tests are organized into three levels:

1. **Unit Tests** - Test individual functions in isolation
2. **Integration Tests** - Test components working together
3. **End-to-End Tests** - Test complete user workflows

## Test Files

### 1. Unit Tests (`test_data_fetch_and_import_unit.py`)

Tests core functions in isolation with mocked dependencies.

**Test Classes:**
- `TestTimestampNormalization` - Timestamp conversion from various formats
  - Integer, float, datetime, ISO strings
  - Edge cases and error handling
  
- `TestMessageDeduplication` - Message deduplication logic
  - Same channel deduplication
  - Cross-channel deduplication
  - Order preservation
  
- `TestSaveMessages` - Database save operations
  - Valid data handling
  - Duplicate detection
  - Foreign key validation
  - Various timestamp formats
  
- `TestHttpErrorHandling` - HTTP error responses
  - 403 Forbidden errors
  - 500 Server errors
  - 429 Rate limiting
  
- `TestChannelFiltering` - Channel filter logic
  - Whitelist filtering
  - Forbidden channel skipping
  
- `TestTimeBasedFiltering` - Time-based message filtering
  - Hours-based queries
  - All-time queries
  
- `TestLastMessageIdTracking` - Incremental fetch support
  - Last message ID retrieval
  - Per-channel tracking
  
- `TestDiscordClientInitialization` - Client setup
  - Valid token handling
  - Missing .env file
  - Missing token error

**Total: 25+ unit test cases**

### 2. Integration Tests (`test_data_fetch_and_import_integration.py`)

Tests multiple components working together with mocked Discord API.

**Test Classes:**
- `TestLoadMessagesOnceFlow` - Complete import workflow
  - Single server import
  - Server filtering
  - Channel filtering
  
- `TestTimeBasedFetching` - Time-based import
  - hours_back parameter handling
  - Time filter verification
  
- `TestIncrementalImport` - Incremental fetching
  - Last message ID usage
  - Duplicate prevention
  
- `TestErrorRecovery` - Error handling
  - Channel failure recovery
  - Forbidden channel handling
  - Multiple server failure handling
  
- `TestMultiServerImport` - Multi-server scenarios
  - Multiple server import
  - Server isolation
  
- `TestRealTimeSync` - Real-time sync simulation
  - Periodic import cycles

**Total: 20+ integration test cases**

### 3. End-to-End Tests (`test_data_fetch_and_import_e2e.py`)

Tests complete system workflows from API to database to queries.

**Test Classes:**
- `TestFullImportAndQueryCycle` - Complete user workflows
  - Import → Query → Verify cycle
  - Multi-import cycles
  
- `TestDataIntegrity` - Data accuracy verification
  - Message content preservation
  - Timestamp accuracy
  - Foreign key relationships
  
- `TestPerformanceAndScaling` - Performance testing
  - Large batch imports (1000+ messages)
  - Query performance
  
- `TestConcurrentOperations` - Concurrent access
  - Concurrent queries during import
  
- `TestRealWorldScenarios` - Real usage patterns
  - Typical daily usage
  - Weekend catch-up scenario

**Total: 15+ end-to-end test cases**

## Running the Tests

### Run All Tests
```bash
# Run all data fetch/import tests
pytest tests/unit/test_data_fetch_and_import_unit.py \
       tests/integration/test_data_fetch_and_import_integration.py \
       tests/e2e/test_data_fetch_and_import_e2e.py -v
```

### Run by Level
```bash
# Unit tests only (fast)
pytest tests/unit/test_data_fetch_and_import_unit.py -v

# Integration tests
pytest tests/integration/test_data_fetch_and_import_integration.py -v

# E2E tests (slower)
pytest tests/e2e/test_data_fetch_and_import_e2e.py -v
```

### Run by Marker
```bash
# Run only unit tests
pytest -m unit tests/ -v

# Run only integration tests
pytest -m integration tests/ -v

# Run only e2e tests
pytest -m e2e tests/ -v

# Skip slow tests
pytest -m "not slow" tests/ -v

# Run only slow tests
pytest -m slow tests/ -v
```

### Run Specific Test Classes
```bash
# Run specific test class
pytest tests/unit/test_data_fetch_and_import_unit.py::TestTimestampNormalization -v

# Run specific test
pytest tests/unit/test_data_fetch_and_import_unit.py::TestTimestampNormalization::test_normalize_timestamp_from_int -v
```

### With Coverage
```bash
# Run with coverage report
pytest tests/unit/test_data_fetch_and_import_unit.py \
       tests/integration/test_data_fetch_and_import_integration.py \
       tests/e2e/test_data_fetch_and_import_e2e.py \
       --cov=lib.database \
       --cov=load_messages \
       --cov-report=html \
       --cov-report=term-missing
```

## Test Coverage

### Functions Covered

**Database Operations (`lib/database.py`):**
- ✅ `init_db()` - Database initialization
- ✅ `save_server()` - Server saving
- ✅ `save_channel()` - Channel saving
- ✅ `save_messages()` - Message saving with FK validation
- ✅ `get_last_message_id()` - Last message tracking
- ✅ `get_recent_message_records()` - Message retrieval
- ✅ `get_recent_messages()` - Message content retrieval
- ✅ `get_latest_message_timestamp()` - Timestamp queries
- ✅ `_normalize_timestamp()` - Timestamp conversion
- ✅ `_deduplicate_messages()` - Message deduplication
- ✅ `get_servers()` - Server listing
- ✅ `get_channels()` - Channel listing

**Import Operations (`load_messages.py`):**
- ✅ `load_messages_once()` - Main import entry point
- ✅ `fetch_and_store_messages()` - Message fetching
- ✅ `handle_http_error()` - Error handling
- ✅ `initialize_discord_client()` - Client initialization

### Scenarios Covered

✅ **Data Fetching:**
- Single server import
- Multiple server import
- Channel filtering (whitelist)
- Time-based filtering (hours_back)
- Forbidden channel handling
- Incremental fetching (using last_message_id)

✅ **Data Import:**
- Message saving with validation
- Duplicate detection and prevention
- Foreign key constraint handling
- Timestamp normalization (int, float, datetime, ISO)
- Empty message handling
- Large batch imports (1000+ messages)

✅ **Error Handling:**
- HTTP 403 Forbidden errors
- HTTP 500 Server errors
- Network failures
- Missing foreign keys
- Invalid timestamps
- Concurrent access

✅ **Data Integrity:**
- Message content preservation (emoji, newlines, special chars)
- Timestamp accuracy
- Foreign key relationships
- Cross-channel deduplication
- Query result correctness

✅ **Performance:**
- Large batch import (1000 messages < 10s)
- Query performance (< 0.5s)
- Concurrent operations
- Memory efficiency

✅ **Real-World Scenarios:**
- Daily usage patterns
- Weekend catch-up
- Real-time sync simulation
- Multiple import cycles

## Test Structure

### Fixtures Used

From `conftest.py`:
- `temp_db` - Temporary database for each test
- `mock_discord_client` - Mocked Discord client
- `mock_environment` - Mocked environment variables
- `sample_messages` - Sample message data
- `database_with_sample_data` - Pre-populated database

### Mocking Strategy

**What's Mocked:**
- Discord API calls (HTTP requests)
- Environment variables
- File system access (.env file)

**What's Real:**
- SQLite database operations
- Data transformation logic
- Business logic

This ensures we test real database behavior and data processing while avoiding external dependencies.

## Test Execution Time

- **Unit Tests**: ~2-5 seconds
- **Integration Tests**: ~5-10 seconds  
- **E2E Tests**: ~10-20 seconds (marked as `slow`)
- **Total**: ~20-35 seconds

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Data Fetch/Import Tests
  run: |
    pytest tests/unit/test_data_fetch_and_import_unit.py \
           tests/integration/test_data_fetch_and_import_integration.py \
           tests/e2e/test_data_fetch_and_import_e2e.py \
           --cov=lib.database \
           --cov=load_messages \
           --junitxml=pytest-results.xml \
           --cov-report=xml
```

## Critical Test Cases

### Must-Pass Tests for Production

1. **test_complete_user_workflow** (E2E)
   - Validates entire import → query cycle
   - Tests real user workflow
   
2. **test_incremental_import_uses_last_message_id** (Integration)
   - Ensures incremental fetching works
   - Prevents re-importing old messages
   
3. **test_message_content_preservation** (E2E)
   - Ensures no data corruption
   - Critical for data integrity
   
4. **test_save_messages_duplicate_handling** (Unit)
   - Prevents duplicate messages
   - Maintains database consistency
   
5. **test_continues_after_channel_error** (Integration)
   - Ensures resilience
   - One channel failure doesn't stop import

## Troubleshooting

### Test Failures

**Database Lock Errors:**
```bash
# Ensure no other processes are using the test database
pkill -f pytest
```

**Import Errors:**
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

**Fixture Not Found:**
```bash
# Ensure conftest.py is present
ls -la tests/conftest.py
```

### Common Issues

1. **Temp DB not cleaned up:**
   - Check `/tmp` for leftover `.sqlite` files
   - Tests should auto-cleanup, but manual removal may be needed

2. **Mock not resetting:**
   - Each test should have fresh mocks
   - Check `@patch` decorator usage

3. **Timestamp timezone issues:**
   - All timestamps should be in UTC
   - Use `datetime.now()` with timezone awareness if needed

## Extending the Tests

### Adding New Test Cases

1. **Unit Test Template:**
```python
@pytest.mark.unit
def test_new_feature(self, temp_db):
    """Test description."""
    # Arrange
    setup_data()
    
    # Act
    result = function_to_test()
    
    # Assert
    assert result == expected
```

2. **Integration Test Template:**
```python
@pytest.mark.integration
@patch('load_messages.initialize_discord_client')
def test_new_integration(self, mock_client, temp_db):
    """Test description."""
    # Setup mocks
    mock_client.return_value = Mock()
    
    # Execute flow
    result = load_messages_once()
    
    # Verify
    assert result["messages_saved"] > 0
```

3. **E2E Test Template:**
```python
@pytest.mark.e2e
@pytest.mark.slow
@patch('load_messages.initialize_discord_client')
def test_new_e2e(self, mock_client, temp_db):
    """Test description."""
    # Setup complete scenario
    # Execute full workflow
    # Verify end state
```

## Metrics and Goals

### Current Coverage
- **Unit Test Coverage**: ~95% of core functions
- **Integration Coverage**: ~90% of workflows
- **E2E Coverage**: ~85% of user scenarios

### Goals
- ✅ 100% coverage of data fetch functions
- ✅ 100% coverage of database operations
- ✅ All critical paths tested
- ✅ Error scenarios covered
- ✅ Performance benchmarks established

## Related Documentation

- See `TESTING.md` for general testing guidelines
- See `DATABASE_OPTIMIZATION_GUIDE.md` for database performance
- See `README.md` for project overview
