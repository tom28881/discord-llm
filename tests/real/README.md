# Real-World Integration Tests

These tests use **REAL Discord API and REAL data** to validate the message import system end-to-end.

## ⚠️ Important Warning

**These tests connect to actual Discord servers and import real messages!**

- Use a **TEST server**, not your production server
- Tests will create a separate database: `data/test_real_db.sqlite`
- API rate limits apply - tests include delays
- Tests are opt-in and skip by default

## Prerequisites

### 1. Discord Setup

You need:
- A Discord account
- A test Discord server (create one for testing)
- At least one channel with some messages
- Your Discord user token

### 2. Get Your Discord Token

**Method 1: Browser DevTools**
1. Open Discord in web browser
2. Open DevTools (F12)
3. Go to Network tab
4. Reload Discord
5. Filter requests by "api"
6. Look for "authorization" header in request headers
7. Copy the token value

**Method 2: Use existing .env**
If you already have `DISCORD_TOKEN` in your `.env`, you're set!

### 3. Get Your Test Server ID

1. Enable Developer Mode in Discord (Settings → Advanced → Developer Mode)
2. Right-click your test server
3. Click "Copy ID"
4. This is your `TEST_SERVER_ID`

### 4. (Optional) Get Test Channel ID

1. Right-click a channel in your test server
2. Click "Copy ID"
3. This is your `TEST_CHANNEL_ID`

## Configuration

Add to your `.env` file:

```bash
# Required for real tests
ENABLE_REAL_TESTS=1
DISCORD_TOKEN=your_discord_token_here
TEST_SERVER_ID=your_test_server_id_here

# Optional - test specific channel only
TEST_CHANNEL_ID=your_test_channel_id_here

# Optional - test configuration
TEST_MAX_MESSAGES=100
TEST_HOURS_BACK=1
TEST_RATE_LIMIT_SLEEP=1.0

# Optional - cleanup test DB after each test
CLEANUP_TEST_DB=0
```

### Configuration Options

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENABLE_REAL_TESTS` | Yes | `0` | Set to `1` to enable real tests |
| `DISCORD_TOKEN` | Yes | - | Your Discord user token |
| `TEST_SERVER_ID` | Yes | - | Test server ID (use test server!) |
| `TEST_CHANNEL_ID` | No | - | Specific channel to test (optional) |
| `TEST_HOURS_BACK` | No | `1` | How many hours of messages to import |
| `TEST_MAX_MESSAGES` | No | `100` | Max messages to import (not enforced by all tests) |
| `TEST_RATE_LIMIT_SLEEP` | No | `1.0` | Seconds to wait between API calls |
| `CLEANUP_TEST_DB` | No | `0` | Set to `1` to delete test DB after tests |

## Running Tests

### Run All Real Tests

```bash
# Run all real tests with verbose output
pytest tests/real/ -m real -v -s

# Or from project root
pytest -m real -v -s
```

### Run Specific Test Classes

```bash
# Test Discord connection only
pytest tests/real/test_real_api_import.py::TestRealDiscordConnection -m real -v -s

# Test message import
pytest tests/real/test_real_api_import.py::TestRealMessageImport -m real -v -s

# Test incremental import
pytest tests/real/test_real_api_import.py::TestRealIncrementalImport -m real -v -s

# Test performance
pytest tests/real/test_real_api_import.py::TestRealPerformance -m real -v -s
```

### Run Specific Test

```bash
pytest tests/real/test_real_api_import.py::TestRealMessageImport::test_import_real_messages -m real -v -s
```

## What Gets Tested

### 1. Discord Connection Tests
- ✅ Can connect to Discord API
- ✅ Can retrieve server list
- ✅ Can retrieve channels from test server

### 2. Message Import Tests
- ✅ Import real messages from Discord
- ✅ Store messages in database
- ✅ Verify server/channel/message relationships
- ✅ Check data integrity (no orphans, no duplicates)
- ✅ Verify message content preservation

### 3. Incremental Import Tests
- ✅ Run multiple imports without creating duplicates
- ✅ Only import new messages on second run
- ✅ Track last message ID correctly

### 4. Performance Tests
- ✅ Measure import speed
- ✅ Measure query performance
- ✅ Verify acceptable performance metrics

### 5. Query Tests
- ✅ Time-based filtering (1h, 24h, all time)
- ✅ Channel-based filtering
- ✅ Result accuracy

## Test Database

Real tests use an **isolated test database**:
- Location: `data/test_real_db.sqlite`
- Separate from production `data/db.sqlite`
- Persists after tests for inspection
- Can be manually deleted

### Inspect Test Database

```bash
# Open with sqlite3
sqlite3 data/test_real_db.sqlite

# View tables
.tables

# View message count
SELECT COUNT(*) FROM messages;

# View recent messages
SELECT content, sent_at FROM messages ORDER BY sent_at DESC LIMIT 10;

# Exit
.quit
```

### Clean Up Test Database

```bash
# Manual cleanup
rm data/test_real_db.sqlite

# Or set in .env for automatic cleanup
CLEANUP_TEST_DB=1
```

## Expected Output

Successful test run example:

```
tests/real/test_real_api_import.py::TestRealDiscordConnection::test_can_connect_to_discord 
🔌 Testing real Discord API connection...
✅ Successfully connected! Found 3 servers
   📁 Test Server (ID: 123456789012345678)
   📁 Another Server (ID: 987654321098765432)
PASSED

tests/real/test_real_api_import.py::TestRealMessageImport::test_import_real_messages 
📥 Importing real messages from server 123456789012345678...
   ⏱️  Time window: Last 1 hour(s)
   📢 Channel filter: [111111111111111111]

✅ Import completed in 2.34 seconds
   📊 Servers processed: 1
   💾 Messages saved: 15

📊 Database contents:
   Servers: 1
   Channels: 1
   Messages: 15

📝 Sample messages (most recent):
   [2025-10-24 11:45] Hey everyone, group buy closing soon!
   [2025-10-24 11:30] Thanks for the update
   [2025-10-24 11:15] Meeting at 3pm today
PASSED
```

## Troubleshooting

### Tests Are Skipped

**Issue**: Tests show "SKIPPED" instead of running

**Solutions**:
1. Ensure `-m real` flag is used: `pytest -m real`
2. Check `ENABLE_REAL_TESTS=1` in `.env`
3. Verify `DISCORD_TOKEN` is set
4. Verify `TEST_SERVER_ID` is set

### Authentication Errors

**Issue**: "401 Unauthorized" or "403 Forbidden"

**Solutions**:
1. Verify `DISCORD_TOKEN` is correct
2. Token may have expired - get a fresh one
3. Ensure you have access to the test server

### No Messages Imported

**Issue**: Import completes but 0 messages saved

**Possible reasons**:
1. No messages in time window - increase `TEST_HOURS_BACK`
2. Channel is empty
3. Permissions - ensure you can read the channel
4. Channel ID is wrong

**Solutions**:
```bash
# Try longer time window
TEST_HOURS_BACK=24

# Try without channel filter (all channels)
# Remove or comment out TEST_CHANNEL_ID
```

### Rate Limit Errors

**Issue**: "429 Too Many Requests"

**Solutions**:
1. Increase `TEST_RATE_LIMIT_SLEEP=2.0`
2. Wait a few minutes before retrying
3. Reduce test scope (fewer channels, shorter time)

### Database Errors

**Issue**: Database locked or foreign key errors

**Solutions**:
1. Ensure no other process is using test DB
2. Delete test DB and retry: `rm data/test_real_db.sqlite`
3. Check file permissions

## Safety Features

Real tests include multiple safety measures:

1. **Opt-in by default** - Must explicitly enable
2. **Separate database** - Never touches production data
3. **Rate limiting** - Built-in delays between API calls
4. **Limited scope** - Tests small time windows
5. **Clear markers** - Easy to identify and skip
6. **Skip in CI** - Won't run in automated pipelines

## CI/CD Integration

Real tests are marked with `skip_ci` and won't run in CI/CD by default.

To run in CI (not recommended unless you have dedicated test server):

```yaml
# GitHub Actions example
- name: Run Real Tests
  if: github.event_name == 'workflow_dispatch'  # Manual trigger only
  env:
    ENABLE_REAL_TESTS: 1
    DISCORD_TOKEN: ${{ secrets.DISCORD_TOKEN }}
    TEST_SERVER_ID: ${{ secrets.TEST_SERVER_ID }}
  run: pytest -m real -v
```

## Best Practices

1. **Use a dedicated test server** - Don't test on production!
2. **Keep test scope small** - Use short time windows (1 hour)
3. **Run periodically** - Not in every CI run
4. **Inspect test database** - Verify results manually
5. **Monitor API usage** - Respect Discord rate limits
6. **Clean up old test DBs** - Delete periodically

## FAQ

**Q: Will these tests modify my Discord server?**
A: No, tests only READ messages, never write or modify.

**Q: How often should I run these tests?**
A: Before major releases or after significant changes to import logic.

**Q: Can I run tests on my production server?**
A: You CAN, but you SHOULDN'T. Always use a test server!

**Q: What if I don't have a test server?**
A: Create one! It's free and takes 2 minutes.

**Q: Do tests count against API rate limits?**
A: Yes, but tests include delays and are limited in scope.

**Q: Can I run multiple test files in parallel?**
A: Not recommended - they share the same test database.

## Related Documentation

- See `../TEST_DATA_FETCH_IMPORT.md` for mocked tests
- See `../../README.md` for project overview
- See `../../TESTING.md` for general testing guidelines
