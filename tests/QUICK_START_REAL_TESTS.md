# Quick Start: Real Data Tests

Get real integration tests running in 5 minutes!

## Step 1: Get Your Discord Token

**Browser Method (Easiest)**:
1. Open Discord in your browser: https://discord.com/app
2. Open DevTools: Press `F12` or `Right-click → Inspect`
3. Go to **Network** tab
4. Reload the page (`Ctrl+R` or `Cmd+R`)
5. Filter network requests by typing: `api`
6. Click on any request
7. Look for **authorization** in Request Headers
8. Copy the token value (long string starting with letters/numbers)

## Step 2: Get Your Test Server ID

1. Open Discord Settings → Advanced → Enable **Developer Mode**
2. Right-click your test server icon
3. Click **Copy ID**
4. This is your `TEST_SERVER_ID`

**Don't have a test server?**
- Click the `+` icon in Discord
- Create a new server (takes 30 seconds)
- Send a few test messages in #general

## Step 3: Configure .env

Add these lines to your `.env` file:

```bash
# Enable real tests
ENABLE_REAL_TESTS=1

# Your Discord token (from Step 1)
DISCORD_TOKEN=paste_your_token_here

# Your test server ID (from Step 2)
TEST_SERVER_ID=paste_your_server_id_here
```

## Step 4: Run Tests

```bash
# Run all real tests
pytest -m real -v -s

# Or run just the connection test first
pytest tests/real/test_real_api_import.py::TestRealDiscordConnection::test_can_connect_to_discord -m real -v -s
```

## Expected Output

If everything works, you'll see:

```
🔌 Testing real Discord API connection...
✅ Successfully connected! Found 2 servers
   📁 My Test Server (ID: 123456789012345678)
PASSED

📥 Importing real messages...
✅ Import completed in 1.23 seconds
   💾 Messages saved: 10

📝 Sample messages (most recent):
   [2025-10-24 11:45] Hello world!
   [2025-10-24 11:30] Test message
PASSED
```

## Troubleshooting

### "Tests skipped"
- Make sure you used `-m real` flag
- Check `ENABLE_REAL_TESTS=1` in .env

### "DISCORD_TOKEN not found"
- Verify token is in .env file
- No spaces around `=`
- Token should be on same line

### "401 Unauthorized"
- Token expired - get a fresh one
- Copy entire token, no extra characters

### "No messages imported"
- Normal if channel is empty in last hour
- Try: `TEST_HOURS_BACK=24` in .env
- Or send a message in your test server first

## What Next?

1. **Inspect the test database**:
   ```bash
   sqlite3 data/test_real_db.sqlite
   SELECT COUNT(*) FROM messages;
   .quit
   ```

2. **Run specific tests**:
   ```bash
   # Test incremental import
   pytest tests/real/test_real_api_import.py::TestRealIncrementalImport -m real -v -s
   
   # Test performance
   pytest tests/real/test_real_api_import.py::TestRealPerformance -m real -v -s
   ```

3. **Read full documentation**:
   - See `tests/real/README.md` for complete guide
   - See `tests/TEST_DATA_FETCH_IMPORT.md` for mocked tests

## Need Help?

- 📖 Full guide: `tests/real/README.md`
- 🔧 Configuration: `.env.test.example`
- 🧪 Test code: `tests/real/test_real_api_import.py`

## Safety Notes

✅ Tests use isolated database (`data/test_real_db.sqlite`)
✅ Never modifies Discord data (read-only)
✅ Won't run accidentally (opt-in only)
✅ Includes rate limiting

⚠️ **Always use a test server, not production!**
