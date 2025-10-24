# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Discord Message Importer - a Python application that fetches messages from Discord servers and provides an AI-powered Streamlit interface for querying and analyzing the message history.

## Common Development Commands

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Import Discord messages
python load_messages.py
# Or for a specific server:
python load_messages.py --server_id 123456789012345678

# Launch the Streamlit web interface
streamlit run streamlit_app.py
```

### Database Management

```bash
# Show database statistics
python scripts/manage_db.py --stats

# Reset the database (WARNING: deletes all data)
python scripts/manage_db.py --reset
```

### Other Utilities

```bash
# Interactive CLI summary tool
python scripts/summary.py --server_id 123456789012345678

# Message deletion utilities
python scripts/delete.py

# Link extraction utilities
python scripts/links.py
```

## Architecture Overview

### Core Components

1. **Message Import Pipeline** (`load_messages.py`):
   - Fetches messages from Discord using the discord.py-self library
   - Handles rate limiting with built-in delays (1s between channels, 10s between servers)
   - Automatically maintains a forbidden channels blocklist
   - Stores messages in SQLite database

2. **Web Interface** (`streamlit_app.py`):
   - Streamlit-based chat interface for querying message history
   - Uses OpenAI API for message summarization and question answering
   - Supports time-range filtering and server selection

3. **Data Layer** (`lib/database.py`):
   - SQLite database with three tables: servers, channels, messages
   - Database stored at `data/db.sqlite`
   - All IDs use Discord's native IDs as primary keys

4. **Discord Integration** (`lib/discord_client.py`):
   - Handles Discord API interactions using user tokens
   - Manages channel fetching and message retrieval
   - Implements retry logic for error handling

5. **LLM Integration** (`lib/llm.py`):
   - Integrates with OpenAI and OpenRouter APIs
   - Handles message summarization and query responses

### Environment Configuration

Required environment variables in `.env`:
- `DISCORD_TOKEN`: Discord user token for API access
- `OPENAI_API_KEY`: OpenAI API key for LLM features
- `OPENROUTER_API_KEY`: (Optional) OpenRouter API key for alternative LLM providers

### Database Schema

- **servers**: `id` (Discord server ID), `name`
- **channels**: `id` (Discord channel ID), `server_id`, `name`
- **messages**: `id` (Discord message ID), `server_id`, `channel_id`, `content`, `sent_at` (Unix timestamp)

### Configuration Management

- `config.json`: Maintains list of forbidden channel IDs that should be skipped
- Automatically updated when 403 errors are encountered
- Backup stored in `data/config.json`

## Key Implementation Details

- Uses discord.py-self library for Discord user token authentication
- Implements incremental message fetching (only new messages since last import)
- Built-in rate limiting to respect Discord API limits
- SQLite database for efficient local storage and retrieval
- Streamlit for rapid web UI development
- OpenAI integration for intelligent message analysis