# Discord Message Importer

A Python-based tool for importing and analyzing Discord messages with a Streamlit web interface for interactive chat-based exploration of message history.

## Features

- **Automated Message Import**: Continuously fetches and stores messages from Discord servers
- **Multi-Server Support**: Can process multiple Discord servers in a single session
- **Smart Channel Filtering**: Automatically skips forbidden channels and maintains a blocklist
- **Web Interface**: Interactive chat interface using Streamlit for querying message history
- **AI-Powered Analysis**: Uses OpenAI models to summarize and answer questions about Discord messages
- **Database Storage**: SQLite database for efficient message storage and retrieval
- **Keyword Search**: Search messages by keywords and time ranges
- **Rate Limiting**: Built-in delays to respect Discord API limits

## Prerequisites

- Python 3.8 or higher
- Discord user token (for fetching messages)
- OpenAI API key (for message summarization)
- OpenRouter API key (optional, for alternative LLM providers)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd discord_message_importer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```bash
touch .env
```

4. Add the following environment variables to `.env`:
```
DISCORD_TOKEN=your_discord_user_token_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here  # Optional
```

## Configuration

### config.json
The `config.json` file contains a list of forbidden channel IDs that will be skipped during message import. This file is automatically created and updated when the bot encounters channels it cannot access.

```json
{
    "forbidden_channels": [
        // Channel IDs that should be skipped
    ]
}
```

## Usage

### 1. Import Discord Messages

Run the main import script to fetch messages from Discord:

```bash
python load_messages.py
```

Options:
- `--server_id`: (Optional) Specify a single server ID to process
  ```bash
  python load_messages.py --server_id 123456789012345678
  ```

The script will:
- Fetch all accessible servers if no server ID is specified
- Process each server's channels sequentially
- Store messages in the SQLite database
- Automatically retry on errors
- Skip forbidden channels

### 2. Application Interfaces

The project ships with two Streamlit dashboards that target different audiences:

- **Primary analytics interface (`streamlit_app.py`)** – presents the AI features (urgent alerts, purchase predictions, sentiment trends, summaries). This is what end users open most of the time. Start it with:

  ```bash
  streamlit run streamlit_app.py
  ```

  The UI lets you select servers, choose the time range (1–720 hours), adjust the importance threshold, and explore all AI insights.

- **Monitoring & ops interface (`streamlit_monitoring.py`)** – operational dashboard for maintainers. It shows pipeline schedules, model training status, notification queues, and health checks. Launch it separately when you need operational visibility:

  ```bash
  streamlit run streamlit_monitoring.py
  ```

Both interfaces can run in parallel on different ports.

### 3. Additional Scripts

#### Database Management
```bash
# Show database statistics
python scripts/manage_db.py --stats

# Reset the database (WARNING: deletes all data)
python scripts/manage_db.py --reset
```

#### Command-Line Summary Tool
```bash
python scripts/summary.py --server_id 123456789012345678
```

This provides an interactive command-line interface for querying messages.

## Project Structure

```
discord_message_importer/
├── load_messages.py        # Main script for importing Discord messages
├── streamlit_app.py        # Web interface for querying messages
├── requirements.txt        # Python dependencies
├── config.json            # Configuration file (auto-generated)
├── .env                   # Environment variables (create this)
├── lib/                   # Core library modules
│   ├── config_manager.py  # Configuration management
│   ├── database.py        # Database operations
│   ├── discord_client.py  # Discord API client
│   └── llm.py            # LLM integration
├── scripts/              # Utility scripts
│   ├── manage_db.py      # Database management tool
│   ├── summary.py        # CLI summary tool
│   ├── delete.py         # Message deletion utilities
│   └── links.py          # Link extraction utilities
└── data/                 # Data directory
    ├── db.sqlite         # SQLite database (auto-created)
    └── config.json       # Backup configuration
```

### Key Library Modules

- `lib/importance_scorer.py` – weighted scoring of mentions, urgency, social signals, and pattern detection for each message.
- `lib/purchase_predictor.py` – analyzes conversations to estimate the probability of group purchases.
- `lib/sentiment_analyzer.py` – sentiment and energy tracking across channels.
- `lib/training_pipeline.py` – scheduled retraining and evaluation pipeline (relies on the `schedule` library).
- `lib/monitoring.py` & `lib/processing_manager.py` – background processing, notifications, and monitoring hooks.
- `lib/llm.py` – integration layer for OpenAI/Gemini models used in AI summaries.

## Database Schema

The SQLite database contains three main tables:

1. **servers**: Discord server information
   - `id`: Discord server ID (primary key)
   - `name`: Server name

2. **channels**: Discord channel information
   - `id`: Discord channel ID (primary key)
   - `server_id`: Associated server ID (foreign key)
   - `name`: Channel name

3. **messages**: Discord messages
   - `id`: Discord message ID (primary key)
   - `server_id`: Server ID (foreign key)
   - `channel_id`: Channel ID (foreign key)
   - `content`: Message content
   - `sent_at`: Unix timestamp

## Important Notes

1. **Discord Token**: This tool requires a Discord user token, not a bot token. Use at your own risk and ensure compliance with Discord's Terms of Service.

2. **Rate Limiting**: The tool includes built-in delays to avoid hitting Discord's rate limits:
   - 1 second between channels
   - 10 seconds between servers
   - 30 seconds retry delay on errors

3. **Privacy**: This tool stores Discord messages locally. Ensure you have permission to archive messages from the servers you're accessing.

4. **API Keys**: Keep your API keys secure and never commit them to version control.

## Dependencies

- The `schedule` package (pinned in `requirements.txt`) powers recurring jobs in `lib/training_pipeline.py`. Ensure it is installed when running training or monitoring services.

## Further Documentation

- `README_PRODUCTION.md` – production deployment checklist, monitoring cadence, alerting.
- `SETUP_GUIDE.md` – detailed environment/bootstrap steps.
- `UZIVATELSKA_PRIRUCKA.md` – Czech end-user guide covering all UI cards and AI features.
- `IMPLEMENTATION_SUMMARY.md` – architectural overview and component breakdown.
- `NOTIFICATION_SYSTEM.md` – notification workflows and integrations.

## Troubleshooting

1. **403 Forbidden Errors**: The tool automatically adds inaccessible channels to the forbidden list.

2. **Database Issues**: Use `python scripts/manage_db.py --reset` to reset the database if corrupted.

3. **Memory Issues**: For large servers, consider processing in smaller batches or increasing the system memory.

4. **Streamlit Connection Issues**: Ensure port 8501 is available or specify a different port:
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```