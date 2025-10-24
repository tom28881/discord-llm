import os
import logging
import argparse
import time
from typing import Optional, Dict, Any, Iterable, Set
from datetime import datetime, timedelta

import requests

from lib.database import init_db, save_messages, get_last_message_id, save_server, save_channel
from lib.discord_client import Discord
from lib.config_manager import load_config, load_forbidden_channels, add_forbidden_channel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('discord_bot')


def fetch_and_store_messages(
    client: Discord,
    forbidden_channels: set,
    config: dict,
    server_id: str,
    server_name: str,
    sleep_between_channels: bool = True,
    channel_filter: Optional[Iterable[int]] = None,
    hours_back: Optional[int] = None
) -> int:
    logger.info("Fetching and storing messages...")
    channel_info = client.get_channel_ids()

    # Save server information
    save_server(server_id, server_name)

    total_saved = 0

    allowed_channels: Optional[Set[int]] = None
    if channel_filter is not None:
        allowed_channels = {int(ch_id) for ch_id in channel_filter}

    min_timestamp = None
    if hours_back is not None:
        min_timestamp = int((datetime.now() - timedelta(hours=hours_back)).timestamp())

    for channel_id, channel_name in channel_info:
        if allowed_channels is not None and channel_id not in allowed_channels:
            continue
        if channel_id in forbidden_channels:
            logger.info(f"Skipping forbidden channel: #{channel_name} (ID: {channel_id})")
            continue

        if not channel_name:  # Skip channels with no name
            continue

        # Save channel information
        save_channel(channel_id, server_id, channel_name)

        last_message_id = get_last_message_id(server_id, channel_id)
        try:
            logger.info(f"Fetching messages from channel: #{channel_name} (ID: {channel_id})")
            new_messages = client.fetch_messages(
                channel_id,
                last_message_id,
                5000,
                min_timestamp=min_timestamp
            )
            if new_messages:
                messages_to_save = [
                    (server_id, channel_id, message_id, content, sent_at)
                    for message_id, content, sent_at in new_messages
                ]
                save_messages(messages_to_save)
                logger.info(f"Stored {len(new_messages)} new messages from channel #{channel_name} (ID: {channel_id})")
                total_saved += len(new_messages)
            else:
                logger.info(f"No new messages in channel #{channel_name} (ID: {channel_id})")
        except requests.HTTPError as http_err:
            handle_http_error(http_err, channel_id, channel_name, config)
        except Exception as err:
            logger.error(f"Unexpected error while fetching messages from channel #{channel_name} (ID: {channel_id}): {err}")
            
        # Sleep for 1 second between channels
        if sleep_between_channels:
            logger.info("Waiting 1 second before next channel")
            time.sleep(1)

    return total_saved


def handle_http_error(http_err: requests.HTTPError, channel_id: str, channel_name: str, config: dict):
    if http_err.response.status_code == 403:
        logger.error(f"Forbidden access to channel #{channel_name} (ID: {channel_id}): {http_err}")
        add_forbidden_channel(config, channel_id)
    else:
        logger.error(f"HTTP error while fetching messages from channel #{channel_name} (ID: {channel_id}): {http_err}")


def initialize_discord_client(server_id: Optional[str]) -> Discord:
    env_path = os.path.join(os.getcwd(), '.env')
    user_token = None

    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('DISCORD_TOKEN='):
                    user_token = line.split('=', 1)[1].strip()
                    break
    else:
        logger.error(".env file not found!")
        raise RuntimeError(".env file not found!")

    if not user_token:
        logger.error("DISCORD_TOKEN not found in .env file")
        raise RuntimeError("DISCORD_TOKEN not found in .env file")

    client = Discord(token=user_token, server_id=server_id)
    logger.info("Discord client initialized.")
    return client


def load_messages_once(
    server_id: Optional[str] = None,
    sleep_between_servers: bool = False,
    sleep_between_channels: bool = False,
    channel_ids: Optional[Iterable[int]] = None,
    hours_back: Optional[int] = None
) -> Dict[str, Any]:
    """Run a single message import cycle and return summary details."""

    logger.info("Starting single-run message import.")

    config = load_config()
    forbidden_channels = load_forbidden_channels(config)

    init_db()
    logger.info("Database initialized.")

    client = initialize_discord_client(None)

    if server_id:
        available_servers = dict(client.get_server_ids())
        resolved_name = available_servers.get(int(server_id), "Specified Server")
        server_info = [(int(server_id), resolved_name)]
    else:
        server_info = client.get_server_ids()

    summary: Dict[str, Any] = {
        "processed_servers": 0,
        "messages_saved": 0,
        "servers": []
    }

    for index, (srv_id, server_name) in enumerate(server_info):
        logger.info(f"Processing server: {server_name} (ID: {srv_id})")
        client.server_id = str(srv_id)

        try:
            channel_filter = None
            if channel_ids is not None:
                channel_filter = [int(ch_id) for ch_id in channel_ids]

            saved_count = fetch_and_store_messages(
                client,
                forbidden_channels,
                config,
                str(srv_id),
                server_name,
                sleep_between_channels=sleep_between_channels,
                channel_filter=channel_filter,
                hours_back=hours_back
            )
            summary["messages_saved"] += saved_count
            summary["servers"].append({
                "id": str(srv_id),
                "name": server_name,
                "messages_saved": saved_count,
                "channels_filter": channel_filter,
                "hours_back": hours_back
            })
            logger.info(
                "Server %s (ID: %s) processed successfully with %d new messages.",
                server_name,
                srv_id,
                saved_count
            )
        except Exception as exc:
            logger.exception(f"Error processing server {server_name} (ID: {srv_id}): {exc}")
            summary["servers"].append({
                "id": str(srv_id),
                "name": server_name,
                "error": str(exc)
            })

        if sleep_between_servers and index < len(server_info) - 1:
            logger.info("Waiting 10 seconds before next server")
            time.sleep(10)

    summary["processed_servers"] = len(server_info)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Load messages from Discord server.")
    parser.add_argument('--server_id', type=str, required=False, help='Optional: Specific Discord server ID to process')
    args = parser.parse_args()

    logger.info("Script started.")
    while True:
        try:
            summary = load_messages_once(
                server_id=args.server_id,
                sleep_between_servers=True,
                sleep_between_channels=True
            )
            logger.info(
                "Cycle completed. Processed %d servers with %d new messages.",
                summary.get("processed_servers", 0),
                summary.get("messages_saved", 0)
            )
        except Exception as e:
            logger.exception(f"An error occurred during execution: {e}")
            # Sleep for 30 seconds if there's an error before retrying
            logger.info("Error occurred, waiting 30 seconds before retry")
            time.sleep(30)


if __name__ == "__main__":
    main()
