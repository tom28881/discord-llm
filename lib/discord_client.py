import logging
import requests
from datetime import datetime
from typing import List, Tuple, Optional

logger = logging.getLogger('discord_bot')

class Discord:
    BASE_URL = 'https://discord.com/api/v9'
    ALLOWED_CHANNEL_TYPES = {0, 5, 13}  # Text, News, Stage channels

    def __init__(self, token: str, server_id: str = None):
        token = token.strip() if token else token

        if not token or len(token) < 10:
            raise ValueError("Valid Discord token must be provided.")
            
        self.server_id = server_id
        self.headers = {
            'Authorization': token,
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/605.1.15 (KHTML, like Gecko) '
                'Version/18.0.1 Safari/605.1.15'
            )
        }

    def get_channel_ids(self) -> List[tuple]:
        logger.info(f"Retrieving channels for server ID {self.server_id}...")
        url = f"{self.BASE_URL}/guilds/{self.server_id}/channels"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        channels = response.json()
        channel_info = []
        for channel in channels:
            channel_type = channel.get('type', 0)
            if channel_type in self.ALLOWED_CHANNEL_TYPES:
                channel_info.append((int(channel['id']), channel['name']))
        logger.info(f"Found {len(channel_info)} channels to process.")
        return channel_info

    def get_server_ids(self) -> List[tuple]:
        logger.info(f"Retrieving server IDs...")
        url = f"{self.BASE_URL}/users/@me/guilds"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        servers = response.json()
        server_info = [(int(server['id']), server['name']) for server in servers]
        logger.info(f"Found {len(server_info)} servers to process.")
        return server_info
        
    def fetch_messages(
        self,
        channel_id: int,
        since_message_id: str = None,
        limit: int = 100,
        min_timestamp: Optional[int] = None
    ) -> List[Tuple[str, str, int]]:
        logger.debug(
            f"Fetching up to {limit} new messages from channel ID {channel_id}."
        )
        all_messages = []
        params = {'limit': 100}  # Discord API limit per request

        use_after_pagination = since_message_id is not None

        if use_after_pagination:
            params['after'] = since_message_id
            params.pop('before', None)

        url = f"{self.BASE_URL}/channels/{channel_id}/messages"
        batch_number = 1

        stop_fetching = False

        while len(all_messages) < limit and not stop_fetching:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            messages = response.json()

            if not messages:
                break

            if use_after_pagination:
                messages.sort(key=lambda item: int(item['id']))

            for message in messages:
                if len(all_messages) >= limit:
                    break
                raw_timestamp = message['timestamp']
                if isinstance(raw_timestamp, str):
                    normalized_timestamp = raw_timestamp.replace('Z', '+00:00')
                    timestamp = int(datetime.fromisoformat(normalized_timestamp).timestamp())
                else:
                    timestamp = int(raw_timestamp)

                if min_timestamp is not None and timestamp < min_timestamp:
                    if use_after_pagination:
                        continue
                    stop_fetching = True
                    break
                all_messages.append((
                    message['id'],
                    message.get('content', ''),
                    timestamp
                ))

            logger.debug(
                f"Fetched {len(messages)} messages from channel ID {channel_id}."
            )

            logger.info(
                f"Batch {batch_number}: Retrieved {len(all_messages)} total messages so far from channel ID {channel_id}."
            )
            batch_number += 1

            if len(messages) < 100:
                # Fetched less than the maximum, no more messages to retrieve
                break

            if use_after_pagination:
                next_after = messages[-1]['id']
                if params.get('after') == next_after:
                    break
                params['after'] = next_after
            else:
                params['before'] = messages[-1]['id']

            if stop_fetching:
                break

        logger.debug(
            f"Total fetched messages from channel ID {channel_id}: {len(all_messages)}."
        )
        return all_messages
