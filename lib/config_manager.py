import os
import json
import logging
from typing import Dict, Set

logger = logging.getLogger('discord_bot')

CONFIG_FILE = 'config.json'
FORBIDDEN_CHANNELS_KEY = "forbidden_channels"

def load_config() -> Dict[str, list]:
    if not os.path.exists(CONFIG_FILE):
        default_config = {FORBIDDEN_CHANNELS_KEY: []}
        save_config(default_config)
        logger.info(f"Created default config file: {CONFIG_FILE}")
        return default_config

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {CONFIG_FILE}")
        return config
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {CONFIG_FILE}")
        return {FORBIDDEN_CHANNELS_KEY: []}

def save_config(config: Dict[str, list]) -> None:
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved config to {CONFIG_FILE}")
    except IOError:
        logger.error(f"Error saving config to {CONFIG_FILE}")

def load_forbidden_channels(config: Dict[str, list]) -> Set[int]:
    return set(config.get(FORBIDDEN_CHANNELS_KEY, []))

def add_forbidden_channel(config: Dict[str, list], channel_id: int) -> None:
    if channel_id not in config[FORBIDDEN_CHANNELS_KEY]:
        config[FORBIDDEN_CHANNELS_KEY].append(channel_id)
        save_config(config)
        logger.info(f"Added channel ID {channel_id} to forbidden_channels.")