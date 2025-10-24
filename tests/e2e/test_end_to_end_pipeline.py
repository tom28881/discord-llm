import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

import requests

from load_messages import load_messages_once
from lib.discord_client import Discord
from lib.database import get_recent_messages
from lib.importance_detector import MessageImportanceDetector


@pytest.mark.e2e
def test_full_pipeline_import_and_analysis(temp_db, sample_patterns):
    """Simulate an end-to-end import cycle and verify downstream analysis."""
    server_id = 123456789012345678
    channel_id = 111111111111111111
    base_timestamp = int(datetime.now().timestamp())

    mock_client = MagicMock(spec=Discord)
    mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
    mock_client.get_channel_ids.return_value = [(channel_id, "general")]
    mock_client.fetch_messages.return_value = [
        ("1001", "URGENT: Server maintenance needed immediately!", base_timestamp),
        ("1002", "New group buy for keyboards starting tomorrow", base_timestamp + 60),
    ]

    with patch("load_messages.initialize_discord_client", return_value=mock_client), \
         patch("load_messages.load_config", return_value={"forbidden_channels": []}), \
         patch("load_messages.load_forbidden_channels", return_value=set()), \
         patch("load_messages.add_forbidden_channel") as mock_add_forbidden:
        summary = load_messages_once(
            sleep_between_servers=False,
            sleep_between_channels=False
        )

    assert summary["processed_servers"] == 1
    assert summary["messages_saved"] == 2
    assert summary["servers"][0]["messages_saved"] == 2
    mock_add_forbidden.assert_not_called()

    recent_messages = get_recent_messages(server_id, hours=24)
    assert len(recent_messages) == 2

    detector = MessageImportanceDetector(sample_patterns)
    scores = [detector.detect_importance(message).score for message in recent_messages]
    assert max(scores) >= 0.7


@pytest.mark.e2e
def test_forbidden_channel_handling(temp_db):
    """Verify that forbidden channels are tracked when a 403 is encountered."""
    server_id = 123456789012345678
    allowed_channel = 111111111111111111
    forbidden_channel = 222222222222222222
    base_timestamp = int(datetime.now().timestamp())

    mock_client = MagicMock(spec=Discord)
    mock_client.get_server_ids.return_value = [(server_id, "Test Server")]
    mock_client.get_channel_ids.return_value = [
        (allowed_channel, "general"),
        (forbidden_channel, "secret"),
    ]
    http_error = requests.HTTPError("Forbidden")
    http_error.response = MagicMock(status_code=403)
    mock_client.fetch_messages.side_effect = [
        [("1001", "Regular update", base_timestamp)],
        http_error,
    ]

    config = {"forbidden_channels": []}

    with patch("load_messages.initialize_discord_client", return_value=mock_client), \
         patch("load_messages.load_config", return_value=config), \
         patch("load_messages.load_forbidden_channels", return_value=set()), \
         patch("load_messages.add_forbidden_channel") as mock_add_forbidden:
        summary = load_messages_once(
            sleep_between_servers=False,
            sleep_between_channels=False
        )

    assert summary["messages_saved"] == 1
    mock_add_forbidden.assert_called_once_with(config, forbidden_channel)
    recent_messages = get_recent_messages(server_id, hours=24)
    assert recent_messages == ["Regular update"]
