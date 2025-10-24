"""
Unit tests for config_manager.py
"""
import pytest
import json
import os
from unittest.mock import patch, mock_open

from lib.config_manager import (
    load_config,
    save_config,
    load_forbidden_channels,
    add_forbidden_channel,
    CONFIG_FILE,
    FORBIDDEN_CHANNELS_KEY
)

@pytest.mark.unit
class TestConfigManager:
    """Test suite for the configuration manager."""

    def test_load_config_file_exists(self):
        """Test loading a valid, existing config file."""
        mock_config_data = {FORBIDDEN_CHANNELS_KEY: [123, 456]}
        mock_json = json.dumps(mock_config_data)
        
        with patch("builtins.open", mock_open(read_data=mock_json)) as mock_file, \
             patch("os.path.exists", return_value=True):
            
            config = load_config()
            
            mock_file.assert_called_once_with(CONFIG_FILE, 'r')
            assert config == mock_config_data

    @patch('lib.config_manager.save_config')
    def test_load_config_file_not_exists(self, mock_save_config):
        """Test that a default config is created if none exists."""
        default_config = {FORBIDDEN_CHANNELS_KEY: []}
        
        with patch("os.path.exists", return_value=False):
            config = load_config()
            
            assert config == default_config
            mock_save_config.assert_called_once_with(default_config)

    def test_load_config_invalid_json(self):
        """Test loading a corrupted or invalid JSON config file."""
        invalid_json = "{'key': 'value'"  # Malformed JSON
        
        with patch("builtins.open", mock_open(read_data=invalid_json)), \
             patch("os.path.exists", return_value=True), \
             patch('lib.config_manager.logger') as mock_logger:
            
            config = load_config()
            
            assert config == {FORBIDDEN_CHANNELS_KEY: []}
            mock_logger.error.assert_called_once()
            assert "Error decoding JSON" in mock_logger.error.call_args[0][0]

    def test_save_config_success(self):
        """Test successfully saving a config file."""
        config_to_save = {FORBIDDEN_CHANNELS_KEY: [789]}
        
        with patch("builtins.open", mock_open()) as mock_file:
            save_config(config_to_save)
            
            mock_file.assert_called_once_with(CONFIG_FILE, 'w')
            handle = mock_file()
            # json.dump with indent calls write multiple times, so we can't use assert_called_once_with.
            # Instead, we check that the full content written to the file is what we expect.
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            assert written_data == json.dumps(config_to_save, indent=4)

    def test_save_config_io_error(self):
        """Test handling of an IOError during saving."""
        config_to_save = {FORBIDDEN_CHANNELS_KEY: [111]}
        
        with patch("builtins.open", mock_open()) as mock_file, \
             patch('lib.config_manager.logger') as mock_logger:
            
            mock_file.side_effect = IOError("Disk full")
            save_config(config_to_save)
            
            mock_logger.error.assert_called_once()
            assert "Error saving config" in mock_logger.error.call_args[0][0]

    def test_load_forbidden_channels(self):
        """Test loading the set of forbidden channels from config."""
        mock_config = {FORBIDDEN_CHANNELS_KEY: [10, 20, 30, 20]}
        
        forbidden_set = load_forbidden_channels(mock_config)
        
        assert isinstance(forbidden_set, set)
        assert forbidden_set == {10, 20, 30}

    def test_load_forbidden_channels_key_missing(self):
        """Test loading forbidden channels when the key is missing from config."""
        mock_config = {"other_key": "other_value"}
        
        forbidden_set = load_forbidden_channels(mock_config)
        
        assert isinstance(forbidden_set, set)
        assert forbidden_set == set()

    @patch('lib.config_manager.save_config')
    def test_add_forbidden_channel_new_id(self, mock_save_config):
        """Test adding a new channel ID to the forbidden list."""
        mock_config = {FORBIDDEN_CHANNELS_KEY: [10, 20]}
        new_channel_id = 30
        
        add_forbidden_channel(mock_config, new_channel_id)
        
        expected_config = {FORBIDDEN_CHANNELS_KEY: [10, 20, 30]}
        assert mock_config == expected_config
        mock_save_config.assert_called_once_with(expected_config)

    @patch('lib.config_manager.save_config')
    def test_add_forbidden_channel_existing_id(self, mock_save_config):
        """Test that adding an existing channel ID does not cause duplicates or resave."""
        mock_config = {FORBIDDEN_CHANNELS_KEY: [10, 20]}
        existing_channel_id = 20
        
        add_forbidden_channel(mock_config, existing_channel_id)
        
        assert mock_config == {FORBIDDEN_CHANNELS_KEY: [10, 20]}
        mock_save_config.assert_not_called()
