"""
Unit tests for conversation_threader.py
"""
import pytest
from unittest.mock import patch, MagicMock, ANY
import numpy as np

from lib.conversation_threader import ConversationThreader


@pytest.fixture
def threader():
    """Provides a ConversationThreader instance with a mocked database and vectorizer."""
    # Patch the TfidfVectorizer where it is imported and used, not globally.
    with patch('sqlite3.connect') as mock_sqlite, \
         patch('lib.conversation_threader.TfidfVectorizer') as MockTfidfVectorizer:

        # 1. Configure the mock class to return a mock instance when called.
        mock_vectorizer_instance = MagicMock()
        MockTfidfVectorizer.return_value = mock_vectorizer_instance

        # 2. Configure the mock instance's methods.
        def mock_fit_transform(texts):
            # Return a mock sparse matrix that has a .toarray() method
            mock_array = np.zeros((len(texts), 50))
            mock_sparse_matrix = MagicMock()
            mock_sparse_matrix.toarray.return_value = mock_array
            return mock_sparse_matrix
        
        mock_vectorizer_instance.fit_transform.side_effect = mock_fit_transform

        # 3. Now, when ConversationThreader is instantiated, it will use the mocked class.
        threader_instance = ConversationThreader(db_path=':memory:')
        
        yield threader_instance, mock_sqlite


@pytest.mark.unit
class TestConversationThreader:
    """Test suite for the ConversationThreader."""

    def test_thread_messages_empty_list(self, threader):
        threader_instance, _ = threader
        assert threader_instance.thread_messages([]) == []

    def test_single_message_is_single_thread(self, threader):
        threader_instance, _ = threader
        messages = [
            {'id': 1, 'content': 'Hello', 'sent_at': 1700000000, 'author_id': 'user1'}
        ]
        
        threads = threader_instance.thread_messages(messages)
        
        assert len(threads) == 1
        assert threads[0]['thread_type'] == 'single'
        assert len(threads[0]['messages']) == 1

    def test_messages_close_in_time_are_grouped(self, threader):
        threader_instance, _ = threader
        messages = [
            {'id': 1, 'content': 'Hello', 'sent_at': 1700000000, 'author_id': 'user1'},
            {'id': 2, 'content': 'How are you?', 'sent_at': 1700000060, 'author_id': 'user1'}
        ]
        
        # This test now relies on the default clustering logic, which is fine for grouping
        threads = threader_instance.thread_messages(messages)
        assert len(threads) == 1
        assert len(threads[0]['messages']) == 2

    def test_messages_far_apart_in_time_are_split(self, threader):
        threader_instance, _ = threader
        messages = [
            {'id': 1, 'content': 'First', 'sent_at': 1700000000},
            {'id': 2, 'content': 'Second', 'sent_at': 1700003600} # 60 mins later
        ]
        
        threads = threader_instance.thread_messages(messages)
        
        assert len(threads) == 2

    def test_determine_thread_type(self, threader):
        threader_instance, _ = threader
        assert threader_instance._determine_thread_type([{}]) == 'single'
        assert threader_instance._determine_thread_type([{}, {}]) == 'exchange'
        assert threader_instance._determine_thread_type([{}, {}, {}]) == 'discussion'
        
        # Content-based checks are only triggered for threads with more than 5 messages
        long_purchase_messages = [
            {'content': 'Thinking about a new keyboard'},
            {'content': 'What is the price on that one?'},
            {'content': 'And what about shipping?'},
            {'content': 'I think I will buy it'},
            {'content': 'Thanks for the info'},
            {'content': 'Order placed'}
        ]
        assert threader_instance._determine_thread_type(long_purchase_messages) == 'purchase_discussion'

    def test_save_thread_is_called(self, threader):
        threader_instance, _ = threader
        messages = [
            {'id': 1, 'content': 'First', 'sent_at': 1700000000},
            {'id': 2, 'content': 'Second', 'sent_at': 1700004000}
        ]
        
        with patch.object(threader_instance, '_save_thread') as mock_save:
            threader_instance.thread_messages(messages)
            assert mock_save.call_count == 2

    def test_get_thread_by_message(self, threader):
        threader_instance, mock_sqlite = threader
        mock_cursor = mock_sqlite.return_value.cursor.return_value
        mock_cursor.fetchone.return_value = {'thread_id': 'thread_123'}
        mock_cursor.fetchall.return_value = [
            MagicMock(keys=lambda: ['thread_id', 'message_id', 'content', 'sent_at', 'position', 'thread_type']), # Mock row object
            MagicMock(keys=lambda: ['thread_id', 'message_id', 'content', 'sent_at', 'position', 'thread_type'])
        ]
        # Configure the mock rows to be dict-addressable
        for i, row in enumerate(mock_cursor.fetchall.return_value):
            row.__getitem__.side_effect = lambda key, i=i: {
                0: {'thread_id': 'thread_123', 'message_id': 1, 'content': 'A', 'sent_at': 1, 'position': 0, 'thread_type': 'exchange'},
                1: {'thread_id': 'thread_123', 'message_id': 2, 'content': 'B', 'sent_at': 2, 'position': 1, 'thread_type': 'exchange'}
            }[i][key]

        thread_data = threader_instance.get_thread_by_message(message_id=1)
        
        assert thread_data is not None
        assert thread_data['thread_id'] == 'thread_123'
        assert len(thread_data['messages']) == 2

    def test_get_thread_by_message_not_found(self, threader):
        threader_instance, mock_sqlite = threader
        mock_cursor = mock_sqlite.return_value.cursor.return_value
        mock_cursor.fetchone.return_value = None
        
        thread_data = threader_instance.get_thread_by_message(message_id=999)
        
        assert thread_data is None

    def test_reply_pattern_detection(self, threader):
        threader_instance, _ = threader
        messages = [
            {'id': 1, 'content': 'Some statement.', 'sent_at': 1700000000},
            {'id': 2, 'content': '@user1 that is interesting', 'sent_at': 1700000010},
            {'id': 3, 'content': '''> > some quote
I agree''', 'sent_at': 1700000020},
        ]
        
        features = threader_instance._extract_features(messages)
        
        reply_feature_column = -1
        assert features[0, reply_feature_column] == 0
        assert features[1, reply_feature_column] > 0
        assert features[2, reply_feature_column] > 0
