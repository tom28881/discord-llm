import sqlite3
import logging
import sys
from pathlib import Path

# Add the project root to the Python path so we can import from lib
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.database import DB_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('delete_messages')

def delete_messages_from_channels(channel_ids: list):
    """
    Deletes all messages from the specified channel IDs.

    Args:
        channel_ids (list): A list of channel IDs (integers) from which to delete messages.
    """
    if not channel_ids:
        logger.info("No channel IDs provided for deletion.")
        return

    # Remove duplicate channel IDs
    channel_ids = list(set(channel_ids))
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # Delete messages from each channel
        for channel_id in channel_ids:
            cursor.execute('DELETE FROM messages WHERE channel_id = ?', (channel_id,))
            deleted_count = cursor.rowcount
            logger.info(f"Deleted {deleted_count} messages from channel {channel_id}")
        
        conn.commit()
        logger.info("Message deletion completed successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error deleting messages: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    # List of channel IDs to delete messages from
    channel_ids_to_delete = [
        1275178398502883429,
        771479819527258142,
        1203178515231678566,
        415334542090567680,
        1183900751643295784,
        1185398325172777000
    ]
    
    delete_messages_from_channels(channel_ids_to_delete)