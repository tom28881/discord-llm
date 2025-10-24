#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the Python path so we can import from lib
sys.path.append(str(Path(__file__).parent.parent))

from lib.database import init_db, get_unique_server_ids, DB_NAME

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('discord_db_manager')

def reset_database():
    """Reset the database by removing all data and reinitializing it."""
    try:
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
            logger.info("Existing database file removed.")
        init_db()
        logger.info("New database initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return False

def show_stats():
    """Show database statistics."""
    try:
        if not os.path.exists(DB_NAME):
            logger.info("Database does not exist.")
            return
            
        server_ids = get_unique_server_ids()
        logger.info(f"Database Statistics:")
        logger.info(f"- Database location: {DB_NAME}")
        logger.info(f"- Database size: {os.path.getsize(DB_NAME) / 1024:.2f} KB")
        logger.info(f"- Number of servers: {len(server_ids)}")
        if server_ids:
            logger.info(f"- Server IDs: {', '.join(map(str, server_ids))}")
    except Exception as e:
        logger.error(f"Error getting database statistics: {e}")

def main():
    parser = argparse.ArgumentParser(description="Discord Database Management Tool")
    parser.add_argument('--reset', action='store_true', help='Reset the database')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    args = parser.parse_args()

    if not (args.reset or args.stats):
        parser.print_help()
        sys.exit(1)

    if args.reset:
        if not reset_database():
            sys.exit(1)

    if args.stats:
        show_stats()

if __name__ == "__main__":
    main()
