import sqlite3
import re
import sys
import argparse
from urllib.parse import urlparse
from collections import defaultdict
from prettytable import PrettyTable
from termcolor import colored
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add the project root to the Python path so we can import from lib
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from lib.database import DB_NAME

def extract_urls(text):
    """Extract URLs from text using regex pattern."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def get_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return url

def format_timestamp(timestamp):
    """Convert Unix timestamp to human-readable format."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S UTC')

def analyze_links(hours=24, min_count=1, server_id=None):
    """
    Analyze links from messages in the database.
    
    Args:
        hours: Number of hours to look back
        min_count: Minimum number of occurrences to include in results
        server_id: Optional server ID to filter messages from a specific server
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Calculate the timestamp for the specified hours ago
    time_threshold = int((datetime.now() - timedelta(hours=hours)).timestamp())

    # Get messages with URLs, optionally filtered by server_id
    query = '''
        SELECT content, sent_at
        FROM messages
        WHERE sent_at >= ?
        AND content LIKE '%http%'
    '''
    params = [time_threshold]
    
    if server_id is not None:
        query += ' AND server_id = ?'
        params.append(server_id)
        
    cursor.execute(query, params)

    # Process results
    domain_counts = defaultdict(int)
    domain_timestamps = defaultdict(list)
    total_urls = 0

    for content, sent_at in cursor.fetchall():
        urls = extract_urls(content)
        for url in urls:
            domain = get_domain(url)
            domain_counts[domain] += 1
            domain_timestamps[domain].append(sent_at)
            total_urls += 1

    conn.close()

    # Filter domains by minimum count
    filtered_domains = {
        domain: count for domain, count in domain_counts.items()
        if count >= min_count
    }

    # Sort domains by count (descending)
    sorted_domains = sorted(
        filtered_domains.items(),
        key=lambda x: (-x[1], x[0].lower())
    )

    # Create table for output
    table = PrettyTable()
    table.field_names = ["Domain", "Count", "First Seen", "Last Seen"]
    table.align = "l"  # Left align text

    for domain, count in sorted_domains:
        timestamps = domain_timestamps[domain]
        first_seen = format_timestamp(min(timestamps))
        last_seen = format_timestamp(max(timestamps))
        
        table.add_row([
            colored(domain, 'cyan'),
            colored(str(count), 'yellow'),
            first_seen,
            last_seen
        ])

    # Print results
    print(f"\nAnalyzed messages from the last {hours} hours")
    print(f"Found {total_urls} total URLs across {len(domain_counts)} unique domains")
    print(f"Showing domains with >= {min_count} occurrences:\n")
    print(table)

def main():
    parser = argparse.ArgumentParser(description='Analyze links from Discord messages')
    parser.add_argument('--hours', type=int, default=24,
                      help='Number of hours to look back (default: 24)')
    parser.add_argument('--min-count', type=int, default=1,
                      help='Minimum number of occurrences to include in results (default: 1)')
    parser.add_argument('--server-id', type=int,
                      help='Filter results to a specific server ID')
    
    args = parser.parse_args()
    
    # Print summary of links
    print(f"\nAnalyzing links from the past {args.hours} hours", end='')
    if args.server_id:
        print(f" for server ID: {args.server_id}")
    else:
        print(" across all servers")
        
    analyze_links(args.hours, args.min_count, args.server_id)

if __name__ == "__main__":
    main()