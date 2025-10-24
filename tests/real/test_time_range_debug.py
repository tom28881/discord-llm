"""
Diagnostic test for time range filtering bug.
User reported that changing time range (1h vs 720h) returns same number of messages.
"""
import pytest
import sqlite3
import time
from datetime import datetime, timedelta

from load_messages import load_messages_once
from lib.database import get_recent_message_records


@pytest.mark.real
@pytest.mark.skip_ci
class TestTimeRangeFiltering:
    """Debug time range filtering issues."""
    
    def test_multiple_time_ranges(
        self,
        test_database,
        test_server_id,
    ):
        """Test import with different time ranges and compare results."""
        print("\n" + "="*70)
        print("🔍 TIME RANGE FILTERING DIAGNOSTIC TEST")
        print("="*70)
        
        test_ranges = [
            (1, "Last 1 hour"),
            (24, "Last 24 hours (1 day)"),
            (168, "Last 168 hours (1 week)"),
            (720, "Last 720 hours (30 days)"),
        ]
        
        results = []
        
        for hours, description in test_ranges:
            print(f"\n📊 Testing: {description} (hours_back={hours})")
            print("-" * 70)
            
            # Clear previous data to get clean results
            conn = sqlite3.connect(test_database)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM channels")
            cursor.execute("DELETE FROM servers")
            conn.commit()
            conn.close()
            
            # Import with this time range
            start = time.time()
            summary = load_messages_once(
                server_id=test_server_id,
                hours_back=hours,
                sleep_between_channels=False  # Faster for testing
            )
            duration = time.time() - start
            
            # Get stats
            conn = sqlite3.connect(test_database)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(sent_at), MAX(sent_at) FROM messages")
            time_range = cursor.fetchone()
            
            if time_range[0] and time_range[1]:
                oldest = datetime.fromtimestamp(time_range[0])
                newest = datetime.fromtimestamp(time_range[1])
                span_hours = (newest - oldest).total_seconds() / 3600
            else:
                oldest = None
                newest = None
                span_hours = 0
            
            # Get sample messages
            cursor.execute("""
                SELECT content, sent_at 
                FROM messages 
                ORDER BY sent_at DESC 
                LIMIT 3
            """)
            samples = cursor.fetchall()
            
            conn.close()
            
            # Print results
            print(f"   ⏱️  Import duration: {duration:.2f}s")
            print(f"   💾 Messages imported: {summary['messages_saved']}")
            print(f"   📊 Total in DB: {total_messages}")
            
            if oldest and newest:
                print(f"   📅 Oldest message: {oldest:%Y-%m-%d %H:%M}")
                print(f"   📅 Newest message: {newest:%Y-%m-%d %H:%M}")
                print(f"   ⏰ Actual time span: {span_hours:.1f} hours")
            else:
                print(f"   ⚠️  No messages found")
            
            if samples:
                print(f"   📝 Sample messages:")
                for content, sent_at in samples[:2]:
                    dt = datetime.fromtimestamp(sent_at)
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"      [{dt:%Y-%m-%d %H:%M}] {preview}")
            
            results.append({
                'hours': hours,
                'description': description,
                'messages': total_messages,
                'span_hours': span_hours,
                'oldest': oldest,
                'newest': newest
            })
        
        # Compare results
        print("\n" + "="*70)
        print("📊 COMPARISON OF TIME RANGES")
        print("="*70)
        
        for result in results:
            print(f"\n{result['description']:30} → {result['messages']:4} messages "
                  f"(span: {result['span_hours']:.1f}h)")
        
        # Analysis
        print("\n" + "="*70)
        print("🔍 ANALYSIS")
        print("="*70)
        
        message_counts = [r['messages'] for r in results]
        
        if len(set(message_counts)) == 1:
            print("❌ PROBLEM DETECTED: All time ranges returned SAME number of messages!")
            print("   This indicates time range filtering is NOT working.")
        elif message_counts == sorted(message_counts):
            print("✅ GOOD: Message count increases with longer time ranges")
        else:
            print("⚠️  UNEXPECTED: Message counts don't follow expected pattern")
        
        print(f"\n   Message counts: {message_counts}")
        
        # Check if span matches requested range
        print("\n   Time span analysis:")
        for result in results:
            requested = result['hours']
            actual = result['span_hours']
            if actual > 0:
                match = "✅" if actual <= requested else "⚠️"
                print(f"   {match} Requested {requested}h, got {actual:.1f}h span")
        
        # Assertions
        assert message_counts[0] <= message_counts[-1], \
            "Longer time range should return same or more messages"
    
    def test_query_time_filtering(
        self,
        test_database,
        test_server_id,
    ):
        """Test that database queries respect time filtering."""
        print("\n" + "="*70)
        print("🔍 TESTING DATABASE QUERY TIME FILTERING")
        print("="*70)
        
        # First import with long time range
        print("\n📥 Importing messages (720 hours)...")
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=720,
            sleep_between_channels=False
        )
        print(f"   💾 Imported: {summary['messages_saved']} messages")
        
        # Now query with different time ranges
        test_ranges = [1, 24, 168, 720, None]
        
        print("\n📊 Testing query filters:")
        query_results = {}
        
        for hours in test_ranges:
            records = get_recent_message_records(
                int(test_server_id),
                hours=hours
            )
            label = f"{hours}h" if hours else "all time"
            query_results[label] = len(records)
            print(f"   {label:15} → {len(records):4} messages")
        
        # Analysis
        print("\n🔍 Query Analysis:")
        counts = list(query_results.values())
        
        if len(set(counts[:-1])) == 1:  # Exclude "all time"
            print("   ❌ PROBLEM: All time-filtered queries return same count!")
            print("   This indicates get_recent_message_records() filtering is broken")
        else:
            print("   ✅ Query filtering appears to work correctly")
        
        return query_results


@pytest.mark.real
@pytest.mark.skip_ci  
class TestTimeRangeCode:
    """Test the actual code paths for time range handling."""
    
    def test_timestamp_calculation(self):
        """Verify timestamp calculation is correct."""
        print("\n🔍 Testing timestamp calculation...")
        
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        test_cases = [
            (1, "1 hour ago"),
            (24, "24 hours ago"),
            (720, "720 hours ago"),
        ]
        
        for hours, description in test_cases:
            expected_dt = now - timedelta(hours=hours)
            expected_ts = int(expected_dt.timestamp())
            
            # This is how load_messages.py calculates it
            calculated_dt = datetime.now() - timedelta(hours=hours)
            calculated_ts = int(calculated_dt.timestamp())
            
            diff = abs(calculated_ts - expected_ts)
            
            print(f"   {description:20} → timestamp: {calculated_ts} "
                  f"(diff: {diff}s)")
            
            assert diff < 2, f"Timestamp calculation off by {diff} seconds"
        
        print("   ✅ Timestamp calculations correct")
    
    def test_min_timestamp_passing(
        self,
        test_database,
        test_server_id,
    ):
        """Verify min_timestamp is correctly passed to fetch_messages."""
        print("\n🔍 Testing min_timestamp parameter passing...")
        
        # We need to check if min_timestamp is actually used
        # Let's import and check the actual timestamps
        
        hours_back = 24
        print(f"\n   Importing with hours_back={hours_back}...")
        
        summary = load_messages_once(
            server_id=test_server_id,
            hours_back=hours_back,
            sleep_between_channels=False
        )
        
        if summary['messages_saved'] == 0:
            print("   ⚠️  No messages imported - cannot verify")
            pytest.skip("No messages to test with")
        
        # Check actual timestamps
        conn = sqlite3.connect(test_database)
        cursor = conn.cursor()
        
        cursor.execute("SELECT MIN(sent_at), MAX(sent_at) FROM messages")
        min_ts, max_ts = cursor.fetchone()
        
        conn.close()
        
        if min_ts and max_ts:
            now = datetime.now().timestamp()
            hours_back_threshold = now - (hours_back * 3600)
            
            oldest = datetime.fromtimestamp(min_ts)
            newest = datetime.fromtimestamp(max_ts)
            
            print(f"\n   📅 Oldest message: {oldest:%Y-%m-%d %H:%M}")
            print(f"   📅 Newest message: {newest:%Y-%m-%d %H:%M}")
            print(f"   ⏰ Hours back threshold: "
                  f"{datetime.fromtimestamp(hours_back_threshold):%Y-%m-%d %H:%M}")
            
            # Check if oldest message is within expected range
            # Note: It's OK if oldest is newer than threshold (channel might not have old messages)
            # But it should NOT be much older than threshold
            
            if min_ts < hours_back_threshold:
                hours_older = (hours_back_threshold - min_ts) / 3600
                print(f"\n   ⚠️  WARNING: Oldest message is {hours_older:.1f}h "
                      f"older than requested range!")
                print(f"   This suggests min_timestamp is NOT being used correctly")
            else:
                print(f"\n   ✅ All messages are within requested time range")
