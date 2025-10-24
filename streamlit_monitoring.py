"""
Enhanced Streamlit dashboard for Discord monitoring with importance scoring and pattern detection.
Provides a quick overview of what you missed while away.
"""

import streamlit as st
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

# Import our enhanced modules
from lib.database_optimized import OptimizedDatabase, init_optimized_db
from lib.importance_scorer import ImportanceScorer, BatchImportanceScorer
from lib.pattern_detector import PatternDetector
from lib.llm import get_completion

# Import new ML modules
from lib.purchase_predictor import PurchasePredictor
from lib.deadline_tracker import DeadlineTracker
from lib.sentiment_analyzer import SentimentAnalyzer
from lib.conversation_threader import ConversationThreader
from load_messages import load_messages_once

# Page configuration
st.set_page_config(
    page_title="Discord Monitor - Never Miss Important Stuff",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile experience
st.markdown("""
<style>
    .important-message {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .group-activity {
        background-color: #d1ecf1;
        border-left: 4px solid #0dcaf0;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .urgent-alert {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .fomo-alert {
        animation: pulse 2s infinite;
        background-color: #ffeaa7;
        border: 2px solid #fdcb6e;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


class DiscordMonitorDashboard:
    """Enhanced dashboard for Discord monitoring."""
    
    def __init__(self):
        self.db = OptimizedDatabase()
        self.importance_scorer = ImportanceScorer()
        self.batch_scorer = BatchImportanceScorer()
        self.pattern_detector = PatternDetector()
        
        # Initialize new ML modules
        self.purchase_predictor = PurchasePredictor()
        self.deadline_tracker = DeadlineTracker()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conversation_threader = ConversationThreader()
        
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'last_check' not in st.session_state:
            st.session_state.last_check = time.time()
        if 'importance_threshold' not in st.session_state:
            st.session_state.importance_threshold = 0.5
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'viewed_messages' not in st.session_state:
            st.session_state.viewed_messages = set()
    
    def run(self):
        """Run the main dashboard."""
        # Header
        st.title("🎯 Discord Monitor - Personal Assistant")
        st.caption(f"Never miss important group activities, purchases, or decisions")
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content tabs
        tabs = st.tabs([
            "📊 Quick Overview", 
            "🚨 Urgent Alerts",
            "🛒 Purchase Predictions",
            "⏰ Deadlines",
            "😊 Group Mood",
            "👥 Group Activities", 
            "💬 Smart Digest",
            "🔍 Search",
            "⚙️ Preferences"
        ])
        
        with tabs[0]:
            self._render_overview()
        
        with tabs[1]:
            self._render_urgent_alerts()
        
        with tabs[2]:
            self._render_purchase_predictions()
        
        with tabs[3]:
            self._render_deadlines()
        
        with tabs[4]:
            self._render_group_mood()
        
        with tabs[5]:
            self._render_group_activities()
        
        with tabs[6]:
            self._render_smart_digest()
        
        with tabs[7]:
            self._render_search()
        
        with tabs[8]:
            self._render_preferences()
        
        # Auto-refresh if enabled
        if st.session_state.auto_refresh:
            time.sleep(30)
            st.rerun()
    
    def _render_sidebar(self):
        """Render sidebar configuration."""
        with st.sidebar:
            st.header("⚡ Quick Controls")
            
            # Server selection
            servers = self.db.get_servers()
            if servers:
                server_names = {s[0]: s[1] for s in servers}
                selected_server = st.selectbox(
                    "Vyber server",
                    options=list(server_names.keys()),
                    format_func=lambda x: server_names[x],
                    key="selected_server"
                )
                st.session_state.server_id = selected_server

                if st.button("⬇️ Stáhnout nové zprávy", use_container_width=True):
                    with st.spinner("Stahuji zprávy z Discordu..."):
                        try:
                            summary = load_messages_once(
                                server_id=str(selected_server),
                                sleep_between_servers=False,
                                sleep_between_channels=False
                            )
                            summary["ran_at"] = time.time()
                            st.session_state.last_import_summary = summary
                            total_saved = summary.get("messages_saved", 0)
                            st.success(f"Staženo {total_saved} nových zpráv.")
                        except Exception as exc:
                            st.error(f"Chyba při stahování zpráv: {exc}")

                last_summary = st.session_state.get("last_import_summary")
                if last_summary:
                    ran_at = datetime.fromtimestamp(last_summary.get("ran_at", time.time()))
                    total_saved = last_summary.get("messages_saved", 0)
                    st.caption(
                        f"Naposledy staženo {ran_at.strftime('%d.%m.%Y %H:%M:%S')} · {total_saved} nových zpráv"
                    )
            else:
                st.warning("No servers found. Run `python load_messages.py` first.")
                st.session_state.server_id = None
                return

            
            # Time range
            time_range = st.slider(
                "Time Range (hours)",
                min_value=1,
                max_value=168,
                value=72,  # Default to 72 hours (3 days) for more meaningful results
                key="time_range"
            )
            
            # Importance threshold
            st.session_state.importance_threshold = st.slider(
                "Importance Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Only show messages above this importance score"
            )
            
            # Auto-refresh toggle
            st.session_state.auto_refresh = st.checkbox(
                "Auto-refresh (30s)",
                value=st.session_state.auto_refresh
            )
            
            # Quick stats
            st.divider()
            st.subheader("📈 Quick Stats")
            
            if st.session_state.server_id:
                stats = self._get_quick_stats(st.session_state.server_id, time_range)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Messages", stats['total_messages'])
                    st.metric("Active Users", stats['unique_authors'])
                with col2:
                    st.metric("Important", stats['important_count'])
                    st.metric("Group Activities", stats['activity_count'])
    
    def _render_overview(self):
        """Render the main overview dashboard."""
        if not st.session_state.server_id:
            st.info("Please select a server from the sidebar.")
            return
        
        st.header("📊 While You Were Away...")
        
        # Time since last check
        time_since = (time.time() - st.session_state.last_check) / 3600
        st.info(f"⏰ Last checked: {time_since:.1f} hours ago")
        
        # Get digest summary
        digest = self.db.get_digest_summary(
            st.session_state.server_id,
            st.session_state.time_range,
            st.session_state.importance_threshold
        )
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🔥 Important Messages</h3>
                <h1>{}</h1>
            </div>
            """.format(len(digest['important_messages'])), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>👥 Group Activities</h3>
                <h1>{}</h1>
            </div>
            """.format(len(digest['group_activities'])), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>💬 Total Messages</h3>
                <h1>{}</h1>
            </div>
            """.format(digest['statistics']['total_messages']), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>👤 Active Users</h3>
                <h1>{}</h1>
            </div>
            """.format(digest['statistics']['unique_authors']), unsafe_allow_html=True)
        
        # FOMO Alert if detected
        fomo_activities = [a for a in digest['group_activities'] if a['type'] == 'fomo_moment']
        if fomo_activities:
            st.markdown("""
            <div class="fomo-alert">
                <h3>⚡ FOMO ALERT!</h3>
                <p>High activity detected - check Group Activities tab immediately!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Activity timeline
        st.subheader("📈 Activity Timeline")
        self._render_activity_timeline(digest)
        
        # Top keywords
        if digest['top_keywords']:
            st.subheader("🔑 Trending Topics")
            keyword_cols = st.columns(min(5, len(digest['top_keywords'])))
            for i, keyword in enumerate(digest['top_keywords'][:5]):
                with keyword_cols[i]:
                    st.info(f"**{keyword}**")
        
        # Mark as checked
        if st.button("✅ Mark All as Read", type="primary"):
            st.session_state.last_check = time.time()
            st.session_state.viewed_messages.update(
                msg['id'] for msg in digest['important_messages']
            )
            st.success("All messages marked as read!")
            st.rerun()
    
    def _render_urgent_alerts(self):
        """Render urgent alerts and high-importance messages."""
        if not st.session_state.server_id:
            st.info("Please select a server from the sidebar.")
            return
        
        st.header("🚨 Urgent Alerts")
        
        # Get high importance messages
        urgent_messages = self.db.get_messages_by_importance(
            st.session_state.server_id,
            hours=st.session_state.time_range,
            min_importance=0.8  # Only very important
        )
        
        if not urgent_messages:
            st.success("✅ No urgent messages - you're all caught up!")
            return
        
        # Group by urgency level
        critical = [m for m in urgent_messages if (m.get('urgency_level') or 0) >= 4]
        high = [m for m in urgent_messages if 2 <= (m.get('urgency_level') or 0) < 4]
        
        if critical:
            st.subheader(f"🔴 Critical ({len(critical)})")
            for msg in critical[:5]:
                self._render_urgent_message(msg)
        
        if high:
            st.subheader(f"🟡 High Priority ({len(high)})")
            for msg in high[:5]:
                self._render_important_message(msg)
    
    def _render_group_activities(self):
        """Render detected group activities."""
        if not st.session_state.server_id:
            st.info("Please select a server from the sidebar.")
            return
        
        st.header("👥 Group Activities")
        
        # Get recent messages for pattern detection
        messages = self.db.get_messages_by_importance(
            st.session_state.server_id,
            hours=st.session_state.time_range,
            min_importance=0.0  # Get all for pattern detection
        )
        
        # Detect activities
        activities = self.pattern_detector.detect_activities(messages)
        
        if not activities:
            st.info("No group activities detected in the selected time range.")
            return
        
        # Group by type
        purchases = [a for a in activities if a.activity_type == 'group_purchase']
        events = [a for a in activities if a.activity_type == 'event_planning']
        decisions = [a for a in activities if a.activity_type == 'group_decision']
        fomo = [a for a in activities if a.activity_type == 'fomo_moment']
        
        # Render each type
        if purchases:
            st.subheader(f"🛒 Group Purchases ({len(purchases)})")
            for activity in purchases:
                self._render_activity(activity)
        
        if events:
            st.subheader(f"📅 Events ({len(events)})")
            for activity in events:
                self._render_activity(activity)
        
        if decisions:
            st.subheader(f"🗳️ Decisions ({len(decisions)})")
            for activity in decisions:
                self._render_activity(activity)
        
        if fomo:
            st.subheader(f"⚡ FOMO Moments ({len(fomo)})")
            for activity in fomo:
                self._render_activity(activity, is_fomo=True)
    
    def _render_smart_digest(self):
        """Render AI-powered smart digest."""
        if not st.session_state.server_id:
            st.info("Please select a server from the sidebar.")
            return
        
        st.header("💬 Smart Digest")
        
        # Get important messages
        messages = self.db.get_messages_by_importance(
            st.session_state.server_id,
            hours=st.session_state.time_range,
            min_importance=st.session_state.importance_threshold
        )
        
        if not messages:
            st.info("No messages above the importance threshold.")
            return
        
        # Generate AI summary
        if st.button("🤖 Generate AI Summary"):
            with st.spinner("Analyzing messages..."):
                summary = self._generate_ai_summary(messages[:50])  # Top 50 messages
                
                st.markdown("### 📝 AI-Generated Summary")
                st.write(summary)
        
        # Expandable message groups by channel
        st.subheader("Messages by Channel")
        
        # Group messages by channel
        by_channel = {}
        for msg in messages:
            channel_id = msg.get('channel_id')
            if channel_id not in by_channel:
                by_channel[channel_id] = []
            by_channel[channel_id].append(msg)
        
        # Render each channel
        for channel_id, channel_messages in by_channel.items():
            channels = self.db.get_channels(st.session_state.server_id)
            channel_name = next((c[1] for c in channels if c[0] == channel_id), f"Channel {channel_id}")
            
            with st.expander(f"#{channel_name} ({len(channel_messages)} messages)"):
                for msg in channel_messages[:10]:  # Show top 10 per channel
                    self._render_message_compact(msg)
    
    def _render_search(self):
        """Render intelligent search interface."""
        if not st.session_state.server_id:
            st.info("Please select a server from the sidebar.")
            return
        
        st.header("🔍 Intelligent Search")
        
        # Search input
        query = st.text_input(
            "Search messages",
            placeholder="Enter keywords, mentions, or topics...",
            key="search_query"
        )
        
        if query:
            with st.spinner("Searching..."):
                results = self.db.search_messages_intelligent(
                    query,
                    server_id=st.session_state.server_id,
                    hours=st.session_state.time_range
                )
                
                if results:
                    st.success(f"Found {len(results)} matching messages")
                    
                    for msg in results[:20]:  # Show top 20 results
                        relevance = msg.get('relevance_score', 0)
                        importance = msg.get('importance_score', 0)
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            self._render_message_compact(msg)
                        with col2:
                            st.metric("Relevance", f"{abs(relevance):.2f}")
                        with col3:
                            st.metric("Importance", f"{importance:.2f}")
                else:
                    st.info("No matching messages found.")
    
    def _render_preferences(self):
        """Render user preferences configuration."""
        st.header("⚙️ Preferences")
        
        st.subheader("🎯 Keyword Importance")
        st.info("Add keywords that are important to you. Messages containing these keywords will be scored higher.")
        
        # Current preferences
        preferences = self.db.get_user_preferences()
        
        if preferences:
            st.write("Current Keywords:")
            for pref in preferences:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.text(pref['keyword'])
                with col2:
                    st.progress(pref['importance_weight'])
                with col3:
                    if st.button("❌", key=f"del_{pref['id']}"):
                        # Delete preference (would need to implement)
                        pass
        
        # Add new keyword
        st.divider()
        st.subheader("Add New Keyword")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_keyword = st.text_input("Keyword", key="new_keyword")
        with col2:
            new_weight = st.slider("Importance", 0.0, 1.0, 0.7, key="new_weight")
        
        if st.button("➕ Add Keyword", type="primary"):
            if new_keyword:
                self.db.update_user_preference(new_keyword, new_weight)
                st.success(f"Added keyword: {new_keyword}")
                st.rerun()
    
    # Helper methods for rendering
    
    def _render_urgent_message(self, message: Dict[str, Any]):
        """Render an urgent message."""
        st.markdown(f"""
        <div class="urgent-alert">
            <strong>🔴 URGENT - Score: {message.get('importance_score', 0):.2f}</strong><br>
            <small>{self._format_timestamp(message.get('sent_at'))}</small><br>
            {message.get('content', '')[:200]}...
        </div>
        """, unsafe_allow_html=True)
    
    def _render_important_message(self, message: Dict[str, Any]):
        """Render an important message."""
        st.markdown(f"""
        <div class="important-message">
            <strong>⚠️ Important - Score: {message.get('importance_score', 0):.2f}</strong><br>
            <small>{self._format_timestamp(message.get('sent_at'))}</small><br>
            {message.get('content', '')[:200]}...
        </div>
        """, unsafe_allow_html=True)
    
    def _render_activity(self, activity, is_fomo: bool = False):
        """Render a group activity."""
        summary = self.pattern_detector.get_activity_summary(activity)
        
        css_class = "fomo-alert" if is_fomo else "group-activity"
        
        st.markdown(f"""
        <div class="{css_class}">
            <strong>{summary}</strong><br>
            <small>Duration: {activity.duration_minutes:.0f} minutes | 
            Participants: {activity.participant_count}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Show key messages if expanded
        if st.checkbox(f"Show details", key=f"activity_{activity.start_time}"):
            for msg in activity.key_messages[:3]:
                st.text(f"  - {msg.get('content', '')[:100]}...")
    
    def _render_message_compact(self, message: Dict[str, Any]):
        """Render a message in compact format."""
        is_new = message.get('id') not in st.session_state.viewed_messages
        
        icon = "🆕" if is_new else ""
        author = message.get('author_name', 'Unknown')
        content = message.get('content', '')[:150]
        timestamp = self._format_timestamp(message.get('sent_at'))
        
        st.markdown(f"{icon} **{author}** - {timestamp}")
        st.text(content + ("..." if len(message.get('content', '')) > 150 else ""))
    
    def _render_activity_timeline(self, digest: Dict[str, Any]):
        """Render activity timeline chart."""
        # Prepare data for timeline
        messages = digest['important_messages']
        if not messages:
            st.info("No activity data for timeline.")
            return
        
        # Create hourly buckets
        df_data = []
        for msg in messages:
            timestamp = msg.get('sent_at', 0)
            dt = datetime.fromtimestamp(timestamp)
            df_data.append({
                'time': dt,
                'hour': dt.hour,
                'importance': msg.get('importance_score', 0)
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Create timeline chart
            fig = px.scatter(
                df, 
                x='time', 
                y='importance',
                title='Message Importance Over Time',
                labels={'importance': 'Importance Score', 'time': 'Time'},
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _get_quick_stats(self, server_id: int, hours: int) -> Dict[str, int]:
        """Get quick statistics for sidebar."""
        digest = self.db.get_digest_summary(server_id, hours, 0.5)
        
        return {
            'total_messages': digest['statistics'].get('total_messages', 0),
            'unique_authors': digest['statistics'].get('unique_authors', 0),
            'important_count': len(digest.get('important_messages', [])),
            'activity_count': len(digest.get('group_activities', []))
        }
    
    def _generate_ai_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate AI summary of messages."""
        # Prepare message text
        message_text = "\n".join([
            f"{msg.get('author_name', 'Unknown')}: {msg.get('content', '')[:200]}"
            for msg in messages[:30]  # Limit to prevent token overflow
        ])
        
        prompt = f"""
        Summarize the following Discord messages, focusing on:
        1. Important decisions or announcements
        2. Group activities (purchases, events, meetings)
        3. Action items or things requiring response
        4. General discussion topics
        
        Messages:
        {message_text}
        
        Provide a concise summary in bullet points.
        """
        
        try:
            summary = get_completion(prompt)
            return summary
        except Exception as e:
            return f"Could not generate summary: {str(e)}"
    
    def _format_timestamp(self, timestamp: Optional[int]) -> str:
        """Format timestamp for display."""
        if not timestamp:
            return "Unknown time"
        
        dt = datetime.fromtimestamp(timestamp)
        now = datetime.now()
        
        # Today/yesterday logic
        if dt.date() == now.date():
            return f"Today {dt.strftime('%I:%M %p')}"
        elif dt.date() == (now - timedelta(days=1)).date():
            return f"Yesterday {dt.strftime('%I:%M %p')}"
        else:
            return dt.strftime('%b %d %I:%M %p')
    
    def _render_purchase_predictions(self):
        """Render purchase predictions tab."""
        st.header("🛒 Purchase Predictions")
        st.caption("AI-detected group purchases and buying opportunities")
        
        if not st.session_state.server_id:
            st.info("Please select a server from the sidebar.")
            return
        
        # Get recent predictions from database
        predictions = self.purchase_predictor.get_recent_predictions(hours=st.session_state.time_range * 2, min_probability=0.6)
        
        # If no saved predictions, analyze recent messages directly
        if not predictions:
            st.info("Analyzing recent messages for purchase opportunities...")
            
            # Get recent messages for analysis
            recent_messages = self.db.get_messages_by_importance(
                st.session_state.server_id,
                hours=st.session_state.time_range,
                min_importance=0.0  # Get all messages for pattern detection
            )
            
            if recent_messages:
                # Analyze messages for purchase patterns
                prediction = self.purchase_predictor.predict_purchase(recent_messages[:100])  # Limit to 100 recent messages
                
                if prediction['probability'] > 0.5:
                    # Display the prediction
                    probability = prediction['probability']
                    metadata = prediction.get('metadata', {})
                    
                    # Color code by probability
                    if probability > 0.8:
                        alert_class = "urgent-alert"
                        icon = "🔴"
                    elif probability > 0.7:
                        alert_class = "group-activity"
                        icon = "🟡"
                    else:
                        alert_class = "important-message"
                        icon = "🟢"
                    
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <strong>{icon} Purchase Opportunity - {probability:.0%} confidence</strong><br>
                        <small>Type: {prediction.get('prediction_type', 'unknown')}</small><br>
                        Based on analysis of {len(recent_messages)} recent messages<br>
                        <small>💰 Price mentions: {', '.join(metadata.get('price_mentions', ['Not specified']))}</small><br>
                        <small>📦 Items: {', '.join(metadata.get('purchase_items', ['Various']))}</small><br>
                        <small>⏰ Urgency: {'⚡' * metadata.get('urgency_level', 0)}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No purchase opportunities detected in recent messages")
            else:
                st.info("No messages found in selected time range")
            return
        
        # Display predictions
        for pred in predictions:
            probability = pred.get('probability', 0)
            metadata = pred.get('metadata', {})
            
            # Color code by probability
            if probability > 0.8:
                alert_class = "urgent-alert"
                icon = "🔴"
            elif probability > 0.7:
                alert_class = "group-activity"
                icon = "🟡"
            else:
                alert_class = "important-message"
                icon = "🟢"
            
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{icon} Purchase Opportunity - {probability:.0%} confidence</strong><br>
                <small>Type: {pred.get('prediction_type', 'unknown')}</small><br>
                {pred.get('content', '')[:200]}...<br>
                <small>💰 Price: {', '.join(metadata.get('price_mentions', ['Not specified']))}</small><br>
                <small>📦 Items: {', '.join(metadata.get('purchase_items', ['Various']))}</small><br>
                <small>⏰ Urgency: {'⚡' * metadata.get('urgency_level', 0)}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_deadlines(self):
        """Render deadlines tab."""
        st.header("⏰ Upcoming Deadlines")
        st.caption("Never miss important dates and time-sensitive opportunities")
        
        if not st.session_state.server_id:
            st.info("Please select a server from the sidebar.")
            return
        
        # Get upcoming deadlines
        deadlines = self.deadline_tracker.get_upcoming_deadlines(hours_ahead=72)
        
        # If no stored deadlines, analyze recent messages for deadline patterns
        if not deadlines:
            st.info("Analyzing recent messages for deadlines...")
            
            # Get recent messages for analysis
            recent_messages = self.db.get_messages_by_importance(
                st.session_state.server_id,
                hours=st.session_state.time_range,
                min_importance=0.0
            )
            
            if recent_messages:
                # Extract deadlines from recent messages
                extracted_deadlines = self.deadline_tracker.extract_deadlines(recent_messages[:100])
                
                if extracted_deadlines:
                    st.subheader("📅 Potential Deadlines Found")
                    for deadline in extracted_deadlines[:5]:  # Show top 5
                        st.markdown(f"""
                        <div class="important-message">
                            <strong>📆 {deadline.get('deadline_text', 'Deadline')}</strong><br>
                            <small>📅 {deadline.get('deadline_datetime', 'Date unclear')}</small><br>
                            <small>⚡ Confidence: {deadline.get('confidence', 0):.0%}</small><br>
                            <small>Message: {deadline.get('source_message', '')[:150]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No deadlines found in recent messages")
            else:
                st.info("No messages found in selected time range")
            return
        
        # Group by urgency
        urgent = [d for d in deadlines if d.get('urgency_level', 0) >= 4]
        normal = [d for d in deadlines if d.get('urgency_level', 0) < 4]
        
        if urgent:
            st.subheader("🚨 Urgent Deadlines")
            for deadline in urgent:
                st.markdown(f"""
                <div class="urgent-alert">
                    <strong>⏰ {deadline.get('deadline_text', 'Deadline')}</strong><br>
                    <small>📅 {deadline.get('deadline_datetime', 'Unknown time')}</small><br>
                    <small>⏳ {deadline.get('time_remaining', 'Time remaining unknown')}</small><br>
                    <small>Message: {deadline.get('content', '')[:150]}...</small>
                </div>
                """, unsafe_allow_html=True)
        
        if normal:
            st.subheader("📅 Regular Deadlines")
            for deadline in normal:
                st.markdown(f"""
                <div class="important-message">
                    <strong>📆 {deadline.get('deadline_text', 'Deadline')}</strong><br>
                    <small>📅 {deadline.get('deadline_datetime', 'Unknown time')}</small><br>
                    <small>⏳ {deadline.get('time_remaining', 'Time remaining unknown')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_group_mood(self):
        """Render group mood and sentiment analysis."""
        st.header("😊 Group Mood & Excitement")
        st.caption("Track the emotional pulse of your Discord communities")
        
        # Get channel mood for each channel
        if hasattr(st.session_state, 'server_id') and st.session_state.server_id:
            channels = self.db.get_channels(st.session_state.server_id)
            
            if channels:
                # Create columns for mood cards
                cols = st.columns(min(3, len(channels)))
                
                for idx, channel in enumerate(channels[:6]):  # Limit to 6 channels
                    channel_info = dict(channel) if hasattr(channel, 'keys') else {'id': channel[0], 'name': channel[1]}
                    channel_id = channel_info['id']
                    
                    # Get mood data from database first, then calculate on-the-fly if empty
                    mood_data = self.sentiment_analyzer.get_channel_mood(channel_id, hours=st.session_state.time_range)
                    
                    # If no mood data in database, analyze recent messages
                    if not mood_data or mood_data.get('activity_level', 0) == 0:
                        # Get recent messages from this channel
                        with self.db.pool.get_connection() as conn:
                            cursor = conn.cursor()
                            time_threshold = int((datetime.now() - timedelta(hours=st.session_state.time_range)).timestamp())
                            cursor.execute('''
                                SELECT COUNT(*) as count 
                                FROM messages 
                                WHERE channel_id = ? AND sent_at >= ?
                            ''', (channel_id, time_threshold))
                            msg_count = cursor.fetchone()[0]
                        
                        # Estimate mood based on activity level
                        if msg_count > 50:
                            mood = 'active'
                            excitement = 0.6
                        elif msg_count > 20:
                            mood = 'positive'
                            excitement = 0.4
                        elif msg_count > 5:
                            mood = 'neutral'
                            excitement = 0.2
                        else:
                            mood = 'quiet'
                            excitement = 0.1
                        
                        mood_data = {
                            'mood': mood,
                            'sentiment': {'excitement_level': excitement},
                            'activity_level': msg_count
                        }
                    
                    with cols[idx % 3]:
                        mood = mood_data.get('mood', 'neutral')
                        sentiment = mood_data.get('sentiment', {})
                        
                        # Mood emoji mapping
                        mood_emojis = {
                            'hyped': '🔥',
                            'active': '⚡',
                            'positive': '😊',
                            'neutral': '😐',
                            'tense': '😟',
                            'quiet': '😴'
                        }
                        
                        # Mood colors
                        mood_colors = {
                            'hyped': '#ff6b6b',
                            'active': '#4ecdc4',
                            'positive': '#95e77e',
                            'neutral': '#f7dc6f',
                            'tense': '#bb8fce',
                            'quiet': '#85929e'
                        }
                        
                        st.markdown(f"""
                        <div style="background: {mood_colors.get(mood, '#f0f0f0')}; 
                                    padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <h4>{mood_emojis.get(mood, '😐')} #{channel_info['name']}</h4>
                            <p><strong>Mood:</strong> {mood.capitalize()}</p>
                            <p><strong>Excitement:</strong> {'🔥' * int(sentiment.get('excitement_level', 0) * 5)}</p>
                            <p><strong>Activity:</strong> {mood_data.get('activity_level', 0)} messages</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Show excitement peaks
            st.subheader("🎉 Recent Excitement Peaks")
            peaks = self.sentiment_analyzer.get_excitement_peaks(hours=st.session_state.time_range)
            
            if peaks:
                for peak in peaks[:5]:
                    st.markdown(f"""
                    <div class="fomo-alert">
                        <strong>🔥 High Energy Message!</strong><br>
                        <small>Excitement: {'⚡' * int(peak.get('excitement_level', 0) * 5)}</small><br>
                        {peak.get('content', '')[:200]}...<br>
                        <small>{peak.get('time_ago', '')}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Find high-activity messages as excitement peaks
                recent_messages = self.db.get_messages_by_importance(
                    st.session_state.server_id,
                    hours=st.session_state.time_range,
                    min_importance=0.7
                )
                
                if recent_messages and len(recent_messages) > 0:
                    st.subheader("📈 High-Importance Messages")
                    for msg in recent_messages[:3]:
                        importance_value = msg.get('importance_score')
                        if isinstance(importance_value, (int, float)):
                            importance_display = f"{importance_value:.1f}"
                        else:
                            importance_display = "N/A"
                        st.markdown(f"""
                        <div class="fomo-alert">
                            <strong>🎯 Important Message ({importance_display} score)!</strong><br>
                            {msg.get('content', '')[:200]}...<br>
                            <small>{self._format_timestamp(msg.get('sent_at'))}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No excitement peaks detected recently")
        else:
            st.warning("Please select a server from the sidebar")


# Main execution
if __name__ == "__main__":
    # Initialize database if needed
    init_optimized_db()
    
    # Run dashboard
    dashboard = DiscordMonitorDashboard()
    dashboard.run()