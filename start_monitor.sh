#!/bin/bash

# Discord Personal Monitoring Assistant - Quick Start Script

echo "🎯 Discord Personal Monitoring Assistant"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if database exists
if [ ! -f "data/db.sqlite" ]; then
    echo "🆕 No database found. Will create optimized database on first run."
else
    # Check if migration is needed
    echo "🔍 Checking database..."
    python -c "
from lib.database_optimized import OptimizedDatabase
import sqlite3
import sys

try:
    conn = sqlite3.connect('data/db.sqlite')
    cursor = conn.cursor()
    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='message_importance'\")
    if not cursor.fetchone():
        print('📊 Database migration needed...')
        sys.exit(1)
    else:
        print('✅ Database is already optimized!')
    conn.close()
except Exception as e:
    print(f'⚠️ Database check failed: {e}')
    sys.exit(1)
" || {
    echo "🔄 Running database migration..."
    echo "yes" | python migrate_db.py
}
fi

echo ""
echo "🚀 Starting Discord Monitor Dashboard..."
echo "========================================="
echo "📱 Open your browser to: http://localhost:8501"
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run streamlit_monitoring.py