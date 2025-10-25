#!/bin/bash

# Quick Update Script for Discord Bot on Hostinger VPS
# Run this script to update and restart the bot

set -e

echo "🔄 Aktualizace Discord Bota..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

APP_DIR="/home/discord-bot"
SERVICE_NAME="discord-bot"

# Check if running as correct user
if [ "$(whoami)" != "discord-bot" ] && [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠ Spusť jako: sudo -u discord-bot $0 nebo jako root${NC}"
    exit 1
fi

# Change to app directory
cd $APP_DIR

# Pull latest changes
echo "📥 Stahuji aktualizace z git..."
if [ "$(whoami)" = "root" ]; then
    sudo -u discord-bot git pull
else
    git pull
fi

# Activate virtual environment and update dependencies
echo "📦 Aktualizuji Python závislosti..."
if [ "$(whoami)" = "root" ]; then
    sudo -u discord-bot bash -c "source venv/bin/activate && pip install --upgrade -r requirements-production.txt"
else
    source venv/bin/activate
    pip install --upgrade -r requirements-production.txt
fi

# Restart service
if [ "$EUID" -eq 0 ]; then
    echo "♻️  Restartuji službu..."
    systemctl restart $SERVICE_NAME
    echo -e "${GREEN}✓ Bot aktualizován a restartován${NC}"
    
    # Show status
    echo ""
    echo "📊 Status:"
    systemctl status $SERVICE_NAME --no-pager | head -10
else
    echo -e "${YELLOW}⚠ Restart služby vyžaduje root - spusť:${NC}"
    echo "   sudo systemctl restart $SERVICE_NAME"
fi

echo ""
echo -e "${GREEN}🎉 Aktualizace dokončena!${NC}"
echo ""
echo "Užitečné příkazy:"
echo "  • Logy: sudo journalctl -u $SERVICE_NAME -f"
echo "  • Status: sudo systemctl status $SERVICE_NAME"
echo "  • Health: curl http://localhost:8080/health"
