#!/bin/bash

# Discord Bot Deployment Script for Hostinger VPS
# This script automates the deployment process

set -e  # Exit on error

echo "🚀 Discord Bot Deployment na Hostinger VPS"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/home/discord-bot"
SERVICE_NAME="discord-bot"
PYTHON_VERSION="3.11"

# Functions
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Prosím spusť tento script jako root (sudo)"
    exit 1
fi

# Step 1: Update system
print_info "Aktualizace systému..."
apt-get update && apt-get upgrade -y
print_status "Systém aktualizován"

# Step 2: Install dependencies
print_info "Instalace závislostí..."
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    sqlite3 \
    gcc \
    g++ \
    supervisor \
    nginx
print_status "Závislosti nainstalovány"

# Step 3: Create application user
print_info "Vytváření uživatele aplikace..."
if ! id -u discord-bot > /dev/null 2>&1; then
    useradd -m -s /bin/bash discord-bot
    print_status "Uživatel discord-bot vytvořen"
else
    print_info "Uživatel discord-bot již existuje"
fi

# Step 4: Create application directory
print_info "Vytváření adresářové struktury..."
mkdir -p $APP_DIR
mkdir -p $APP_DIR/logs
mkdir -p $APP_DIR/data
mkdir -p $APP_DIR/backups
chown -R discord-bot:discord-bot $APP_DIR
print_status "Adresáře vytvořeny"

# Step 5: Clone or update repository
print_info "Stahování aplikace..."
cd $APP_DIR
if [ -d ".git" ]; then
    sudo -u discord-bot git pull
    print_status "Aplikace aktualizována"
else
    print_error "Zkopíruj soubory do $APP_DIR nebo naklonuj git repository"
    exit 1
fi

# Step 6: Create virtual environment
print_info "Vytváření Python virtual environment..."
cd $APP_DIR
if [ ! -d "venv" ]; then
    sudo -u discord-bot python3.11 -m venv venv
    print_status "Virtual environment vytvořen"
else
    print_info "Virtual environment již existuje"
fi

# Step 7: Install Python dependencies
print_info "Instalace Python balíčků..."
sudo -u discord-bot $APP_DIR/venv/bin/pip install --upgrade pip
sudo -u discord-bot $APP_DIR/venv/bin/pip install -r $APP_DIR/requirements-production.txt
print_status "Python balíčky nainstalovány"

# Step 8: Check for .env file
if [ ! -f "$APP_DIR/.env" ]; then
    print_error ".env soubor nenalezen!"
    print_info "Vytvoř .env soubor s následujícím obsahem:"
    echo ""
    cat << EOF
DISCORD_TOKEN=tvuj_discord_token
GOOGLE_API_KEY=tvuj_google_api_key
OPENAI_API_KEY=
OPENROUTER_API_KEY=
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKER_COUNT=4
MEMORY_LIMIT_MB=1024
LLM_DAILY_COST_LIMIT=10.0
EOF
    echo ""
    print_info "Uložit do: $APP_DIR/.env"
    exit 1
else
    print_status ".env soubor nalezen"
    chown discord-bot:discord-bot $APP_DIR/.env
    chmod 600 $APP_DIR/.env
fi

# Step 9: Create systemd service
print_info "Vytváření systemd service..."
cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=Discord LLM Bot
After=network.target

[Service]
Type=simple
User=discord-bot
Group=discord-bot
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/python enhanced_main.py
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/logs/bot.log
StandardError=append:$APP_DIR/logs/error.log

# Resource limits
MemoryLimit=1.5G
CPUQuota=200%

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR/data $APP_DIR/logs $APP_DIR/backups

[Install]
WantedBy=multi-user.target
EOF

print_status "Systemd service vytvořena"

# Step 10: Configure health check endpoint with nginx (optional)
print_info "Konfigurace nginx pro health check..."
cat > /etc/nginx/sites-available/discord-bot << EOF
server {
    listen 80;
    server_name _;
    
    location /health {
        proxy_pass http://localhost:8080/health;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    location /metrics {
        proxy_pass http://localhost:8080/metrics;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        # Optional: Add basic auth here
    }
}
EOF

if [ ! -f "/etc/nginx/sites-enabled/discord-bot" ]; then
    ln -s /etc/nginx/sites-available/discord-bot /etc/nginx/sites-enabled/
fi

# Test nginx configuration
nginx -t && systemctl reload nginx
print_status "Nginx nakonfigurován"

# Step 11: Enable and start service
print_info "Spouštění služby..."
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME
print_status "Služba spuštěna"

# Step 12: Setup log rotation
print_info "Konfigurace log rotation..."
cat > /etc/logrotate.d/discord-bot << EOF
$APP_DIR/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 discord-bot discord-bot
    sharedscripts
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF
print_status "Log rotation nakonfigurována"

# Step 13: Setup automatic updates (optional)
print_info "Vytváření update scriptu..."
cat > $APP_DIR/update.sh << 'EOF'
#!/bin/bash
cd /home/discord-bot
git pull
source venv/bin/activate
pip install -r requirements-production.txt
sudo systemctl restart discord-bot
echo "Bot aktualizován a restartován"
EOF

chmod +x $APP_DIR/update.sh
chown discord-bot:discord-bot $APP_DIR/update.sh
print_status "Update script vytvořen"

# Final status check
echo ""
echo "=========================================="
echo "🎉 Deployment dokončen!"
echo "=========================================="
echo ""
print_info "Status služby:"
systemctl status $SERVICE_NAME --no-pager
echo ""
print_info "Užitečné příkazy:"
echo "  • Status: sudo systemctl status $SERVICE_NAME"
echo "  • Stop: sudo systemctl stop $SERVICE_NAME"
echo "  • Start: sudo systemctl start $SERVICE_NAME"
echo "  • Restart: sudo systemctl restart $SERVICE_NAME"
echo "  • Logs: sudo journalctl -u $SERVICE_NAME -f"
echo "  • Log soubory: $APP_DIR/logs/"
echo "  • Health check: curl http://localhost:8080/health"
echo "  • Update bota: $APP_DIR/update.sh"
echo ""
print_status "Bot běží na pozadí a automaticky se restartuje při pádu"
print_status "Health check dostupný na: http://$(hostname -I | awk '{print $1}')/health"
