#!/bin/bash

# One-Line Installer for Discord Bot on Hostinger VPS
# Usage: curl -sSL https://raw.githubusercontent.com/tom28881/discord-llm/main/install.sh | sudo bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

print_success() {
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
    print_error "Tento script musí běžet jako root"
    echo "Spusť: curl -sSL https://raw.githubusercontent.com/tom28881/discord-llm/main/install.sh | sudo bash"
    exit 1
fi

print_header "Discord Bot Installer pro Hostinger VPS"

# Configuration
REPO_URL="https://github.com/tom28881/discord-llm.git"
APP_DIR="/home/discord-bot"
GITHUB_RAW="https://raw.githubusercontent.com/tom28881/discord-llm/main"

print_info "Kontrola systému..."

# Step 1: Update system
print_info "Aktualizace systému..."
apt-get update -qq
print_success "Systém aktualizován"

# Step 2: Install dependencies
print_info "Instalace závislostí..."
apt-get install -y -qq git python3.11 python3.11-venv python3-pip curl sqlite3 gcc g++ > /dev/null 2>&1
print_success "Závislosti nainstalovány"

# Step 3: Create user
print_info "Vytváření uživatele..."
if ! id -u discord-bot > /dev/null 2>&1; then
    useradd -m -s /bin/bash discord-bot
    print_success "Uživatel discord-bot vytvořen"
else
    print_info "Uživatel discord-bot již existuje"
fi

# Step 4: Clone repository
print_info "Stahování projektu z GitHubu..."
if [ -d "$APP_DIR/.git" ]; then
    print_info "Projekt již existuje, aktualizuji..."
    cd $APP_DIR
    sudo -u discord-bot git pull -q
else
    rm -rf $APP_DIR
    sudo -u discord-bot git clone -q $REPO_URL $APP_DIR
fi
cd $APP_DIR
print_success "Projekt stažen"

# Step 5: Create directories
print_info "Vytváření adresářů..."
mkdir -p $APP_DIR/{logs,data,backups}
chown -R discord-bot:discord-bot $APP_DIR
print_success "Adresáře vytvořeny"

# Step 6: Python virtual environment
print_info "Nastavení Python prostředí..."
if [ ! -d "$APP_DIR/venv" ]; then
    sudo -u discord-bot python3.11 -m venv venv
fi
sudo -u discord-bot $APP_DIR/venv/bin/pip install -q --upgrade pip
sudo -u discord-bot $APP_DIR/venv/bin/pip install -q -r $APP_DIR/requirements-production.txt
print_success "Python prostředí připraveno"

# Step 7: Check for .env
print_header "Konfigurace .env souboru"

if [ -f "$APP_DIR/.env" ]; then
    print_info ".env soubor již existuje"
    read -p "Chceš vytvořit nový? (y/N): " recreate_env
    if [[ $recreate_env =~ ^[Yy]$ ]]; then
        rm $APP_DIR/.env
    fi
fi

if [ ! -f "$APP_DIR/.env" ]; then
    print_info "Vytvoření .env konfigurace..."
    echo ""
    
    # Use interactive env creator
    sudo -u discord-bot bash -c "cd $APP_DIR && ./create_env.sh"
    
    if [ ! -f "$APP_DIR/.env" ]; then
        print_error ".env soubor nebyl vytvořen!"
        print_info "Můžeš ho vytvořit ručně:"
        echo "  cd $APP_DIR"
        echo "  cp .env.example .env"
        echo "  nano .env"
        exit 1
    fi
fi

print_success ".env soubor připraven"

# Step 8: Install systemd service
print_info "Instalace systemd služby..."
cp $APP_DIR/discord-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable discord-bot
print_success "Služba nainstalována"

# Step 9: Optional nginx
read -p "Chceš nastavit nginx pro health check endpoint? (Y/n): " setup_nginx
if [[ ! $setup_nginx =~ ^[Nn]$ ]]; then
    if command -v nginx &> /dev/null; then
        print_info "Konfigurace nginx..."
        
        cat > /etc/nginx/sites-available/discord-bot << 'EOF'
server {
    listen 80;
    server_name _;
    
    location /health {
        proxy_pass http://localhost:8080/health;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /metrics {
        proxy_pass http://localhost:8080/metrics;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF
        
        ln -sf /etc/nginx/sites-available/discord-bot /etc/nginx/sites-enabled/
        nginx -t && systemctl reload nginx
        print_success "Nginx nakonfigurován"
    else
        print_info "Nginx není nainstalován, přeskakuji..."
    fi
fi

# Step 10: Start service
print_header "Spouštění bota"

print_info "Startuji službu..."
systemctl start discord-bot
sleep 3

print_header "🎉 Instalace dokončena!"

# Show status
if systemctl is-active --quiet discord-bot; then
    print_success "Bot úspěšně běží!"
    echo ""
    systemctl status discord-bot --no-pager -l | head -15
else
    print_error "Bot se nespustil! Kontroluj logy:"
    echo "  sudo journalctl -u discord-bot -n 50"
    exit 1
fi

echo ""
print_header "📋 Užitečné příkazy"
echo ""
echo "  Status bota:"
echo "    sudo systemctl status discord-bot"
echo ""
echo "  Logy (real-time):"
echo "    sudo journalctl -u discord-bot -f"
echo ""
echo "  Health check:"
echo "    curl http://localhost:8080/health"
echo ""
echo "  Restart bota:"
echo "    sudo systemctl restart discord-bot"
echo ""
echo "  Update bota:"
echo "    sudo -u discord-bot $APP_DIR/update_bot.sh"
echo ""
echo "  Editace konfigurace:"
echo "    sudo nano $APP_DIR/.env"
echo "    sudo systemctl restart discord-bot"
echo ""

if command -v nginx &> /dev/null && systemctl is-active --quiet nginx; then
    SERVER_IP=$(hostname -I | awk '{print $1}')
    echo -e "${GREEN}🌐 Health check dostupný na: http://$SERVER_IP/health${NC}"
    echo ""
fi

print_success "Bot je připraven! 🚀"
