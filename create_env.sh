#!/bin/bash

# Interactive .env file creator for Discord Bot
# This script helps you create a proper .env configuration

echo "🔧 Discord Bot - .env Configuration Setup"
echo "=========================================="
echo ""

ENV_FILE=".env"

# Check if .env already exists
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  .env soubor již existuje!"
    read -p "Chceš ho přepsat? (y/N): " overwrite
    if [[ ! $overwrite =~ ^[Yy]$ ]]; then
        echo "Ukončeno. Žádné změny nebyly provedeny."
        exit 0
    fi
    echo ""
fi

echo "📝 Zadej následující údaje:"
echo ""

# Discord Token
echo "1️⃣  Discord User Token"
echo "   Jak získat: Developer Tools (F12) → Application → Local Storage → discord.com → token"
read -p "   DISCORD_TOKEN: " discord_token

echo ""

# Google Gemini API Key
echo "2️⃣  Google Gemini API Key"
echo "   Získej na: https://aistudio.google.com/app/apikey"
read -p "   GOOGLE_API_KEY: " google_api_key

echo ""

# Optional APIs
echo "3️⃣  Volitelné API klíče (stiskni Enter pro přeskočení)"
read -p "   OPENAI_API_KEY (optional): " openai_api_key
read -p "   OPENROUTER_API_KEY (optional): " openrouter_api_key
read -p "   PERPLEXITY_API_KEY (optional): " perplexity_api_key

echo ""

# Configuration
echo "4️⃣  Konfigurace"
read -p "   Environment (production/development) [production]: " environment
environment=${environment:-production}

read -p "   Log Level (DEBUG/INFO/WARNING/ERROR) [INFO]: " log_level
log_level=${log_level:-INFO}

read -p "   Worker Count (2-16) [4]: " worker_count
worker_count=${worker_count:-4}

read -p "   Memory Limit MB [1024]: " memory_limit
memory_limit=${memory_limit:-1024}

read -p "   Daily LLM Cost Limit USD [10.0]: " cost_limit
cost_limit=${cost_limit:-10.0}

echo ""

# Create .env file
echo "💾 Vytvářím .env soubor..."

cat > $ENV_FILE << EOF
# ==============================================
# Discord LLM Bot - Environment Configuration
# ==============================================
# Generated: $(date)

# ==================
# Required API Keys
# ==================

DISCORD_TOKEN=$discord_token
GOOGLE_API_KEY=$google_api_key

# ==================
# Optional API Keys
# ==================

OPENAI_API_KEY=$openai_api_key
OPENROUTER_API_KEY=$openrouter_api_key
PERPLEXITY_API_KEY=$perplexity_api_key

# ==================
# Environment Settings
# ==================

ENVIRONMENT=$environment
LOG_LEVEL=$log_level

# ==================
# Performance Settings
# ==================

WORKER_COUNT=$worker_count
MEMORY_LIMIT_MB=$memory_limit
MAX_QUEUE_SIZE=10000

# ==================
# LLM Configuration
# ==================

LLM_DAILY_COST_LIMIT=$cost_limit
DEFAULT_LLM_PROVIDER=gemini

# ==================
# Database Settings
# ==================

DATABASE_PATH=data/db.sqlite
DATABASE_BACKUP_INTERVAL_HOURS=6
DATABASE_MAX_BACKUPS=48

# ==================
# Monitoring Settings
# ==================

HEALTH_CHECK_PORT=8080
HEALTH_CHECK_INTERVAL_SECONDS=30

# ==================
# Real Test Configuration
# ==================

ENABLE_REAL_TESTS=0
TEST_SERVER_ID=
TEST_HOURS_BACK=1

# ==================
# Alert Configuration (Optional)
# ==================

EMAIL_ENABLED=false
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=
EMAIL_PASSWORD=
EMAIL_FROM=
EMAIL_TO=

SLACK_ENABLED=false
SLACK_WEBHOOK_URL=

WEBHOOK_ENABLED=false
WEBHOOK_URL=
EOF

# Set proper permissions
chmod 600 $ENV_FILE

echo ""
echo "✅ .env soubor vytvořen!"
echo ""
echo "📋 Shrnutí konfigurace:"
echo "   • Environment: $environment"
echo "   • Log Level: $log_level"
echo "   • Workers: $worker_count"
echo "   • Memory Limit: ${memory_limit}MB"
echo "   • Cost Limit: \$${cost_limit}/day"
echo ""
echo "🔒 Soubor zabezpečen (permissions: 600)"
echo ""
echo "✨ Nyní můžeš spustit deployment:"
echo "   chmod +x deploy_hostinger.sh"
echo "   sudo ./deploy_hostinger.sh"
echo ""
echo "📝 Pro úpravu konfigurace:"
echo "   nano .env"
echo ""
