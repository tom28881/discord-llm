# 🚀 Nasazení Discord Bota na Hostinger VPS

Kompletní průvodce nasazením Discord LLM bota na Hostinger VPS hosting.

## 📋 Požadavky

### Co budeš potřebovat:

1. **Hostinger VPS** (minimálně)
   - 2 GB RAM
   - 2 CPU jádra
   - 20 GB disk
   - Ubuntu 20.04+ nebo Debian 11+

2. **API Klíče**
   - Discord User Token
   - Google Gemini API Key (nebo OpenAI/OpenRouter)

3. **Přístup k serveru**
   - SSH přístup
   - Root nebo sudo oprávnění

## 🎯 Rychlé nasazení (automatické)

### Krok 1: Připojení na VPS

```bash
ssh root@tvoje-hostinger-ip
```

### Krok 2: Stažení projektu

```bash
cd /home
git clone https://github.com/tvuj-repo/discord-llm.git discord-bot
cd discord-bot
```

Nebo nahraj soubory pomocí SFTP/SCP:
```bash
# Z lokálního počítače
scp -r . root@tvoje-hostinger-ip:/home/discord-bot/
```

### Krok 3: Vytvoření .env souboru

```bash
cd /home/discord-bot
nano .env
```

Vlož následující obsah (nahraď vlastními klíči):
```bash
# Discord API
DISCORD_TOKEN=tvuj_discord_token_zde

# Google Gemini API
GOOGLE_API_KEY=tvuj_google_api_key_zde

# Volitelné API
OPENAI_API_KEY=
OPENROUTER_API_KEY=

# Konfigurace prostředí
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKER_COUNT=4
MEMORY_LIMIT_MB=1024
LLM_DAILY_COST_LIMIT=10.0

# Health check
HEALTH_CHECK_PORT=8080
```

Uložit: `Ctrl+X`, potom `Y`, potom `Enter`

### Krok 4: Spuštění deployment scriptu

```bash
chmod +x deploy_hostinger.sh
sudo ./deploy_hostinger.sh
```

Script automaticky:
- ✅ Nainstaluje všechny závislosti
- ✅ Vytvoří uživatele a adresáře
- ✅ Nastaví Python virtual environment
- ✅ Vytvoří systemd službu
- ✅ Nakonfiguruje nginx pro health check
- ✅ Spustí bota

### Krok 5: Ověření

```bash
# Zkontroluj status
sudo systemctl status discord-bot

# Sleduj logy
sudo journalctl -u discord-bot -f

# Health check
curl http://localhost:8080/health
```

## 🛠️ Manuální nasazení (krok za krokem)

Pokud preferuješ manuální kontrolu nebo máš problémy s automatickým scriptem:

### 1. Příprava systému

```bash
# Aktualizace systému
sudo apt-get update && sudo apt-get upgrade -y

# Instalace závislostí
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    sqlite3 \
    gcc \
    g++ \
    nginx
```

### 2. Vytvoření uživatele

```bash
# Vytvoř uživatele pro aplikaci
sudo useradd -m -s /bin/bash discord-bot

# Vytvoř aplikační adresář
sudo mkdir -p /home/discord-bot/{logs,data,backups}
sudo chown -R discord-bot:discord-bot /home/discord-bot
```

### 3. Nahrání aplikace

```bash
# Zkopíruj soubory do /home/discord-bot
cd /home/discord-bot

# Nastav oprávnění
sudo chown -R discord-bot:discord-bot /home/discord-bot
```

### 4. Python prostředí

```bash
# Přepni na uživatele discord-bot
sudo -u discord-bot bash

# Vytvoř virtual environment
python3.11 -m venv venv

# Aktivuj virtual environment
source venv/bin/activate

# Nainstaluj závislosti
pip install --upgrade pip
pip install -r requirements-production.txt
```

### 5. Konfigurace .env

```bash
# Vytvoř .env soubor (jako discord-bot uživatel)
nano .env
```

Vlož konfiguraci (viz výše) a uložit.

```bash
# Zabezpeč .env soubor
chmod 600 .env
```

### 6. Vytvoření systemd služby

```bash
# Opusť discord-bot uživatele
exit

# Zkopíruj service soubor
sudo cp discord-bot.service /etc/systemd/system/

# Nebo vytvoř manuálně
sudo nano /etc/systemd/system/discord-bot.service
```

Obsah viz soubor `discord-bot.service` v projektu.

### 7. Spuštění služby

```bash
# Reload systemd
sudo systemctl daemon-reload

# Povolit automatický start
sudo systemctl enable discord-bot

# Spustit službu
sudo systemctl start discord-bot

# Zkontrolovat status
sudo systemctl status discord-bot
```

### 8. Nginx konfigurace (volitelné)

```bash
# Vytvoř nginx config
sudo nano /etc/nginx/sites-available/discord-bot
```

Obsah:
```nginx
server {
    listen 80;
    server_name tvoje-domena.cz;  # nebo IP adresa
    
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
```

Aktivuj config:
```bash
sudo ln -s /etc/nginx/sites-available/discord-bot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 📊 Správa a monitoring

### Základní příkazy

```bash
# Status služby
sudo systemctl status discord-bot

# Restart bota
sudo systemctl restart discord-bot

# Stop bota
sudo systemctl stop discord-bot

# Start bota
sudo systemctl start discord-bot

# Sledování logů (real-time)
sudo journalctl -u discord-bot -f

# Poslední 100 řádků logů
sudo journalctl -u discord-bot -n 100

# Logy od včerejška
sudo journalctl -u discord-bot --since yesterday
```

### Log soubory

```bash
# Aplikační logy
tail -f /home/discord-bot/logs/bot.log

# Error logy
tail -f /home/discord-bot/logs/error.log

# Systémové logy
sudo journalctl -u discord-bot
```

### Health check

```bash
# Lokální kontrola
curl http://localhost:8080/health

# Detailní status
curl http://localhost:8080/health/detailed

# Metriky
curl http://localhost:8080/metrics

# Z vnějšku (pokud máš nginx)
curl http://tvoje-ip/health
```

### Monitoring zdrojů

```bash
# Paměť a CPU využití
systemctl status discord-bot

# Detailní info o procesech
ps aux | grep python

# Využití disku
du -sh /home/discord-bot/*

# Velikost databáze
ls -lh /home/discord-bot/data/
```

## 🔄 Aktualizace bota

### Automatická aktualizace

```bash
# Použij update script
sudo -u discord-bot /home/discord-bot/update.sh
```

### Manuální aktualizace

```bash
# Přepni na discord-bot uživatele
sudo -u discord-bot bash
cd /home/discord-bot

# Pull změny z gitu
git pull

# Aktualizuj závislosti
source venv/bin/activate
pip install --upgrade -r requirements-production.txt

# Opusť uživatele
exit

# Restart služby
sudo systemctl restart discord-bot
```

## 🔧 Řešení problémů

### Bot se nespustí

```bash
# Zkontroluj logy
sudo journalctl -u discord-bot -n 50

# Zkontroluj .env soubor
sudo -u discord-bot cat /home/discord-bot/.env

# Test Python prostředí
sudo -u discord-bot bash
cd /home/discord-bot
source venv/bin/activate
python enhanced_main.py
```

### Chyba "Permission denied"

```bash
# Zkontroluj oprávnění
ls -la /home/discord-bot

# Oprav oprávnění
sudo chown -R discord-bot:discord-bot /home/discord-bot
sudo chmod 600 /home/discord-bot/.env
```

### Databáze je zamčená

```bash
# Stop službu
sudo systemctl stop discord-bot

# Zkontroluj proces
ps aux | grep python

# Zabij visící proces (pokud existuje)
sudo pkill -f "python.*enhanced_main.py"

# Start službu
sudo systemctl start discord-bot
```

### Vysoká paměť/CPU

```bash
# Zkontroluj resource usage
systemctl status discord-bot

# Sniž WORKER_COUNT v .env
sudo -u discord-bot nano /home/discord-bot/.env
# Změň: WORKER_COUNT=2

# Restart
sudo systemctl restart discord-bot
```

### Health check nefunguje

```bash
# Zkontroluj, jestli port 8080 běží
sudo netstat -tulpn | grep 8080

# Zkontroluj firewall
sudo ufw status

# Povolí port (pokud je firewall aktivní)
sudo ufw allow 8080
```

## 🔐 Zabezpečení

### Firewall konfigurace

```bash
# Aktivuj UFW firewall
sudo ufw enable

# Povolí SSH
sudo ufw allow ssh

# Povolí HTTP/HTTPS (pokud používáš nginx)
sudo ufw allow 80
sudo ufw allow 443

# Povolí health check (volitelně)
sudo ufw allow 8080

# Zkontroluj status
sudo ufw status
```

### Zabezpečení API klíčů

```bash
# .env soubor by měl být read-only pro discord-bot uživatele
sudo chmod 600 /home/discord-bot/.env
sudo chown discord-bot:discord-bot /home/discord-bot/.env

# Nikdy necommituj .env do gitu!
# Je již v .gitignore
```

### Automatické bezpečnostní aktualizace

```bash
# Nainstaluj unattended-upgrades
sudo apt-get install unattended-upgrades

# Zapni automatické aktualizace
sudo dpkg-reconfigure -plow unattended-upgrades
```

## 📈 Optimalizace výkonu

### Pro větší servery

Edituj `.env`:
```bash
WORKER_COUNT=8              # Více workerů
MEMORY_LIMIT_MB=2048        # Více paměti
MAX_QUEUE_SIZE=20000        # Větší fronta
```

### Pro menší VPS (1 GB RAM)

```bash
WORKER_COUNT=2
MEMORY_LIMIT_MB=512
MAX_QUEUE_SIZE=5000
```

## 🔄 Zálohy

### Manuální záloha databáze

```bash
# Vytvoř zálohu
sudo -u discord-bot sqlite3 /home/discord-bot/data/db.sqlite ".backup '/home/discord-bot/backups/db-$(date +%Y%m%d-%H%M%S).sqlite'"

# Nebo jednoduše zkopíruj
sudo -u discord-bot cp /home/discord-bot/data/db.sqlite /home/discord-bot/backups/db-backup.sqlite
```

### Automatická záloha (cron)

```bash
# Edituj crontab pro discord-bot uživatele
sudo -u discord-bot crontab -e

# Přidej řádek pro denní zálohu ve 2:00
0 2 * * * sqlite3 /home/discord-bot/data/db.sqlite ".backup '/home/discord-bot/backups/db-$(date +\%Y\%m\%d).sqlite'"

# Přidej čištění starých záloh (starší než 7 dní)
0 3 * * * find /home/discord-bot/backups/ -name "db-*.sqlite" -mtime +7 -delete
```

### Stažení záloh lokálně

```bash
# Z lokálního počítače
scp root@tvoje-hostinger-ip:/home/discord-bot/backups/db-*.sqlite ./backups/
```

## 📞 Podpora

### Užitečné informace pro diagnostiku

Pokud potřebuješ pomoc, připrav tyto informace:

```bash
# Verze systému
lsb_release -a

# Python verze
python3 --version

# Status služby
systemctl status discord-bot

# Poslední error logy
sudo journalctl -u discord-bot -p err -n 20

# Resource usage
free -h
df -h
top -bn1 | head -20
```

## 🎉 Hotovo!

Bot nyní běží na pozadí a automaticky se restartuje při pádu nebo restartu serveru.

### Kontrola, že vše funguje:

```bash
✅ sudo systemctl status discord-bot  # Služba běží
✅ curl http://localhost:8080/health  # Health check OK
✅ sudo journalctl -u discord-bot -n 20  # Žádné chyby
```

### Kde najít informace:

- **Status**: `sudo systemctl status discord-bot`
- **Logy**: `sudo journalctl -u discord-bot -f`
- **Health**: `curl http://localhost:8080/health`
- **Metriky**: `curl http://localhost:8080/metrics`

---

**Poznámka**: Pro Docker deployment, viz `README_PRODUCTION.md`
