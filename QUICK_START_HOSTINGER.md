# ⚡ Quick Start - Hostinger VPS Deployment

Rychlý průvodce pro nasazení Discord bota na Hostinger během 5 minut.

## 🚀 Jednoduchý postup (5 minut)

### 1. Připoj se na VPS
```bash
ssh root@tvoje-hostinger-ip
```

### 2. Nahraj projekt
```bash
cd /home
# Možnost A: Git
git clone https://tvuj-repo/discord-llm.git discord-bot

# Možnost B: SCP z lokálního PC
# scp -r /path/to/discord-llm root@ip:/home/discord-bot
```

### 3. Vytvoř .env soubor
```bash
cd /home/discord-bot
nano .env
```

Vlož (nahraď vlastními klíči):
```env
DISCORD_TOKEN=tvuj_discord_token
GOOGLE_API_KEY=tvuj_google_api_key
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKER_COUNT=4
MEMORY_LIMIT_MB=1024
```
Uložit: `Ctrl+X` → `Y` → `Enter`

### 4. Spusť automatický deployment
```bash
chmod +x deploy_hostinger.sh
sudo ./deploy_hostinger.sh
```

### 5. Hotovo! ✅
```bash
# Zkontroluj status
sudo systemctl status discord-bot

# Sleduj logy
sudo journalctl -u discord-bot -f
```

---

## 📋 Nejpoužívanější příkazy

### Správa služby
```bash
sudo systemctl status discord-bot    # Status
sudo systemctl restart discord-bot   # Restart
sudo systemctl stop discord-bot      # Stop
sudo systemctl start discord-bot     # Start
```

### Logy
```bash
sudo journalctl -u discord-bot -f    # Real-time logy
sudo journalctl -u discord-bot -n 50 # Posledních 50 řádků
tail -f /home/discord-bot/logs/bot.log  # Aplikační logy
```

### Health check
```bash
curl http://localhost:8080/health    # Základní check
curl http://localhost:8080/metrics   # Metriky
```

### Aktualizace bota
```bash
sudo -u discord-bot /home/discord-bot/update.sh
```

---

## 🔧 Řešení problémů

### Bot se nespustí
```bash
# 1. Zkontroluj logy
sudo journalctl -u discord-bot -n 20

# 2. Test .env souboru
cat /home/discord-bot/.env

# 3. Manuální spuštění pro debug
sudo -u discord-bot bash
cd /home/discord-bot
source venv/bin/activate
python enhanced_main.py
```

### Oprav oprávnění
```bash
sudo chown -R discord-bot:discord-bot /home/discord-bot
sudo chmod 600 /home/discord-bot/.env
```

### Restart všeho
```bash
sudo systemctl daemon-reload
sudo systemctl restart discord-bot
```

---

## 📊 Monitoring

### Rychlá diagnostika
```bash
# Status + využití zdrojů
systemctl status discord-bot

# Paměť a CPU
top -p $(pgrep -f enhanced_main.py)

# Velikost databáze
ls -lh /home/discord-bot/data/

# Volné místo
df -h
```

---

## 🔄 Běžné úkoly

### Změna konfigurace
```bash
# Edituj .env
sudo nano /home/discord-bot/.env

# Restart pro aplikaci změn
sudo systemctl restart discord-bot
```

### Záloha databáze
```bash
sudo -u discord-bot cp /home/discord-bot/data/db.sqlite \
  /home/discord-bot/backups/backup-$(date +%Y%m%d).sqlite
```

### Stažení záloh lokálně
```bash
# Z tvého PC
scp root@tvoje-ip:/home/discord-bot/backups/*.sqlite ./
```

---

## 🆘 Rychlá pomoc

### Kde najít informace

| Co potřebuješ | Příkaz |
|---------------|---------|
| Je bot živý? | `systemctl status discord-bot` |
| Co dělá bot? | `journalctl -u discord-bot -f` |
| Kolik žere paměť? | `systemctl status discord-bot` |
| Funguje API? | `curl localhost:8080/health` |
| Chyby v logách | `journalctl -u discord-bot -p err` |

### Kompletní diagnostika
```bash
# Spusť tohle a pošli výstup
echo "=== Status ===" && systemctl status discord-bot && \
echo "=== Logs ===" && journalctl -u discord-bot -n 30 && \
echo "=== Health ===" && curl -s localhost:8080/health && \
echo "=== Resources ===" && free -h && df -h
```

---

## 📝 Pro více detailů

- **Kompletní průvodce**: `HOSTINGER_DEPLOYMENT.md`
- **Produkční setup**: `README_PRODUCTION.md`
- **Docker deployment**: `docker/docker-compose.yml`

---

## ✅ Checklist po nasazení

- [ ] Bot běží: `systemctl status discord-bot`
- [ ] Health check OK: `curl localhost:8080/health`
- [ ] Žádné chyby: `journalctl -u discord-bot -n 20`
- [ ] .env je zabezpečený: `ls -l /home/discord-bot/.env` (600)
- [ ] Automatický restart: `systemctl is-enabled discord-bot` (enabled)

---

**Vše funguje? Skvělé! Bot teď běží 24/7 na pozadí. 🎉**
