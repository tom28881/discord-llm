# 🚀 NASAZENÍ NA TVŮJ HOSTINGER VPS

**IP Adresa:** `72.61.184.8`  
**OS:** Ubuntu 24.04 LTS  
**Status:** Připraveno k nasazení ✅

---

## ⚡ METODA 1: ONE-LINE INSTALL (Nejrychlejší - 3 minuty)

### Krok 1: Připoj se na VPS

```bash
ssh root@72.61.184.8
```

Zadej své root heslo (pokud se zeptá, řekni "yes" pro trust certificate)

### Krok 2: Spusť automatický instalátor

```bash
curl -sSL https://raw.githubusercontent.com/tom28881/discord-llm/main/install.sh | sudo bash
```

### Krok 3: Vyplň API klíče

Script se tě zeptá na:
- **Discord Token** (z F12 → Application → Local Storage)
- **Google Gemini API Key** (z https://aistudio.google.com/app/apikey)

### Krok 4: Hotovo! 🎉

Bot se automaticky spustí. Zkontroluj:

```bash
sudo systemctl status discord-bot
```

---

## 🛠️ METODA 2: Manuální (5 minut)

### Krok 1: Připoj se

```bash
ssh root@72.61.184.8
```

### Krok 2: Aktualizuj systém

```bash
apt-get update && apt-get upgrade -y
```

### Krok 3: Nainstaluj závislosti

```bash
apt-get install -y git python3.11 python3.11-venv python3-pip curl sqlite3 gcc g++
```

### Krok 4: Vytvoř uživatele

```bash
useradd -m -s /bin/bash discord-bot
```

### Krok 5: Stáhni projekt

```bash
cd /home
sudo -u discord-bot git clone https://github.com/tom28881/discord-llm.git discord-bot
cd discord-bot
```

### Krok 6: Vytvoř .env interaktivně

```bash
chmod +x create_env.sh
sudo -u discord-bot ./create_env.sh
```

Vyplň své API klíče.

### Krok 7: Spusť deployment

```bash
chmod +x deploy_hostinger.sh
./deploy_hostinger.sh
```

---

## 🔑 API Klíče - Kde je získat

### Discord Token
1. Otevři Discord v prohlížeči
2. Stiskni `F12` (Developer Tools)
3. Přejdi na **Application** tab
4. V levém menu: **Local Storage** → **discord.com**
5. Najdi řádek `token` a zkopíruj hodnotu (bez uvozovek)

### Google Gemini API Key
1. Jdi na: https://aistudio.google.com/app/apikey
2. Přihlaš se Google účtem
3. Klikni **Create API Key**
4. Zkopíruj klíč

---

## 📊 Po nasazení - Užitečné příkazy

### Na tvém VPS (72.61.184.8):

```bash
# Status bota
sudo systemctl status discord-bot

# Živé logy
sudo journalctl -u discord-bot -f

# Restart bota
sudo systemctl restart discord-bot

# Stop bota
sudo systemctl stop discord-bot

# Start bota
sudo systemctl start discord-bot

# Health check
curl http://localhost:8080/health

# Editace konfigurace
sudo nano /home/discord-bot/.env
sudo systemctl restart discord-bot
```

### Z internetu:

```bash
# Health check z vnějšku
curl http://72.61.184.8/health

# (funguje pokud máš nginx)
```

---

## 🔄 Update bota v budoucnu

```bash
ssh root@72.61.184.8
sudo -u discord-bot /home/discord-bot/update_bot.sh
```

---

## 🐛 Řešení problémů

### Bot se nespustí

```bash
# Zkontroluj logy
sudo journalctl -u discord-bot -n 50

# Zkontroluj .env
sudo cat /home/discord-bot/.env

# Manuální test
sudo -u discord-bot bash
cd /home/discord-bot
source venv/bin/activate
python enhanced_main.py
```

### Permission chyby

```bash
sudo chown -R discord-bot:discord-bot /home/discord-bot
sudo chmod 600 /home/discord-bot/.env
```

### Databáze locked

```bash
sudo systemctl stop discord-bot
sudo pkill -f "python.*enhanced_main.py"
sudo systemctl start discord-bot
```

---

## ✅ Checklist po nasazení

- [ ] SSH připojení funguje: `ssh root@72.61.184.8`
- [ ] Bot service běží: `systemctl is-active discord-bot`
- [ ] Health check OK: `curl localhost:8080/health`
- [ ] Žádné chyby v logách: `journalctl -u discord-bot -n 20`
- [ ] .env je zabezpečený: `ls -la /home/discord-bot/.env` (600)

---

## 🎯 Quick Commands Cheat Sheet

```bash
# Připojení
ssh root@72.61.184.8

# Status
systemctl status discord-bot

# Logy
journalctl -u discord-bot -f

# Restart
systemctl restart discord-bot

# Update
sudo -u discord-bot /home/discord-bot/update_bot.sh

# Health
curl localhost:8080/health
```

---

## 📞 Potřebuješ pomoct?

1. **Zkontroluj logy**: `sudo journalctl -u discord-bot -n 100`
2. **Health check**: `curl http://localhost:8080/health`
3. **Restart**: `sudo systemctl restart discord-bot`

**Bot poběží 24/7 s automatickým restartem při pádu!** 🚀
