# 🎯 Discord Monitor - Uživatelská příručka

## 📋 Obsah
1. [Přehled aplikace](#přehled-aplikace)
2. [První kroky - Jak importovat Discord zprávy](#první-kroky---jak-importovat-discord-zprávy)
3. [Návod k jednotlivým funkcím](#návod-k-jednotlivým-funkcím)
4. [Proč mohou být některé karty prázdné](#proč-mohou-být-některé-karty-prázdné)
5. [Jak nastavit časové filtry a rozsahy](#jak-nastavit-časové-filtry-a-rozsahy)
6. [Vysvětlení ML funkcí](#vysvětlení-ml-funkcí)
7. [Interpretace výsledků](#interpretace-výsledků)
8. [Řešení problémů](#řešení-problémů)

---

## 📊 Přehled aplikace

Discord Monitor je inteligentní asistent, který analyzuje vaše Discord zprávy a pomáhá vám nenechat si ujít důležité diskuse, skupinové nákupy, termíny či náladu komunity.

**Klíčové funkce:**
- 🚨 **Urgentní upozornění** - Důležité zprávy, které vyžadují pozornost
- 🛒 **Predikce nákupů** - AI detekce skupinových nákupů a příležitostí
- ⏰ **Sledování termínů** - Automatické rozpoznání deadlinů
- 😊 **Nálada skupiny** - Analýza sentimentu a energie komunity
- 👥 **Skupinové aktivity** - Detekce společných akcí a rozhodování
- 💬 **Chytrý souhrn** - AI-generované přehledy diskusí
- 🔍 **Inteligentní vyhledávání** - Pokročilé vyhledávání ve zprávách

**Aktuální stav:** Aplikace obsahuje **69,705 zpráv** v databázi a běží na adrese http://localhost:8502

---

## 🚀 První kroky - Jak importovat Discord zprávy

### Před spuštěním importu

1. **Ujistěte se, že máte správně nakonfigurovaný `.env` soubor:**
   ```bash
   DISCORD_TOKEN=váš_discord_token
   OPENAI_API_KEY=váš_openai_klíč
   ```

2. **Aktivujte virtuální prostředí:**
   ```bash
   source venv/bin/activate
   ```

### Spuštění importu zpráv

#### Možnost A: Import ze všech serverů
```bash
python load_messages.py
```
- Importuje zprávy ze všech Discord serverů, ke kterým máte přístup
- Proces může trvat několik minut až hodin (závisí na množství dat)
- Automaticky přeskakuje zakázané kanály

#### Možnost B: Import z konkrétního serveru
```bash
python load_messages.py --server_id 123456789012345678
```
- Rychlejší pro testování nebo aktualizaci konkrétního serveru
- ID serveru najdete v Discord nastavení nebo v aplikaci

### Co se děje během importu

1. **Inicializace databáze** - Vytvoří se nebo aktualizuje SQLite databáze
2. **Připojení k Discordu** - Ověří se přihlašovací token
3. **Získání seznamu serverů** - Najde všechny dostupné servery
4. **Pro každý server:**
   - Získá seznam kanálů
   - Stáhne nové zprávy (pouze ty, které ještě nejsou v databázi)
   - Uloží do databáze s úplnými informacemi
5. **Rate limiting** - Automatické pauzy (1s mezi kanály, 10s mezi servery)

### Indikátory průběhu

**V terminálu uvidíte:**
```
2024-09-12 16:44:33 - INFO - Processing server: Můj Server (ID: 123456789)
2024-09-12 16:44:34 - INFO - Fetching messages from channel: #obecné (ID: 987654321)
2024-09-12 16:44:35 - INFO - Stored 156 new messages from channel #obecné
```

**Když je import hotový:**
- Aplikace běží nepřetržitě a pravidelně kontroluje nové zprávy
- Data jsou uložena v `data/db.sqlite`
- Zakázané kanály se ukládají do `config.json`

---

## 🎮 Návod k jednotlivým funkcím

### 📊 Quick Overview (Rychlý přehled)
**Co zobrazuje:**
- Souhrnné metriky za vybrané období
- Počet důležitých zpráv a skupinových aktivit
- Timeline aktivity s grafy
- Klíčová slova a trendy

**Jak použít:**
1. Vyberte server v postranním panelu
2. Nastavte časový rozsah (doporučeno: 72 hodin)
3. Prohlédněte si metriky a grafy
4. Klikněte na "Mark All as Read" pro označení jako přečtené

### 🚨 Urgent Alerts (Urgentní upozornění)
**Co zobrazuje:**
- Zprávy s vysokým skóre důležitosti (nad 0.8)
- Kritické a vysoce prioritní zprávy
- Barevné rozlišení podle urgentnosti

**Proč může být prázdné:**
- Žádné zprávy nedosáhly vysokého skóre důležitosti
- Příliš krátký časový rozsah
- ML model ještě neanalyzoval zprávy

**Jak interpretovat:**
- 🔴 **Kritické:** Vyžadují okamžitou pozornost
- 🟡 **Vysoká priorita:** Důležité, ale ne akutní

### 🛒 Purchase Predictions (Predikce nákupů)
**Co dělá:**
- AI analyzuje zprávy pro detekci skupinových nákupů
- Rozpoznává zmínky o cenách, produktech, termínech
- Hodnotí pravděpodobnost realizace nákupu

**Typické vzory:**
- "Kdo má zájem o společný nákup...?"
- Zmínky o slevách a akčních nabídkách
- Diskuse o rozdělování nákladů
- Termíny plateb a objednávek

**Indikátory:**
- **🔴 80%+ pravděpodobnost:** Velmi pravděpodobný nákup
- **🟡 70-80%:** Střední pravděpodobnost
- **🟢 50-70%:** Nízká pravděpodobnost

### ⏰ Deadlines (Termíny)
**Co detekuje:**
- Datumy a časové lhůty ve zprávách
- Termíny plateb, registrací, akcí
- Urgentnost podle zbývajícího času

**Vzory rozpoznávání:**
- "do 15. října"
- "nejpozději v pátek"
- "deadline je..."
- "musí být hotové do"

**Kategorie:**
- **🚨 Urgentní:** Méně než 24 hodin
- **📅 Normální:** Více než den

### 😊 Group Mood (Nálada skupiny)
**Co analyzuje:**
- Sentiment zpráv (pozitivní/negativní)
- Úroveň aktivity v kanálech
- Emocionální energie diskusí

**Indikátory nálady:**
- 🔥 **Hyped:** Vysoká energie, nadšení
- ⚡ **Active:** Aktivní diskuse
- 😊 **Positive:** Pozitivní atmosféra
- 😐 **Neutral:** Neutrální tón
- 😟 **Tense:** Napětí v diskusi
- 😴 **Quiet:** Málo aktivity

### 👥 Group Activities (Skupinové aktivity)
**Typy aktivit:**
- 🛒 **Group Purchases:** Společné nákupy
- 📅 **Events:** Plánování akcí
- 🗳️ **Decisions:** Skupinová rozhodování
- ⚡ **FOMO Moments:** Vysoká aktivita ("fear of missing out")

**Jak se detekují:**
- Vysoká koncentrace zpráv v krátkém čase
- Více účastníků v diskusi
- Klíčová slova související s aktivitami

### 💬 Smart Digest (Chytrý souhrn)
**Funkce:**
- AI-generované shrnutí diskusí
- Zprávy seskupené podle kanálů
- Expandovatelné detaily

**Použití:**
1. Klikněte na "Generate AI Summary"
2. Počkejte na analýzu (může trvat 30-60 sekund)
3. Prohlédněte si souhrn a detaily podle kanálů

### 🔍 Search (Vyhledávání)
**Možnosti vyhledávání:**
- Klíčová slova
- Jména uživatelů (@username)
- Témata diskusí
- Kombinace filtrů

**Výsledky obsahují:**
- **Relevance score:** Jak dobře zpráva odpovídá dotazu
- **Importance score:** Důležitost zprávy
- Kontextové informace

---

## ❓ Proč mohou být některé karty prázdné

### 🕐 Časové filtry
**Nejčastější příčina:** Příliš krátký časový rozsah
- **Doporučení:** Nastavte alespoň 72 hodin (3 dny)
- **Pro analýzu trendů:** 168 hodin (1 týden)

### 📊 Práh důležitosti
**Problém:** Vysoký práh důležitosti (nad 0.7)
- **Řešení:** Snižte práh na 0.3-0.5
- **Důvod:** ML model může být konzervativní v hodnocení

### 🤖 ML analýza probíhá
**Nové zprávy:** ML funkce potřebují čas na analýzu
- **Čekací doba:** 5-10 minut po importu nových zpráv
- **Řešení:** Obnovte stránku po chvíli

### 📱 Nízká aktivita
**V databázi není dostatek dat:**
- Málo zpráv v daném období
- Kanály s nízkou aktivitou
- Žádné skupinové aktivity

### 🔧 Technické problémy
**Možné příčiny:**
- Databázová chyba
- Chybějící ML modely
- Nesprávná konfigurace

---

## ⚙️ Jak nastavit časové filtry a rozsahy

### Postranní panel - Quick Controls

#### 🎯 Server Selection (Výběr serveru)
- Vyberte konkrétní Discord server pro analýzu
- Pokud není server vidět, spusťte `python load_messages.py`

#### ⏰ Time Range (Časový rozsah)
**Posuvník: 1-168 hodin**

**Doporučená nastavení:**
- **24 hodin:** Pro denní přehled
- **72 hodin:** Standardní nastavení (3 dny)
- **168 hodin:** Týdenní analýza
- **Více než týden:** Pro dlouhodobé trendy

#### 📊 Importance Threshold (Práh důležitosti)
**Rozsah: 0.0 - 1.0**

**Interpretace:**
- **0.0-0.3:** Všechny zprávy včetně méně důležitých
- **0.4-0.6:** Středně důležité zprávy
- **0.7-1.0:** Pouze velmi důležité zprávy

**Tip:** Začněte s 0.5 a podle potřeby upravujte

#### 🔄 Auto-refresh
- **Zapnuto:** Automatické obnovení každých 30 sekund
- **Doporučeno:** Pouze pro aktivní monitoring

### 📈 Quick Stats (Rychlé statistiky)
**Zobrazované metriky:**
- **Total Messages:** Celkový počet zpráv
- **Active Users:** Počet aktivních uživatelů
- **Important:** Důležité zprávy nad prahem
- **Group Activities:** Detekované skupinové aktivity

---

## 🤖 Vysvětlení ML funkcí

### 🎯 Importance Scoring (Hodnocení důležitosti)
**Jak funguje:**
- Analyzuje obsah zprávy
- Hodnotí kontext a reakce
- Přiřadí skóre 0.0-1.0

**Faktory ovlivňující skóre:**
- Zmínky uživatelů (@username)
- Klíčová slova (nákup, termín, akce)
- Délka zprávy a strukturovanost
- Reakce od ostatních uživatelů
- Frekvence určitých slov

**Kalibrace:**
- Skóre nad 0.8: Kriticky důležité
- 0.6-0.8: Velmi důležité
- 0.4-0.6: Středně důležité
- Pod 0.4: Běžné zprávy

### 🛒 Purchase Prediction (Predikce nákupů)
**Rozpoznávací vzory:**
```
• "Kdo má zájem o..."
• "Společný nákup"
• "Rozdělíme si náklady"
• "Sleva jen do..."
• Zmínky o cenách (Kč, €, $)
• "Objednávám, kdo ještě?"
```

**Metadata extrakce:**
- **Price mentions:** Detekované ceny
- **Purchase items:** Identifikované produkty
- **Urgency level:** Naléhavost (1-5 ⚡)
- **Participants:** Potenciální účastníci

### ⏰ Deadline Detection (Detekce termínů)
**Rozpoznávané formáty:**
```
• "do 15.10.2024"
• "nejpozději v pátek"
• "deadline je zítra"
• "musí být hotové do"
• "termín: "
• Relativní časy (za 3 dny, příští týden)
```

**Urgentnost:**
- **Level 5:** < 6 hodin
- **Level 4:** < 24 hodin
- **Level 3:** < 3 dny
- **Level 2:** < 1 týden
- **Level 1:** > 1 týden

### 😊 Sentiment Analysis (Analýza nálady)
**Dimenze analýzy:**
- **Positive/Negative:** Pozitivita obsahu
- **Excitement level:** Úroveň nadšení
- **Activity intensity:** Intenzita diskuse

**Detekované emoce:**
- Radost, nadšení, vzrušení
- Stres, frustraci, napětí
- Neutralita, klid
- Humor, sarkasmus

### 👥 Pattern Recognition (Rozpoznávání vzorů)
**Skupinové vzory:**
- **Clustering:** Seskupování souvisejících zpráv
- **Thread detection:** Rozpoznání vláken diskuse
- **Activity bursts:** Výbuchy aktivity
- **Decision patterns:** Vzory rozhodování

---

## 📊 Interpretace výsledků

### 📈 Metriky a jejich význam

#### Importance Score
```
0.9-1.0: 🔴 KRITICKÉ - Okamžitá pozornost
0.7-0.8: 🟠 VYSOKÁ - Brzy řešit
0.5-0.6: 🟡 STŘEDNÍ - Průměrná priorita
0.3-0.4: 🟢 NÍZKÁ - Informativní
0.0-0.2: ⚪ BĚŽNÉ - Běžné zprávy
```

#### Purchase Probability
```
90-100%: 🔴 Téměř jistý nákup - Připravte se!
70-89%:  🟡 Vysoká pravděpodobnost - Sledujte
50-69%:  🟢 Střední šance - Mějte na paměti
30-49%:  🔵 Nízká pravděpodobnost
0-29%:   ⚪ Nepravděpodobné
```

#### Activity Level
```
50+ zpráv/hodina: 🔥 Velmi vysoká aktivita
20-49: ⚡ Vysoká aktivita  
10-19: 📈 Střední aktivita
5-9:   📊 Nízká aktivita
0-4:   😴 Téměř žádná aktivita
```

### 🎨 Barevné kódování

#### Urgentnost
- **🔴 Červená:** Kritické, vyžaduje okamžitou pozornost
- **🟡 Žlutá:** Důležité, řešit brzy
- **🟢 Zelená:** Normální priorita
- **🔵 Modrá:** Informativní
- **⚪ Šedá:** Běžné zprávy

#### Typ aktivity
- **🛒 Nákupy:** Modrozelená (#0dcaf0)
- **⚡ FOMO:** Pulsující žlutá
- **😊 Pozitivní:** Zelená
- **😟 Negativní:** Červenofialová

### 📊 Grafy a vizualizace

#### Timeline Graph
- **Osa X:** Čas (hodiny/dny)
- **Osa Y:** Skóre důležitosti
- **Velikost bodů:** Intenzita aktivity
- **Barva:** Typ aktivity

#### Mood Charts
- **Barvy:** Reprezentují náladu kanálu
- **Intenzita:** Úroveň aktivity
- **Ikony:** Emoji podle nálady

---

## 🔧 Řešení problémů

### 🚨 Časté problémy a řešení

#### Prázdné karty/tabulky
**Problém:** Některé karty neobsahují data

**Řešení:**
1. **Zkontrolujte časový rozsah**
   ```
   • Zvětšete na 72+ hodin
   • Pro analýzu trendů použijte 168 hodin
   ```

2. **Snižte práh důležitosti**
   ```
   • Z 0.7 na 0.3-0.5
   • Umožní zobrazit více zpráv
   ```

3. **Aktualizujte data**
   ```bash
   python load_messages.py --server_id YOUR_SERVER_ID
   ```

4. **Obnovte stránku**
   - Ctrl+F5 (Windows) nebo Cmd+Shift+R (Mac)
   - Vyčistí cache a znovu načte data

#### Chyby v import procesu
**Chyba:** "403 Forbidden"
```
• Kanál je privátní nebo nemáte oprávnění
• Automaticky se přidá do forbidden_channels
• Normální chování, pokračuje na další kanál
```

**Chyba:** "Token invalid"
```
• Zkontrolujte DISCORD_TOKEN v .env
• Token může být expirovaný
• Vygenerujte nový token v Discord Developer Portal
```

#### Pomalé načítání
**Příčiny a řešení:**
1. **Velké množství dat**
   - Použijte konkrétní server ID
   - Omezte časový rozsah

2. **Slabé internetové připojení**
   - Proces pozastavte a spusťte znovu
   - Importování pokračuje odkud skončilo

3. **AI analýza probíhá**
   - Počkejte 5-10 minut
   - ML modely analyzují nová data na pozadí

#### Database chyby
**Chyba:** "Database locked"
```bash
# Restartujte aplikaci
pkill -f streamlit
streamlit run streamlit_monitoring.py --server.port 8502
```

**Chyba:** "Table doesn't exist"
```bash
# Reinicializujte databázi
python -c "from lib.database_optimized import init_optimized_db; init_optimized_db()"
```

### 📞 Debug informace

#### Logování
```bash
# Zobrazit logy importu
tail -f nohup.out

# Nebo sledovat v terminálu
python load_messages.py --server_id YOUR_ID
```

#### Database inspection
```bash
# Zkontrolovat počet zpráv
python -c "
from lib.database_optimized import OptimizedDatabase
db = OptimizedDatabase()
print(f'Messages: {db.get_message_count()}')
"
```

#### Status check
```bash
# Rychlý status check
python -c "
from lib.database_optimized import OptimizedDatabase
db = OptimizedDatabase()
servers = db.get_servers()
print(f'Servers: {len(servers)}')
for server_id, server_name in servers:
    print(f'  {server_name}: {server_id}')
"
```

---

## 🎓 Tipy pro efektivní používání

### ⚡ Best Practices

1. **Pravidelná aktualizace dat**
   ```bash
   # Spouštějte denně nebo nastavte cron job
   python load_messages.py
   ```

2. **Optimální nastavení**
   ```
   • Časový rozsah: 72 hodin
   • Práh důležitosti: 0.5
   • Auto-refresh: pouze při aktivním sledování
   ```

3. **Personalizace**
   - Přidejte klíčová slova v Preferences
   - Nastavte důležitost podle vašich zájmů
   - Používejte různé časové rozsahy pro různé účely

4. **Monitoring workflow**
   ```
   1. Rychlý Overview - celkový stav
   2. Urgent Alerts - co vyžaduje pozornost
   3. Purchase Predictions - obchodní příležitosti
   4. Deadlines - časové lhůty
   5. Group Mood - atmosféra komunity
   ```

### 🔮 Pokročilé použití

#### Custom Keywords
- Přidejte specifické termíny pro vaše komunity
- Nastavte vyšší váhu pro důležitá témata
- Příklady: "meetup", "giveaway", "collab", názvy projektů

#### Time-based Analysis
- **Ráno (8-10):** Overnight summary, dlouhý časový rozsah
- **Odpoledne (14-16):** Aktuální aktivita, střední rozsah
- **Večer (20-22):** Denní wrap-up, kratší rozsah

#### Multi-server Strategy
- Různé servery = různé účely
- Gaming servery: FOMO a events
- Work servery: deadlines a decisions
- Social servery: mood a general discussions

---

## 🆘 Kontakt a podpora

### 📧 Získání podpory

**Technické problémy:**
1. Zkontrolujte tuto příručku
2. Podívejte se na logy v terminálu
3. Restartujte aplikaci
4. Zkuste reinicializovat databázi

**Feature requests:**
- Dokumentujte požadovanou funkcionalnost
- Popište use case
- Navrhněte UI/UX řešení

### 🔄 Aktualizace a údržba

**Pravidelně:**
```bash
# Aktualizace zpráv
python load_messages.py

# Kontrola stavu databáze
python scripts/manage_db.py --stats
```

**Měsíčně:**
- Vyčistěte staré logy
- Zkontrolujte velikost databáze
- Aktualizujte ML modely pokud je to možné

**Záložní kopie:**
```bash
# Záloha databáze
cp data/db.sqlite data/backup_$(date +%Y%m%d).sqlite
```

---

## 🎯 Shrnutí

Discord Monitor je mocný nástroj pro sledování a analýzu Discord komunikace. Správným nastavením časových rozsahů, prahů důležitosti a pravidelnou aktualizací dat získáte cenné insights o vašich komunitách.

**Klíčové body:**
- ⏰ **Časový rozsah:** 72 hodin pro optimální výsledky
- 📊 **Práh důležitosti:** 0.5 jako výchozí hodnota
- 🔄 **Pravidelné aktualizace:** Denní import nových zpráv
- 🎯 **Personalizace:** Vlastní klíčová slova a priority

**Pamatujte:**
- ML analýza potřebuje čas na zpracování nových dat
- Prázdné karty často znamenají správnou funkci (žádné urgentní problémy)
- Aplikace je navržena jako preventivní nástroj, ne real-time monitor

**Užijte si efektivnější Discord komunikaci! 🚀**