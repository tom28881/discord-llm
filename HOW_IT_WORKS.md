# Jak funguje Discord LLM - Kompletní průvodce

## 🔄 Celkový Flow Systému

```
Discord API → Import → Database → UI Filter → LLM → Odpověď
```

---

## 📥 1. STAHOVÁNÍ ZPRÁV (Import)

### Co se stahuje?

**Odpověď: VŠECHNY zprávy ze VŠECH kanálů vybraného serveru**

```python
# V load_messages.py
def fetch_and_store_messages():
    # 1. Získá seznam VŠECH kanálů na serveru
    channel_info = client.get_channel_ids()
    
    # 2. Pro KAŽDÝ kanál:
    for channel_id, channel_name in channel_info:
        # Stáhne zprávy z tohoto kanálu
        new_messages = client.fetch_messages(
            channel_id,
            last_message_id,  # ← Inkrementální import!
            limit=5000,
            min_timestamp=...  # ← Time filter
        )
        # Uloží do databáze
        save_messages(messages_to_save)
```

### Filtry při importu:

✅ **Server filter**: Vyber konkrétní server
✅ **Channel filter**: Můžeš vybrat jen některé kanály (nebo všechny)
✅ **Time filter**: `hours_back` parametr (např. 720 hodin = 30 dní)
✅ **Forbidden channels**: Automaticky skipuje zakázané kanály (403 error)

### Příklad:

```
Server: InvestičníFlow (809811760273555506)
Kanály: 63 kanálů celkem
Time range: 720 hodin (30 dní)

IMPORT STÁHNE:
- #💰│investování → 450 zpráv
- #🗣│představte-se → 120 zpráv  
- #🕹│off-topic → 300 zpráv
- ... všechny ostatní kanály
---
CELKEM: 7,847 zpráv uloženo do DB
```

---

## 💾 2. DATABÁZE (SQLite)

### Co je v databázi?

```sql
servers
├── id (server ID)
└── name (server name)

channels  
├── id (channel ID)
├── server_id (FK → servers)
└── name (channel name)

messages
├── id (message ID)
├── server_id (FK → servers)
├── channel_id (FK → channels)
├── content (text zprávy)
└── sent_at (timestamp)
```

### Klíčové body:

✅ **Všechny zprávy jsou uloženy trvale**
✅ **Zprávy jsou organizované podle server → channel → message**
✅ **Inkrementální import**: Příští import stáhne jen NOVÉ zprávy
✅ **Deduplikace**: Duplicitní zprávy se automaticky skipují

---

## 🖥️ 3. STREAMLIT UI - FILTRY

### Co můžeš filtrovat v UI?

#### A) **Server Selection** (Configuration panel)
```
Vyber server: InvestičníFlow ✓
→ Načte zprávy JEN z tohoto serveru
```

#### B) **Channel Selection** (Configuration panel)
```
Channels:
☐ All channels          → Všechny kanály
☑ #investování          → Jen tento kanál
☐ #off-topic
☐ #kryptoměny
```

#### C) **Time Frame** (Configuration panel)
```
Time Frame (hours): [720] 
→ Zobrazí zprávy z posledních 720 hodin
```

### Jak to funguje v kódu:

```python
# streamlit_app.py - _refresh_recent_records()

records = get_recent_message_records(
    server_id=st.session_state.server_id,      # ← Server filter
    hours=st.session_state.hours,              # ← Time filter (720h)
    channel_ids=st.session_state.channel_ids,  # ← Channel filter
    limit=None                                 # ← Bez limitu!
)

# VÝSLEDEK:
# - Pokud "All channels": Načte ze VŠECH kanálů
# - Pokud jen "#investování": Načte JEN z tohoto kanálu
# - Time range: Jen zprávy z posledních X hodin
```

### Příklad výsledků:

| Nastavení | Výsledek |
|-----------|----------|
| Server: InvestičníFlow<br>Channels: All<br>Time: 720h | **1,563 zpráv** ze všech 63 kanálů |
| Server: InvestičníFlow<br>Channels: #investování<br>Time: 720h | **450 zpráv** jen z tohoto kanálu |
| Server: InvestičníFlow<br>Channels: All<br>Time: 24h | **113 zpráv** ze všech kanálů |

---

## 💬 4. CHAT KONVERZACE

### Jak funguje chat memory?

```python
# streamlit_app.py - display_chat()

# Chat history je uložena v session_state
messages = st.session_state.get("messages", [])

# Struktura:
messages = [
    {"role": "user", "content": "Dotaz 1"},
    {"role": "assistant", "content": "Odpověď 1"},
    {"role": "user", "content": "Dotaz 2"},
    {"role": "assistant", "content": "Odpověď 2"},
]
```

### ✅ ANO - Pamatuje si konverzaci!

**Konverzace se NEMAZZE když:**
- ❌ Změníš channel filter
- ❌ Změníš time range
- ❌ Změníš server

**Konverzace se VYMAŽĚ když:**
- ✅ Restartuje Streamlit
- ✅ Refreshneš browser (F5)
- ✅ Explicitně vyčistíš session

### Příklad konverzace:

```
TY: "Co bylo diskutováno o OpenAI?"
AI: [odpověď založená na ALL channels, 720h]

← Změníš channel na #investování ←

TY: "A co konkrétně v investování?"
AI: [odpověď založená na JEN #investování, 720h]
     [PAMATUJE SI předchozí kontext!]
```

### Jak LLM vidí kontext:

```python
# Při každém dotazu se pošle:

1. CHAT HISTORY (předchozí konverzace)
   - "Co bylo diskutováno o OpenAI?"
   - [AI odpověď]

2. CURRENT CONTEXT (aktuální zprávy)
   - Zprávy podle AKTUÁLNÍHO filtru (server/channel/time)

3. CURRENT QUESTION
   - "A co konkrétně v investování?"
```

---

## 🎯 5. KOMPLETNÍ WORKFLOW - PŘÍKLAD

### Scenario 1: Základní použití

```
1. IMPORT (Configuration panel)
   Server: InvestičníFlow
   Channels: All channels
   Time: 720 hours
   → Fetch messages now
   
   VÝSLEDEK: Stáhlo 7,847 zpráv ze všech kanálů

2. CHAT (Chat Interface)
   TY: "Jaké byly hlavní témata diskuse?"
   
   SYSTÉM:
   ✓ Načte 1,563 zpráv (720h, all channels)
   ✓ ML filtering: Vybere 200 nejdůležitějších
   ✓ Context limit: Vezme 1000 zpráv max
   ✓ Pošle do Gemini API
   
   AI: [Strukturovaná odpověď s citacemi]

3. POKRAČOVÁNÍ KONVERZACE
   TY: "Můžeš to shrnout stručněji?"
   
   SYSTÉM:
   ✓ PAMATUJE SI předchozí odpověď
   ✓ Použije stejný context
   ✓ Vygeneruje stručnější verzi
```

### Scenario 2: Změna filtru během konverzace

```
1. CHAT - Round 1
   Filters: Server=InvestičníFlow, Channels=All, Time=720h
   TY: "Co bylo diskutováno celkově?"
   AI: [Odpověď ze VŠECH kanálů]

2. ZMĚNA FILTRU
   Channels: Změn na JEN #investování

3. CHAT - Round 2
   Filters: Server=InvestičníFlow, Channels=#investování, Time=720h
   TY: "A co konkrétně v kanále investování?"
   
   SYSTÉM:
   ✓ STÁLE PAMATUJE předchozí dotaz
   ✓ ALE načte NOVÝ context (jen #investování)
   ✓ Odpověď bude JEN z #investování
   
   AI: [Odpověď jen z #investování + kontext z předchozí konverzace]
```

### Scenario 3: Nový import během konverzace

```
1. CHAT
   TY: "Co bylo dnes?"
   AI: [Odpověď na základě starých dat]

2. NOVÝ IMPORT (mezitím přišly nové zprávy)
   → Fetch messages now
   VÝSLEDEK: Stáhlo 15 nových zpráv

3. CHAT
   TY: "A co nového?"
   
   SYSTÉM:
   ✓ PAMATUJE předchozí konverzaci
   ✓ ALE načte AKTUÁLNÍ data (včetně nových 15 zpráv)
   
   AI: [Odpověď včetně nových zpráv]
```

---

## 🔧 6. NASTAVITELNÉ LIMITY

### Context Limits (kolik zpráv jde do LLM)

```python
# streamlit_app.py řádek 631-641

if hours <= 24:
    max_context = 300 zpráv
elif hours <= 168:  # 1 týden
    max_context = 500 zpráv
else:  # 720+ hodin
    max_context = 1000 zpráv
```

### Import Limit

```python
# load_messages.py řádek 69
client.fetch_messages(
    channel_id,
    last_message_id,
    limit=5000  # Max 5000 zpráv per channel per import
)
```

### ML Filtering (Enable Smart Filtering)

```python
# Když je ZAPNUTÝ:
✓ Ohodnotí každou zprávu importance score
✓ Filtruje podle importance_threshold
✓ Použije jen důležité zprávy

# Když je VYPNUTÝ:
✓ Použije všechny zprávy (do context limitu)
```

---

## 📊 7. SUMMARY - Co se děje krok za krokem

### IMPORT PHASE:
```
1. Vyber server → získej všechny kanály
2. Pro každý kanál:
   - Stáhni zprávy (filter: time, last_message_id)
   - Ulož do databáze
3. Celkem: X,XXX zpráv uloženo
```

### QUERY PHASE:
```
1. UI načte zprávy z DB
   Filter: server_id, channel_ids, hours
   → Vrátí N zpráv

2. ML Filtering (optional)
   → Filtruje na M důležitých zpráv

3. Context Limit
   → Vezme max 300-1000 zpráv

4. Build References
   → Vytvoří očíslovaný seznam zpráv

5. Send to LLM
   Context: references + chat_history + question
   → Gemini API generates answer

6. Display Answer
   → Strukturovaná odpověď s citacemi
```

---

## ✅ FAQ

### Q: Když změním channel, začne nová konverzace?
**A: NE!** Konverzace pokračuje, ale **context se mění** podle nového filtru.

### Q: Stahují se zprávy při každém dotazu?
**A: NE!** Zprávy se stahují jen při "Fetch messages now". Chat query jen **čte z databáze**.

### Q: Kolik zpráv vidí LLM?
**A: 300-1000** podle time range (+ optional ML filtering).

### Q: Jak často mám dělat import?
**A: Podle potřeby:**
- Realtime sync: Automaticky každých X minut
- Manual: "Fetch messages now" když chceš nová data

### Q: Může LLM vidět zprávy ze všech serverů najednou?
**A: NE!** Vždy je vybraný JEN 1 server. Ale vidí všechny kanály z toho serveru (pokud je "All channels").

---

## 🎯 Pro efektivní použití:

1. **Import jednou denně** (nebo před důležitou analýzou)
2. **Začni s "All channels"** pro široký přehled
3. **Zužuj na konkrétní kanály** pro detaily
4. **Využij konverzaci** - ptej se postupně, LLM si pamatuje kontext
5. **Trust ML filtering** - zvýrazní důležité zprávy

---

**Máš další otázky?** Ptej se! 🚀
