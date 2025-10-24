# 📚 Nápovědový systém - Implementace

## ✅ Co bylo přidáno

### 1. **Hlavní nápověda** (Sidebar Expander)

**Lokace**: Úplně nahoře v sidebaru, `📚 Nápověda & Návod`

**Obsah**:
```
- 🚀 Rychlý start (3 kroky)
- 📋 Nastavení (vysvětlení všech polí)
- 💬 Chat (jak používat konverzaci)
- 🔄 Import (jak stahovat zprávy)
- 💡 Tipy (best practices)
- 🆘 Problémy? (troubleshooting)
```

**Jak použít**:
- Klikni na `📚 Nápověda & Návod` v sidebaru
- Expander se rozbalí s kompletním návodem
- Defaultně je sbalený (expanded=False)

---

### 2. **Tooltips u každého pole**

#### Configuration Section

**Discord Server**
```
🎯 Vyber Discord server který chceš analyzovat. 
Zprávy se stahují jen z vybraného serveru.
```

**Channels**
```
📢 Filtruj podle kanálů. Prázdné = všechny kanály serveru. 
Vyber konkrétní kanály pro užší zaměření (např. jen #investování).
```

**Time Frame (hours)**
```
⏰ Časové okno pro analýzu. 24h = poslední den, 
168h = týden, 720h = měsíc. Ovlivňuje kolik zpráv se načte z databáze.
```

#### AI Features Section

**User ID**
```
👤 Tvé jedinečné ID pro personalizované hodnocení důležitosti zpráv. 
Např: 'user123' nebo tvé Discord jméno.
```

**Enable Smart Filtering**
```
🧠 Zapne AI filtrování - automaticky vybere nejdůležitější zprávy. 
Vypni pro zobrazení všech zpráv (může být pomalé).
```

**Importance Threshold**
```
🎯 Práh důležitosti zpráv. 0.0 = všechny zprávy | 
0.5 = střední důležitost | 1.0 = jen kritické zprávy. 
Doporučeno: 0.0-0.3
```

**Show Group Activities**
```
🎯 Zobrazí detekované skupinové aktivity: 
nákupy, eventy, konsenzy. Vyžaduje ML model.
```

#### Data Import Section

**Batch window (hours)**
```
📦 Kolik hodin zpětně stáhnout při manuálním importu. 
Např: 720h = stáhne zprávy za posledních 30 dní.
```

**Fetch messages now** (button)
```
📥 Klikni pro okamžité stažení zpráv z Discordu. 
Stahuje podle Batch window.
```

**Realtime lookback (hours)**
```
🔄 Při automatické synchronizaci se stahují zprávy nové než X hodin. 
Doporučeno: 1-2h pro aktuální konverzace.
```

**Refresh interval (seconds)**
```
⏱️ Jak často kontrolovat nové zprávy na Discordu (v sekundách). 
60s = každou minutu. Nižší hodnota = častější kontrola.
```

**Enable realtime sync**
```
🔁 Automaticky stahuje nové zprávy podle Refresh interval. 
Zapni pro živou synchronizaci s Discordem.
```

**Sync once now** (button)
```
⚡ Jednorázová synchronizace podle Realtime lookback. 
Použij pro rychlou kontrolu nových zpráv bez čekání na automatický interval.
```

---

## 🎯 Jak to používat

### Pro uživatele:

1. **Otevři aplikaci** (http://localhost:8501)
2. **Klikni na `📚 Nápověda & Návod`** v sidebaru
3. **Přečti si rychlý start** (3 kroky)
4. **Při vyplňování formuláře** najeď myší na ⓘ ikonu u každého pole
5. **Zobrazí se tooltip** s vysvětlením

### Pro vývojáře:

```python
# Přidat tooltip k poli:
st.sidebar.selectbox(
    "Název pole",
    options=...,
    help="📌 Tady je nápověda pro uživatele"
)

# Přidat tooltip k buttonu:
st.sidebar.button(
    "Název tlačítka",
    help="🔘 Nápověda co tlačítko dělá"
)

# Přidat expander s návodem:
with st.sidebar.expander("📚 Nápověda", expanded=False):
    st.markdown("""
    ### Nadpis
    
    Obsah nápovědy v markdown formátu...
    """)
```

---

## 📊 Statistiky

**Celkem tooltips**: 13  
**Hlavní nápověda**: 1 expander  
**Pokryté sekce**: 
- ✅ Configuration (3 pole)
- ✅ AI Features (4 pole)
- ✅ Data Import (6 polí/tlačítek)

**Emoji použité**: 🎯 📢 ⏰ 👤 🧠 🎯 📦 📥 🔄 ⏱️ 🔁 ⚡

---

## 💡 Best Practices

### Pro psaní tooltips:

1. **Začni emoji** - vizuální, rychlé rozpoznání
2. **První věta = CO to dělá** - jasně a stručně
3. **Druhá věta = JAK to použít** - praktický příklad
4. **Doporučení** - pokud je potřeba (např. "Doporučeno: 0.0-0.3")
5. **Maximálně 2-3 věty** - nepřekombinovat

### Příklad DOBRÉHO tooltipu:
```
⏰ Časové okno pro analýzu. 24h = poslední den, 168h = týden, 
720h = měsíc. Ovlivňuje kolik zpráv se načte z databáze.
```

### Příklad ŠPATNÉHO tooltipu:
```
Number of hours to look back for messages.
```
(Proč špatný: Není jasné co to konkrétně ovlivňuje, chybí příklady, není v češtině)

---

## 🔄 Aktualizace

### Když přidáš nové pole:

1. **Přidej `help` parameter** s tooltipem
2. **Použij relevantní emoji**
3. **Vysvětli CO a JAK**
4. **Testuj** - najeď myší a zkontroluj čitelnost

### Když změníš funkcionalitu:

1. **Aktualizuj tooltip** u příslušného pole
2. **Aktualizuj hlavní nápovědu** pokud je to důležitá změna
3. **Aktualizuj HOW_IT_WORKS.md** pokud mění celkový flow

---

## 📝 Checklist pro nové features

```markdown
Když přidávám nové pole/tlačítko:

- [ ] Přidal jsem `help` parameter
- [ ] Tooltip má emoji
- [ ] Tooltip vysvětluje CO pole dělá
- [ ] Tooltip vysvětluje JAK to použít
- [ ] Tooltip je v češtině
- [ ] Tooltip má max 2-3 věty
- [ ] Testoval jsem tooltip v UI
- [ ] Aktualizoval jsem hlavní nápovědu (pokud nutné)
```

---

## 🎨 Použité emoji a jejich význam

| Emoji | Význam | Kde použito |
|-------|--------|-------------|
| 🎯 | Target/Výběr | Discord Server, Importance |
| 📢 | Komunikace | Channels |
| ⏰ | Čas | Time Frame |
| 👤 | Uživatel | User ID |
| 🧠 | AI/Inteligence | Smart Filtering |
| 📦 | Balíček/Dávka | Batch window |
| 📥 | Stahování | Fetch messages |
| 🔄 | Synchronizace | Realtime lookback |
| ⏱️ | Časovač | Refresh interval |
| 🔁 | Smyčka/Opakování | Enable realtime |
| ⚡ | Rychlá akce | Sync once now |

---

## 🚀 Výsledek

**Před**: Uživatel neví co dělají pole, zkoušet náhodně  
**Po**: Každé pole má jasnou nápovědu, uživatel ví přesně co nastavit

**User Experience**: 📈 Významně zlepšeno!

---

## 📞 Support

Pokud uživatel stále nerozumí:
1. Zkontroluj tooltip - je dostatečně jasný?
2. Přidej příklad do hlavní nápovědy
3. Rozšiř HOW_IT_WORKS.md
4. Vytvoř video návod (budoucnost)

---

**Status**: ✅ Kompletní nápovědový systém implementován  
**Datum**: 2025-10-24  
**Testováno**: Ano  
**Production ready**: Ano
