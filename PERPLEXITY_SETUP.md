# 🌐 Perplexity API - Setup Guide

## 🚀 Quick Setup (3 kroky)

### 1. Získej Perplexity API Key

1. Jdi na: https://www.perplexity.ai/settings/api
2. Vytvoř nový API key
3. Zkopíruj ho

### 2. Přidej do `.env`

```bash
# Open .env file
nano .env

# Add this line:
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Save and exit (Ctrl+X, Y, Enter)
```

### 3. Restart Streamlit

```bash
# Kill current Streamlit
pkill -f "streamlit run"

# Start again
streamlit run streamlit_app.py
```

---

## ✅ Ověření že to funguje

### Test 1: Ticker Detection

```bash
python3 -c "
from lib.perplexity_enrichment import TickerDetector

# Test uppercase
print('Test 1:', TickerDetector.detect_tickers('Proč kupují PYPL?'))
# Expected: ['PYPL']

# Test lowercase
print('Test 2:', TickerDetector.detect_tickers('proč pypl?'))
# Expected: ['PYPL']

# Test dollar notation
print('Test 3:', TickerDetector.detect_tickers('Co je s \$BTC?'))
# Expected: ['BTC']

# Test multiple
print('Test 4:', TickerDetector.detect_tickers('PYPL vs AAPL'))
# Expected: ['PYPL', 'AAPL'] or similar
"
```

### Test 2: V Streamlit UI

1. Otevři aplikaci: http://localhost:8501
2. Vyber server s investičními diskusemi
3. Zadej: **"proč pypl?"**
4. Měl bys vidět:
   - 📊 Discord Context
   - 🌐 Real-Time Market Context (PYPL)
   - Status: "🌐 + Real-time market data"

---

## 🆘 Troubleshooting

### "PERPLEXITY_API_KEY not found"

**Problem:** API key není v .env nebo není načtený

**Solution:**
```bash
# Check .env file
cat .env | grep PERPLEXITY

# Should output:
# PERPLEXITY_API_KEY=pplx-xxxxx...

# If not there, add it:
echo "PERPLEXITY_API_KEY=your_key_here" >> .env

# Restart Streamlit
pkill -f "streamlit run" && streamlit run streamlit_app.py &
```

### "No ticker detected"

**Problem:** Systém nedetekoval ticker v dotazu

**Solution:**
```
❌ "proč to kupujou?"  → žádný ticker
✅ "proč pypl?"        → PYPL detected
✅ "proč PYPL?"        → PYPL detected
✅ "proč $PYPL?"       → PYPL detected
```

### "Perplexity API error"

**Problem:** API volání selhalo

**Solutions:**
1. **Check API key validity**: https://www.perplexity.ai/settings/api
2. **Check rate limits**: Max 10 calls/hour na free tier
3. **Check logs**:
```bash
# Run Streamlit in foreground to see logs
streamlit run streamlit_app.py

# Look for lines:
# INFO: Should enrich with Perplexity: True
# INFO: Detected tickers: ['PYPL']
# INFO: Detected ticker PYPL in question, enriching...
# ERROR: Perplexity API error: [error message]
```

---

## 📊 API Limits & Pricing

### Free Tier:
- 10 requests/hour
- Cache helps: same ticker = instant (cached 15 min)

### Paid Tier:
- 100+ requests/hour
- Faster responses
- Priority access

**Recommendation:** Start with free tier, upgrade if needed

---

## 🧪 Debug Mode

### Enable verbose logging:

```python
# In streamlit_app.py, change:
logging.basicConfig(level=logging.INFO)

# To:
logging.basicConfig(level=logging.DEBUG)

# Restart Streamlit
```

### Check logs in real-time:

```bash
# Run Streamlit without & (foreground)
streamlit run streamlit_app.py

# Ask question with ticker
# Watch console for:
INFO: Should enrich with Perplexity: True
INFO: Detected tickers: ['PYPL']
INFO: Detected ticker PYPL in question, enriching with Perplexity
INFO: Calling Perplexity API for PYPL
INFO: Successfully enriched answer with Perplexity for PYPL
```

---

## ✅ Success Checklist

- [ ] Perplexity API key obtained
- [ ] API key added to `.env`
- [ ] Streamlit restarted
- [ ] Ticker detection works (`python3 test above`)
- [ ] UI shows "🌐 + Real-time market data" status
- [ ] Enriched answer has both sections:
  - 📊 Discord Community Insight
  - 🌐 Real-Time Market Context

---

## 💡 Tips

1. **Cache is your friend**: Same ticker queries within 15min = instant
2. **Be specific**: "proč PYPL?" > "co myslíte?"
3. **Check rate limits**: 10 calls/hour = enough for normal use
4. **DYOR**: Perplexity provides data, but verify sources

---

**Ready to go! Ask questions with tickers and get enriched answers!** 🚀
