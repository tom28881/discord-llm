import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from lib.llm import get_completion
from lib.database import (
    get_latest_message_timestamp,
    get_recent_message_records,
    init_db,
    get_servers,
    get_channels,
)
from lib.ml_serving import MLInferenceService
from lib.training_pipeline import TrainingScheduler, TrainingConfig
from lib.ml_system import ImportanceLevel
from datetime import datetime, timedelta
import logging
import pandas as pd
import time
from load_messages import load_messages_once

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('streamlit_app')

load_dotenv()
init_db()

google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize ML system (singleton pattern)
@st.cache_resource
def initialize_ml_system():
    """Initialize ML system components - cached to prevent reinitialization"""
    try:
        ml_service = MLInferenceService()
        training_config = TrainingConfig()
        
        # Try to load existing models
        if not ml_service.model_registry.get_active_model("champion"):
            logger.info("No trained models found. ML features will use heuristic fallback.")
        
        return ml_service, training_config
    except Exception as e:
        logger.error(f"Error initializing ML system: {e}")
        return None, None

# Get ML system
ml_service, training_config = initialize_ml_system()

def _format_timedelta(delta: timedelta) -> str:
    minutes_total = int(delta.total_seconds() // 60)
    if minutes_total <= 1:
        return "1 min ago"
    if minutes_total < 60:
        return f"{minutes_total} min ago"
    hours, minutes = divmod(minutes_total, 60)
    if hours < 24:
        return f"{hours} h {minutes} min ago"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h ago"


def _set_chat_prompt(value: str) -> None:
    st.session_state.pending_question = value or ""


def _generate_suggestions(
    records: List[Dict[str, Any]],
    channel_names: List[str],
    hours: int,
) -> List[str]:
    if not records:
        return []

    suggestions: List[str] = []
    channel_label = channel_names[0] if channel_names else "vybraných kanálech"
    scoped_hours = min(hours, 72)

    suggestions.append(
        f"Shrň dění v {channel_label} za posledních {scoped_hours} hodin."
    )

    suggestions.append(
        "Jaké akce nebo další kroky vyplývají z poslední konverzace?"
    )

    latest_text = records[-1]["content"].strip()
    if latest_text:
        snippet = latest_text[:90]
        if len(latest_text) > 90:
            snippet += "…"
        suggestions.append(f"Co řeší zpráva \"{snippet}\" a jaké jsou reakce?")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_suggestions: List[str] = []
    for item in suggestions:
        if item not in seen:
            seen.add(item)
            unique_suggestions.append(item)

    return unique_suggestions[:3]


def summarize_messages(
    references: list,
    chat_history: list,
    question: str,
    metadata: Optional[dict] = None,
) -> str:
    if not references and not chat_history:
        return "No references found."

    enumerated_messages = "\n".join(
        f"{idx + 1}. {ref}" for idx, ref in enumerate(references)
    )

    history_entries = [
        f"{msg['role']}: {msg['content']}" for msg in chat_history[:-1]
    ]
    history_text = "\n".join(history_entries).strip()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    prompt = (
        "Jsi profesionální analytik který analyzuje Discord konverzace a poskytuje přesné, strukturované odpovědi.\n\n"
        
        "**PRAVIDLA:**\n"
        "1. Odpovídej VÝHRADNĚ na základě poskytnutých zpráv\n"
        "2. Buď konkrétní, přesný a faktický\n"
        "3. Cituj čísla zpráv [1], [2] jako důkaz\n"
        "4. NEZAHRNUJ technické detaily (URLs, message IDs, raw data)\n"
        "5. Pokud informace chybí, řekni to explicitně\n"
        "6. Odpovídej v češtině, profesionálně a srozumitelně\n\n"
        
        f"<metadata>\nServer: {metadata.get('server', 'N/A')}\n"
        f"Časové období: {metadata.get('time_range_hours', 'N/A')} hodin\n"
        f"Analyzováno zpráv: {metadata.get('total_messages_considered', 0)}\n"
        f"Použito pro odpověď: {metadata.get('messages_used_for_answer', 0)}\n</metadata>\n\n"
        
        f"<zpravy>\n{enumerated_messages}\n</zpravy>\n\n"
        
        f"<historie_chatu>\n{history_text}\n</historie_chatu>\n\n"
        
        f"<dotaz>\n{question}\n</dotaz>\n\n"
        
        "**FORMÁT ODPOVĚDI:**\n\n"
        "## Shrnutí\n"
        "[Stručný, jasný a faktický přehled hlavních bodů]\n\n"
        
        "## Klíčové body\n"
        "[Seznam konkrétních zjištění s citacemi, např:]\n"
        "• **Bod 1:** [popis] [1, 3]\n"
        "• **Bod 2:** [popis] [5]\n\n"
        
        "## Detaily\n"
        "[Podrobnější rozvedení relevantních informací]\n\n"
        
        "## Evidence\n"
        "[Seznam relevantních výňatků ze zpráv:]\n"
        "• **[1]** \"[krátký čitelný excerpt]\" → [co to znamená]\n"
        "• **[2]** \"[krátký čitelný excerpt]\" → [co to znamená]\n\n"
        
        "## Závěr\n"
        "**Spolehlivost:** [Vysoká/Střední/Nízká] | **Pokrytí tématu:** [X%]\n"
        "[Finální shrnutí nebo doporučení]\n\n"
        
        "⚠️ DŮLEŽITÉ: Nepoužívej technické odkazy, URL linky ani Discord message IDs v odpovědi. "
        "Vše formuluj čitelně a profesionálně."
    )

    try:
        summary = get_completion(prompt)
        return summary

    except Exception as e:
        logger.error(f"Error summarizing messages: {e}")
        return f"An error occurred while summarizing the messages: {str(e)}"


def _run_import(mode: str, hours_back: Optional[int], show_spinner: bool = True) -> Dict[str, Any]:
    """Run load_messages_once and update session state metadata."""

    def _execute() -> Dict[str, Any]:
        return load_messages_once(
            server_id=str(st.session_state.server_id),
            sleep_between_servers=False,
            sleep_between_channels=False,
            channel_ids=st.session_state.channel_ids,
            hours_back=hours_back,
        )

    if show_spinner:
        with st.spinner("Importing messages from Discord..."):
            summary = _execute()
    else:
        summary = _execute()

    st.session_state.last_import_summary = summary
    st.session_state.last_import_time = datetime.now().isoformat()
    st.session_state.last_import_mode = mode
    st.session_state.last_import_hours = hours_back
    st.session_state.last_import_channels = st.session_state.channel_ids
    return summary


def _refresh_recent_records() -> Dict[str, Any]:
    """Load recent records for the active filters and cache in session state."""

    server_id = st.session_state.get("server_id")
    if not server_id:
        st.session_state.recent_records = []
        st.session_state.recent_messages = []
        st.session_state.recent_records_metadata = {}
        return {"records": [], "metadata": {}, "total": 0}

    selected_channel_ids = st.session_state.get("channel_ids")

    try:
        records = get_recent_message_records(
            server_id=server_id,
            hours=st.session_state.hours,
            keywords=None,
            channel_ids=selected_channel_ids,
            limit=None,  # No limit - show all messages in time range
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error fetching recent messages for preview: %s", exc)
        records = []

    records = sorted(records, key=lambda r: r["sent_at"]) if records else []
    st.session_state.recent_records = records
    st.session_state.recent_messages = [rec["content"] for rec in records]

    metadata: Dict[str, Any] = {
        "message_count": len(records),
        "channel_ids": selected_channel_ids or [],
        "hours": st.session_state.hours,
    }

    if records:
        earliest_dt = datetime.fromtimestamp(records[0]["sent_at"])
        latest_dt = datetime.fromtimestamp(records[-1]["sent_at"])
        metadata.update(
            {
                "earliest_message_iso": earliest_dt.isoformat(),
                "latest_message_iso": latest_dt.isoformat(),
            }
        )
    else:
        metadata.update(
            {
                "earliest_message_iso": None,
                "latest_message_iso": None,
            }
        )

    st.session_state.recent_records_metadata = metadata
    return {"records": records, "metadata": metadata, "total": len(records)}


def load_configuration():
    st.sidebar.header("Configuration")

    servers = get_servers()
    server_dict = dict(servers)  # Convert list of tuples to dict for easy lookup

    if 'server_id' not in st.session_state:
        st.session_state.server_id = servers[0][0] if servers else ""
    if 'hours' not in st.session_state:
        st.session_state.hours = 720  # Default to 720 hours
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "default_user"
    if 'use_ml_filtering' not in st.session_state:
        st.session_state.use_ml_filtering = True
    if 'importance_threshold' not in st.session_state:
        st.session_state.importance_threshold = 0.5
    if 'channel_ids' not in st.session_state:
        st.session_state.channel_ids = None
    if 'channel_lookup' not in st.session_state:
        st.session_state.channel_lookup = {}
    if 'server_name' not in st.session_state:
        st.session_state.server_name = server_dict.get(st.session_state.server_id, "")
    if 'batch_hours' not in st.session_state:
        st.session_state.batch_hours = 24
    if 'realtime_hours' not in st.session_state:
        st.session_state.realtime_hours = 1
    if 'realtime_interval' not in st.session_state:
        st.session_state.realtime_interval = 30
    if 'realtime_enabled' not in st.session_state:
        st.session_state.realtime_enabled = False
    if 'realtime_status' not in st.session_state:
        st.session_state.realtime_status = "Realtime sync inactive"
    if 'last_import_summary' not in st.session_state:
        st.session_state.last_import_summary = None
    if 'last_import_time' not in st.session_state:
        st.session_state.last_import_time = None
    if 'last_import_mode' not in st.session_state:
        st.session_state.last_import_mode = None
    if 'last_import_hours' not in st.session_state:
        st.session_state.last_import_hours = None
    if 'last_import_channels' not in st.session_state:
        st.session_state.last_import_channels = None

    with st.sidebar.expander(" Nápověda & Návod", expanded=False):
        st.markdown("""
        ### Rychlý start
        
        1. **Vyber server** → Zvol Discord server
        2. **Stáhni zprávy** → "Fetch messages now"
        3. **Zadej dotaz** → V Chat Interface
        
        ### Nastavení
        
        **Discord Server**  
        Vyber server ze kterého chceš analyzovat zprávy.
        
        **Channels**  
        - Prázdné = Všechny kanály
        - Vyber konkrétní pro filtrování
        
        **Time Frame**  
        Kolik hodin zpětně načíst (1-720h)
        
        **Smart Filtering** 
        AI filtruje důležité zprávy. Vypni pro všechny zprávy.
        
        **Importance Threshold**  
        0.0 = všechny | 1.0 = jen nejdůležitější
        
        ### Chat
        
        - **Ptej se v češtině** - AI odpovídá česky
        - **Konverzace se pamatuje** - Můžeš navazovat
        - **Změní-li se filtr** - Context se aktualizuje
        
        ### Import
        
        **Fetch messages now**  
        Stáhne zprávy z Discord (podle Batch window)
        
        **Realtime Sync**  
        Automaticky stahuje nové zprávy
        
        ### Tipy
        
        - Začni široko (All channels, 720h)
        - Zužuj postupně (konkrétní kanály)
        - Využij konverzaci (ptej se postupně)
        - Důvěřuj Smart Filtering
        
        ### Problémy?
        
        - **Žádné zprávy?** → Zkontroluj Time Frame
        - **Pomalé?** → Zmenši Time Frame nebo zapni Smart Filtering
        - **Nefunguje AI?** → Zkontroluj GOOGLE_API_KEY v .env
        """)

    st.sidebar.header("Configuration")

    if servers:
        server_names = [name for _, name in servers]
        previous_server_id = st.session_state.server_id
        selected_name = st.sidebar.selectbox(
            "Discord Server",
            server_names,
            index=server_names.index(server_dict[st.session_state.server_id]) if st.session_state.server_id in server_dict else 0,
            help=" Vyber Discord server který chceš analyzovat. Zprávy se stahují jen z vybraného serveru."
        )
        # Update server_id based on selected name
        selected_server_id = [id for id, name in servers if name == selected_name][0]
        if selected_server_id != previous_server_id:
            st.session_state.channel_ids = None
            st.session_state.channel_lookup = {}
        st.session_state.server_id = selected_server_id
        st.session_state.server_name = selected_name

        channels = get_channels(st.session_state.server_id)
        if channels:
            channel_lookup = {channel_id: name for channel_id, name in channels}
            st.session_state.channel_lookup = channel_lookup

            label_to_id = {f"#{name}": channel_id for channel_id, name in channels}
            option_labels = list(label_to_id.keys())

            previously_selected = st.session_state.channel_ids or []
            default_labels = [
                label for label, channel_id in label_to_id.items()
                if channel_id in previously_selected
            ]

            selected_labels = st.sidebar.multiselect(
                "Channels",
                options=option_labels,
                default=default_labels,
                help="📢 Filtruj podle kanálů. Prázdné = všechny kanály serveru. Vyber konkrétní kanály pro užší zaměření (např. jen #investování)."
            )

            if selected_labels:
                st.session_state.channel_ids = [label_to_id[label] for label in selected_labels]
            else:
                st.session_state.channel_ids = None

    hours = st.sidebar.number_input(
        "Time Frame (hours)",
        min_value=1,
        max_value=720,
        value=st.session_state.hours,
        help="⏰ Časové okno pro analýzu. 24h = poslední den, 168h = týden, 720h = měsíc. Ovlivňuje kolik zpráv se načte z databáze."
    )
    st.session_state.hours = hours
    
    # ML Configuration Section
    st.sidebar.header("🤖 AI Features")
    
    # User ID for personalization
    user_id = st.sidebar.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="👤 Tvé jedinečné ID pro personalizované hodnocení důležitosti zpráv. Např: 'user123' nebo tvé Discord jméno."
    )
    st.session_state.user_id = user_id
    
    # ML filtering toggle
    use_ml_filtering = st.sidebar.checkbox(
        "Enable Smart Filtering",
        value=st.session_state.use_ml_filtering,
        help="🧠 Zapne AI filtrování - automaticky vybere nejdůležitější zprávy. Vypni pro zobrazení všech zpráv (může být pomalé)."
    )
    st.session_state.use_ml_filtering = use_ml_filtering
    
    if use_ml_filtering:
        # Importance threshold
        importance_threshold = st.sidebar.slider(
            "Importance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.importance_threshold,
            step=0.1,
            help="🎯 Práh důležitosti zpráv. 0.0 = všechny zprávy | 0.5 = střední důležitost | 1.0 = jen kritické zprávy. Doporučeno: 0.0-0.3"
        )
        st.session_state.importance_threshold = importance_threshold
        
        # Show activity detection
        show_activities = st.sidebar.checkbox(
            "Show Group Activities",
            value=True,
            help="🎯 Zobrazí detekované skupinové aktivity: nákupy, eventy, konsenzy. Vyžaduje ML model."
        )
        st.session_state.show_activities = show_activities
        
        # Show ML system status
        if ml_service:
            health_status = ml_service.get_system_health()
            champion_model = health_status.get("models", {}).get("champion")
            
            if champion_model:
                st.sidebar.success("✅ ML System Active")
                champion_perf = health_status.get("performance", {}).get("champion", {})
                accuracy = champion_perf.get("accuracy", 0)
                if accuracy > 0:
                    st.sidebar.metric("Model Accuracy", f"{accuracy:.1%}")
        else:
            st.sidebar.warning("⚠️ Using Heuristic Fallback")
    else:
        st.sidebar.error("❌ ML System Unavailable")

    st.sidebar.header("Data Import")

    st.session_state.batch_hours = st.sidebar.slider(
        "Batch window (hours)",
        min_value=1,
        max_value=720,
        value=st.session_state.batch_hours,
        help="📦 Kolik hodin zpětně stáhnout při manuálním importu. Např: 720h = stáhne zprávy za posledních 30 dní."
    )

    if st.session_state.server_id:
        if st.sidebar.button("Fetch messages now", use_container_width=True, help="📥 Klikni pro okamžité stažení zpráv z Discordu. Stahuje podle Batch window."):
            summary = _run_import("batch", st.session_state.batch_hours)
            st.sidebar.success(f"Imported {summary['messages_saved']} messages")
    else:
        st.sidebar.info("Select a server to enable importing.")

    st.sidebar.subheader("Realtime Sync")
    st.session_state.realtime_hours = st.sidebar.slider(
        "Realtime lookback (hours)",
        min_value=1,
        max_value=24,
        value=st.session_state.realtime_hours,
        help="🔄 Při automatické synchronizaci se stahují zprávy nové než X hodin. Doporučeno: 1-2h pro aktuální konverzace."
    )

    st.session_state.realtime_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=10,
        max_value=300,
        step=5,
        value=st.session_state.realtime_interval,
        help="⏱️ Jak často kontrolovat nové zprávy na Discordu (v sekundách). 60s = každou minutu. Nižší hodnota = častější kontrola."
    )

    realtime_toggle = st.sidebar.toggle(
        "Enable realtime sync",
        value=st.session_state.realtime_enabled,
        help="🔁 Automaticky stahuje nové zprávy podle Refresh interval. Zapni pro živou synchronizaci s Discordem."
    )
    if realtime_toggle != st.session_state.realtime_enabled:
        st.session_state.realtime_enabled = realtime_toggle
        if not realtime_toggle:
            st.session_state.realtime_status = "Realtime sync inactive"

    if st.session_state.server_id:
        if st.sidebar.button(
            "Sync once now",
            key="sync_once_button",
            use_container_width=True,
            help="⚡ Jednorázová synchronizace podle Realtime lookback. Použij pro rychlou kontrolu nových zpráv bez čekání na automatický interval."
        ):
            summary = _run_import("manual", st.session_state.realtime_hours)
            st.sidebar.success(f"Synced {summary['messages_saved']} messages")
            st.session_state.realtime_status = f"Manual sync at {datetime.now():%H:%M:%S}"

    if st.session_state.server_id and st.session_state.realtime_enabled:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore
        except ImportError:
            st_autorefresh = None

        if st_autorefresh:
            st_autorefresh(
                interval=st.session_state.realtime_interval * 1000,
                key="realtime_autorefresh",
            )
        else:
            st.sidebar.warning(
                "Install 'streamlit-autorefresh' for automatic refreshes."
            )

        summary = _run_import(
            "realtime",
            st.session_state.realtime_hours,
            show_spinner=False,
        )
        st.session_state.realtime_status = (
            f"Realtime sync {datetime.now():%H:%M:%S} | +{summary['messages_saved']} messages"
        )

    if st.session_state.last_import_summary:
        try:
            ts_display = datetime.fromisoformat(st.session_state.last_import_time).strftime("%Y-%m-%d %H:%M:%S") if st.session_state.last_import_time else "unknown"
        except Exception:
            ts_display = st.session_state.last_import_time or "unknown"
        st.sidebar.caption(
            f"Last import ({st.session_state.last_import_mode or 'n/a'}) at {ts_display}. Saved {st.session_state.last_import_summary.get('messages_saved', 0)} messages."
        )

    st.sidebar.caption(st.session_state.realtime_status)

    latest_ts = get_latest_message_timestamp(
        st.session_state.server_id,
        channel_ids=st.session_state.channel_ids,
    )
    if latest_ts:
        latest_dt = datetime.fromtimestamp(latest_ts)
        freshness = _format_timedelta(datetime.now() - latest_dt)
        st.sidebar.caption(
            f"📥 Latest message imported: {latest_dt:%Y-%m-%d %H:%M} ({freshness})"
        )
    else:
        st.sidebar.caption("No messages imported yet for the selected scope.")

def display_chat():
    st.header("Chat Interface")

    messages = st.session_state.get("messages", [])

    if st.session_state.get("last_import_summary"):
        try:
            ts = st.session_state.get("last_import_time")
            ts_label = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "unknown"
        except Exception:
            ts_label = st.session_state.get("last_import_time", "unknown")
        st.caption(
            f"Last import ({st.session_state.get('last_import_mode', 'n/a')}) at {ts_label}."
        )

    if st.session_state.get("realtime_status"):
        st.caption(st.session_state.realtime_status)

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "recent_records" not in st.session_state:
        st.session_state.recent_records = []
    if "recent_records_metadata" not in st.session_state:
        st.session_state.recent_records_metadata = {}

    with st.spinner("Loading recent messages..."):
        recent_info = _refresh_recent_records()

    recent_records = recent_info["records"]
    total_considered = recent_info["total"]

    channel_lookup = st.session_state.get("channel_lookup", {})
    selected_channels = st.session_state.get("channel_ids")
    if selected_channels:
        channel_names = [
            f"#{channel_lookup.get(ch_id, ch_id)}" for ch_id in selected_channels
        ]
    else:
        channel_names = ["All channels"]

    if recent_records:
        preview_header = (
            f"Loaded {len(recent_records)} messages from the last {st.session_state.hours} hours"
        )
        st.caption(preview_header)
        with st.expander("Recent messages preview", expanded=False):
            for record in recent_records[-10:]:
                channel_name = channel_lookup.get(record["channel_id"], str(record["channel_id"]))
                ts = datetime.fromtimestamp(record["sent_at"]).strftime("%Y-%m-%d %H:%M")
                st.markdown(f"**[{ts}] #{channel_name}** {record['content']}")
    else:
        st.info("V zadaném rozsahu nebyly nalezeny žádné zprávy.")

    suggestions = _generate_suggestions(
        recent_records,
        channel_names,
        st.session_state.hours,
    )
    if suggestions:
        st.markdown("### 💡 Doporučené dotazy")
        cols = st.columns(min(len(suggestions), 3))
        for idx, suggestion in enumerate(suggestions):
            col = cols[idx % len(cols)]
            col.button(
                suggestion,
                key=f"chat_suggestion_{idx}",
                on_click=_set_chat_prompt,
                args=(suggestion,),
            )

    pending_prompt = st.session_state.pop("pending_question", None)
    prompt = st.chat_input("Enter your question")
    if not prompt:
        prompt = pending_prompt
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    messages.append({"role": "user", "content": prompt})
    st.session_state.messages = messages

    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    keywords = None

    records = st.session_state.get("recent_records", [])
    filtered_records = records
    importance_scores: List[Tuple[float, float, int]] = []
    importance_summary: Optional[Tuple[int, float, int]] = None

    if records and st.session_state.get("use_ml_filtering", False) and ml_service:
        scored_records = []
        for record in records:
            message_data = {
                "id": record["id"],
                "content": record["content"],
                "server_id": st.session_state.server_id,
                "channel_id": record["channel_id"],
                "sent_at": record["sent_at"],
            }
            try:
                prediction = ml_service.predict_message_importance(
                    st.session_state.user_id,
                    message_data,
                )
                scored_records.append(
                    {
                        **record,
                        "importance": prediction.personalized_score,
                        "confidence": prediction.confidence,
                        "level": prediction.importance_level,
                    }
                )
            except Exception as exc:
                logger.warning("Error scoring message %s: %s", record["id"], exc)
                scored_records.append(
                    {
                        **record,
                        "importance": 0.5,
                        "confidence": 0.3,
                        "level": 1,
                    }
                )

        threshold = st.session_state.importance_threshold
        filtered_records = [rec for rec in scored_records if rec["importance"] >= threshold]

        if not filtered_records:
            filtered_records = sorted(scored_records, key=lambda x: x["importance"], reverse=True)[:100]
        else:
            filtered_records = sorted(filtered_records, key=lambda x: x["importance"], reverse=True)[:200]

        filtered_records.sort(key=lambda r: r["sent_at"])
        importance_scores = [
            (rec["importance"], rec["confidence"], rec["level"])
            for rec in filtered_records
        ]
        st.session_state.message_scores = importance_scores
        if importance_scores:
            avg_importance = sum(score for score, _, _ in importance_scores) / len(importance_scores)
            high_importance = sum(1 for score, _, _ in importance_scores if score > 0.7)
            importance_summary = (len(importance_scores), avg_importance, high_importance)
    else:
        st.session_state.message_scores = []

    # Dynamic context limit based on time range
    hours = st.session_state.get("hours", 720)
    if hours <= 24:
        max_context = 300
    elif hours <= 168:  # 1 week
        max_context = 500
    else:  # 720+ hours
        max_context = 1000
    
    if len(filtered_records) > max_context:
        filtered_records = filtered_records[-max_context:]

    if not filtered_records:
        assistant_reply = "V zadaném rozsahu jsem nenašel žádné zprávy."
        with st.chat_message("assistant"):
            st.warning(assistant_reply)
        messages.append({"role": "assistant", "content": assistant_reply})
        st.session_state.messages = messages
        st.session_state.recent_messages = []
        return

    st.session_state.recent_messages = [rec["content"] for rec in filtered_records]

    channel_lookup = st.session_state.get("channel_lookup", {})
    references = []
    for record in filtered_records:
        channel_name = channel_lookup.get(record["channel_id"], str(record["channel_id"]))
        # Clean timestamp format (just date + time)
        timestamp = datetime.fromtimestamp(record["sent_at"]).strftime("%d.%m %H:%M")
        # Clean and limit content length
        content = record['content'].strip()
        # Remove excessive whitespace
        content = " ".join(content.split())
        # Limit length for cleaner context
        if len(content) > 300:
            content = content[:297] + "..."
        references.append(f"[{timestamp}] #{channel_name}: {content}")

    base_metadata = st.session_state.get("recent_records_metadata", {})
    metadata = {
        "server": st.session_state.get("server_name"),
        "channels": channel_names,
        "time_range_hours": st.session_state.hours,
        "total_messages_considered": total_considered,
        "messages_used_for_answer": len(filtered_records),
        **base_metadata,
    }
    metadata["channel_count"] = len(channel_names)
    if importance_summary:
        count, avg_importance, high_importance = importance_summary
        metadata["importance_stats"] = {
            "evaluated_messages": count,
            "average_score": round(avg_importance, 3),
            "high_importance_count": high_importance,
            "threshold": st.session_state.importance_threshold,
        }

    if (
        st.session_state.get("show_activities", False)
        and st.session_state.get("use_ml_filtering", False)
        and ml_service
    ):
        with st.spinner("Detecting group activities..."):
            try:
                activities_prediction = ml_service.detect_group_activities(
                    st.session_state.server_id,
                    st.session_state.hours,
                )

                if activities_prediction.activities or activities_prediction.fomo_moments:
                    with st.expander("🎯 Detected Group Activities", expanded=True):
                        for activity in activities_prediction.activities[:5]:
                            if activity.urgency_score > 0.7:
                                emoji = "🔥"
                            elif activity.urgency_score > 0.4:
                                emoji = "⚡"
                            else:
                                emoji = "💬"
                            st.markdown(
                                f"**{emoji} {activity.activity_type.title()} Activity**\n"
                                f"- {activity.summary}\n"
                                f"- Participants: {len(activity.participants)}\n"
                                f"- Confidence: {activity.confidence:.1%}\n"
                                f"- Duration: {activity.time_span}"
                            )

                        if activities_prediction.fomo_moments:
                            st.markdown("### 🚨 FOMO Alert!")
                            for fomo in activities_prediction.fomo_moments[:3]:
                                st.error(f"**{fomo.activity_type.title()}**: {fomo.summary}")
            except Exception as exc:
                logger.warning("Error detecting activities: %s", exc)

    with st.spinner("Summarizing..."):
        summary = summarize_messages(
            references=references,
            chat_history=messages,
            question=prompt,
            metadata=metadata,
        )

    analyzed_caption = (
        f"Analyzed {total_considered} messages; using {len(filtered_records)} for the answer."
    )

    with st.chat_message("assistant"):
        if importance_summary:
            _, avg_importance, high_importance = importance_summary
            st.info(
                f"📊 {analyzed_caption} Avg importance: {avg_importance:.1%} | High priority: {high_importance}"
            )
        else:
            st.caption(f"📊 {analyzed_caption}")

        st.markdown(summary)

    messages.append({"role": "assistant", "content": summary})
    st.session_state.messages = messages

def display_ml_admin():
    """Display ML system administration interface"""
    
    st.header("🔧 ML System Administration")
    
    # Help section at the top
    with st.expander("📚 Co je ML Administration & Jak na to?", expanded=False):
        st.markdown("""
        ### 🤖 Co to dělá?
        
        ML systém **automaticky hodnotí důležitost Discord zpráv** a filtruje je pro tebe.
        
        **Výsledek:** Vidíš jen důležité zprávy místo tisíců spamu! 🎯
        
        ---
        
        ### 🎓 Jak na to (3 kroky):
        
        **1. Zapni Smart Filtering** (v Configuration)
        - Zapne AI filtrování zpráv
        - Vidíš jen důležité zprávy
        
        **2. Poskytuj Feedback** (níže v "Provide Feedback")
        - Ohodnoť 5-10 zpráv (🔇 Noise → 🔥 Urgent)
        - Model se učí tvé preference
        - Čím více feedbacku, tím lepší výsledky!
        
        **3. Trénuj model** (1x týdně)
        - Klikni "Incremental Update"
        - Model se aktualizuje s tvým feedbackem
        - Accuracy se zlepší!
        
        ---
        
        ### 📊 Co znamenají sekce?
        
        **System Health** 📊  
        Ukazuje aktuální stav modelu (přesnost, verze)
        
        **Training Controls** 🚀  
        Tlačítka pro trénování a validaci modelu
        
        **Provide Feedback** 👍  
        Ohodnoť zprávy aby se model zlepšil
        
        ---
        
        ### 💡 Tipy:
        
        - Začni s feedbackem (10+ zpráv)
        - Po feedbacku klikni "Incremental Update"
        - Sleduj jak se zlepšuje Accuracy
        - Opakuj týdně pro nejlepší výsledky
        
        ### ⚠️ Poznámka:
        
        ML systém je v **BETA** fázi. Funguje, ale není dokonalý (71% accuracy).
        Potřebuje tvůj feedback pro zlepšení!
        """)
    
    if not ml_service:
        st.error("❌ ML system není dostupný. Zkontroluj instalaci knihoven.")
        st.info("💡 Tip: Spusť `pip install -r requirements.txt`")
        return
    
    # Training Status (if any training is running)
    if 'training_jobs' not in st.session_state:
        st.session_state.training_jobs = []
    
    # Check for running jobs
    running_jobs = [job for job in st.session_state.training_jobs if job.get('status') == 'running']
    
    # Auto-refresh when jobs are running
    if running_jobs:
        try:
            from streamlit_autorefresh import st_autorefresh
            # Refresh every 30 seconds
            st_autorefresh(interval=30000, key="ml_training_refresh")
        except ImportError:
            # Fallback: show manual refresh instruction
            pass
    
    if running_jobs:
        with st.expander("🔄 Training Progress (Průběh tréninku)", expanded=True):
            st.success("✨ Auto-refresh zapnutý! Stránka se obnovuje každých 30 sekund.")
            st.caption("💡 Pro manuální refresh stiskni F5 nebo klikni tlačítko níže.")
            
            for idx, job in enumerate(running_jobs):
                st.markdown(f"### 🏃 {job['type'].upper().replace('_', ' ')} - běží...")
                
                # Calculate estimated progress based on elapsed time
                elapsed = (datetime.now() - job['started_at']).total_seconds()
                
                # Estimate progress (rough estimation)
                if job['type'] == 'incremental':
                    estimated_duration = 180  # 3 minutes
                elif job['type'] == 'validation':
                    estimated_duration = 120  # 2 minutes
                else:  # full_retrain
                    estimated_duration = 7200  # 2 hours
                
                estimated_progress = min(int((elapsed / estimated_duration) * 100), 99)
                job['progress'] = estimated_progress
                
                # Progress bar with animation
                st.progress(estimated_progress / 100, text=f"Pokrok: {estimated_progress}% (odhad)")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Job ID", f"...{job['job_id'][-8:]}")
                with col2:
                    job_type_display = job['type'].replace('_', ' ').title()
                    st.metric("Typ", job_type_display)
                with col3:
                    minutes_elapsed = int(elapsed // 60)
                    seconds_elapsed = int(elapsed % 60)
                    st.metric("Běží", f"{minutes_elapsed}m {seconds_elapsed}s")
                with col4:
                    minutes_remaining = max(0, int((estimated_duration - elapsed) // 60))
                    st.metric("Zbývá", f"~{minutes_remaining}m")
                
                st.caption(f"⏱️ Začalo: {job['started_at'].strftime('%H:%M:%S')} | 🎯 Typ: {job_type_display}")
                
                # Actions
                col_action1, col_action2 = st.columns(2)
                with col_action1:
                    if st.button(f"✅ Označit jako hotové", key=f"complete_{idx}", help="Klikni když víš že job je dokončený"):
                        job['status'] = 'completed'
                        st.success("✅ Job označen jako dokončený!")
                        st.rerun()
                
                with col_action2:
                    if st.button(f"🔄 Refresh status", key=f"refresh_{idx}", help="Manuální obnovení statusu"):
                        st.rerun()
                
                st.divider()
    
    # System Health
    with st.expander("📊 System Health (Stav modelu)", expanded=True):
        st.info("👁️ **Co vidím zde:** Aktuální stav ML modelu - která verze běží a jak je přesná.")
        
        health = ml_service.get_system_health()
        
        col1, col2 = st.columns(2)
        
        with col1:
            champion = health["models"].get("champion", "None")
            st.metric(
                "Champion Model",
                champion if champion else "❌ Žádný model",
                help="🏆 Aktuálně používaný model v produkci. Tento model hodnotí zprávy."
            )
            champion_perf = health["performance"].get("champion", {})
            if champion_perf.get("accuracy"):
                accuracy = champion_perf['accuracy']
                st.metric(
                    "Champion Accuracy",
                    f"{accuracy:.1%}",
                    help="🎯 Přesnost modelu. 85%+ = výborné, 70-85% = dobré, <70% = potřeba zlepšit feedbackem."
                )
                
                # Accuracy interpretation
                if accuracy >= 0.85:
                    st.success("✅ Výborná přesnost!")
                elif accuracy >= 0.70:
                    st.warning("⚠️ Dobrá přesnost, ale můžeš zlepšit feedbackem.")
                else:
                    st.error("❌ Nízká přesnost! Poskytni více feedbacku.")
        
        with col2:
            challenger = health["models"].get("challenger", "None")
            st.metric(
                "Challenger Model",
                challenger if challenger else "❌ Žádný",
                help="🥈 Testovací model. Pokud je lepší než Champion, nahradí ho."
            )
            challenger_perf = health["performance"].get("challenger", {})
            if challenger_perf.get("accuracy"):
                st.metric(
                    "Challenger Accuracy",
                    f"{challenger_perf['accuracy']:.1%}",
                    help="🎯 Přesnost testovacího modelu."
                )
    
    # Training Controls
    with st.expander("🚀 Training Controls (Trénování modelu)", expanded=False):
        st.info("🎓 **Kdy použít:** Po poskytnutí feedbacku (10+ zpráv) klikni 'Incremental Update' pro aktualizaci modelu.")
        
        st.markdown("""
        **📋 Co dělají tlačítka:**
        - **Start Full Retraining** → Kompletní přetrénování (hodiny, jen když máš 1000+ feedbacků)
        - **Incremental Update** ⭐ → Rychlá aktualizace (minuty, použij toto!)
        - **Model Validation** → Ověř přesnost modelu
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔄 Full Retraining**")
            st.caption("Trvání: Hodiny | Kdy: 1000+ feedbacků")
            if st.button("Start Full Retraining", help="🔄 Kompletní přetrénování modelu od nuly. Použij JEN když máš hodně dat (1000+ feedbacků). Trvá několik hodin."):
                try:
                    from lib.training_pipeline import TrainingScheduler
                    from lib.ml_serving import ModelRegistry, PerformanceMonitor
                    
                    scheduler = TrainingScheduler(
                        training_config,
                        ml_service.model_registry,
                        ml_service.performance_monitor
                    )
                    job_id = scheduler.schedule_training_job("full_retrain", priority=True)
                    
                    # Track job in session state
                    st.session_state.training_jobs.append({
                        'job_id': job_id,
                        'type': 'full_retrain',
                        'status': 'running',
                        'progress': 0,
                        'started_at': datetime.now()
                    })
                    
                    st.success(f"✅ Training job scheduled: {job_id}")
                    st.info("⏳ Trénování běží na pozadí. Scroll nahoru a rozbal '🔄 Training Progress' pro sledování.")
                    st.info("🔄 Auto-refresh: Stránka se automaticky obnoví každých 30s")
                    
                    # Force rerun to show progress section
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error scheduling training: {e}")
        
        with col2:
            st.markdown("**⚡ Incremental Update** ⭐")
            st.caption("Trvání: Minuty | Kdy: Po feedbacku")
            if st.button("Incremental Update", type="primary", help="⚡ Doporučeno! Rychlá aktualizace modelu s novým feedbackem. Použij po každém poskytnutí 10+ feedbacků. Trvá jen pár minut."):
                try:
                    from lib.training_pipeline import TrainingScheduler
                    
                    scheduler = TrainingScheduler(
                        training_config,
                        ml_service.model_registry,
                        ml_service.performance_monitor
                    )
                    job_id = scheduler.schedule_training_job("incremental")
                    
                    # Track job in session state
                    st.session_state.training_jobs.append({
                        'job_id': job_id,
                        'type': 'incremental',
                        'status': 'running',
                        'progress': 0,
                        'started_at': datetime.now()
                    })
                    
                    st.success(f"✅ Incremental training scheduled: {job_id}")
                    st.info("⏳ Aktualizace běží. Scroll nahoru a rozbal '🔄 Training Progress' pro sledování.")
                    st.info("🔄 Auto-refresh: Stránka se automaticky obnoví každých 30s")
                    
                    # Force rerun to show progress section
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error scheduling training: {e}")
        
        with col3:
            st.markdown("**✓ Model Validation**")
            st.caption("Trvání: Minuty | Kdy: Po tréninku")
            if st.button("Model Validation", help="✓ Ověří přesnost modelu na testovacích datech. Použij po tréninku aby ses ujistil že model funguje správně."):
                try:
                    from lib.training_pipeline import TrainingScheduler
                    
                    scheduler = TrainingScheduler(
                        training_config,
                        ml_service.model_registry,
                        ml_service.performance_monitor
                    )
                    job_id = scheduler.schedule_training_job("validation")
                    
                    # Track job in session state
                    st.session_state.training_jobs.append({
                        'job_id': job_id,
                        'type': 'validation',
                        'status': 'running',
                        'progress': 0,
                        'started_at': datetime.now()
                    })
                    
                    st.success(f"✅ Validation job scheduled: {job_id}")
                    st.info("⏳ Validace běží. Scroll nahoru a rozbal '🔄 Training Progress' pro sledování.")
                    st.info("🔄 Auto-refresh: Stránka se automaticky obnoví každých 30s")
                    
                    # Force rerun to show progress section
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error scheduling validation: {e}")
    
    # User Feedback Interface
    with st.expander("👍 Provide Feedback (Zlepši model!)", expanded=True):
        st.info("""
        🎯 **Jak na to:**
        1. Přečti si zprávu níže
        2. Posuň slider podle důležitosti (🔇 Noise → 🔥 Urgent)
        3. Klikni "Submit"
        4. Opakuj pro 5-10 zpráv
        5. Pak jdi do "Training Controls" a klikni "Incremental Update"
        
        **Výsledek:** Model se naučí tvé preference! ✨
        """)
        
        st.markdown("### 📝 Ohodnoť tyto zprávy:")
        st.caption("Zobrazuji náhodný vzorek zpráv z databáze. Tvé hodnocení pomůže modelu se zlepšit.")
        
        # Sample recent messages for feedback
        try:
            import sqlite3
            from lib.database import DB_NAME
            
            conn = sqlite3.connect(DB_NAME)
            sample_query = """
                SELECT id, content, sent_at 
                FROM messages 
                WHERE length(content) > 20 
                ORDER BY RANDOM() 
                LIMIT 5
            """
            
            sample_df = pd.read_sql_query(sample_query, conn)
            conn.close()
            
            if not sample_df.empty:
                for idx, row in enumerate(sample_df.iterrows(), 1):
                    _, row = row
                    with st.container():
                        st.markdown(f"### 📨 Zpráva {idx}/5")
                        
                        # Message content in a nice box
                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <b>Obsah:</b> {row['content'][:250]}{'...' if len(row['content']) > 250 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"📅 Datum: {datetime.fromtimestamp(row['sent_at']).strftime('%d.%m.%Y %H:%M')}")
                        
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            rating = st.select_slider(
                                "Jak důležitá je tato zpráva?",
                                options=[1, 2, 3, 4, 5],
                                value=3,
                                format_func=lambda x: ["🔇 Noise (spam, lol, emojis)", "📝 Low (běžná konverzace)", "📋 Normal (normální diskuse)", "⚡ Important (důležité info)", "🔥 Urgent (group buy, event!)"][x-1],
                                key=f"rating_{row['id']}",
                                help="Posuň slider podle důležitosti zprávy. Tvé hodnocení pomůže modelu se učit."
                            )
                        
                        with col2:
                            if st.button("✅ Submit", key=f"submit_{row['id']}", type="primary", use_container_width=True):
                                ml_service.record_user_feedback(
                                    st.session_state.user_id,
                                    row['id'],
                                    "explicit_rating",
                                    explicit_rating=rating
                                )
                                st.success("✅ Feedback uložen! Díky!")
                                st.balloons()
                        
                        st.divider()
                
        except Exception as e:
            st.error(f"Error loading sample messages: {e}")

def main():
    if not google_api_key:
        st.error("Google API key not found. Please set GOOGLE_API_KEY in the .env file.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "recent_messages" not in st.session_state:
        st.session_state.recent_messages = []

    # Main app navigation
    page = st.sidebar.selectbox("Navigation", ["Chat Interface", "ML Administration"])
    
    if page == "Chat Interface":
        load_configuration()
        
        server_id = st.session_state.get("server_id", "")
        if not server_id:
            st.info("Please enter the Discord Server ID in the sidebar to begin.")
        else:
            display_chat()
    
    elif page == "ML Administration":
        display_ml_admin()

if __name__ == "__main__":
    main()
