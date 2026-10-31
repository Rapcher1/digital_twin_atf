import json
import os
import re
import sys
import uuid
from datetime import datetime

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from twin import GroqClient, ContextWindowManager, DigitalTwinBrain
    USE_REAL_TWIN = True
except Exception:
    USE_REAL_TWIN = False
    GroqClient = None
    ContextWindowManager = None
    DigitalTwinBrain = None

try:
    from digital_twin_safety import DigitalTwinSafetySystem
except Exception:
    class DigitalTwinSafetySystem:
        def __init__(self, *args, **kwargs):
            pass

        def check_request(self, text, context=""):
            return {
                "blocked": False,
                "reasons": [],
                "pii": {},
                "content_categories": [],
                "sensitive_topic": False,
                "adversarial_prompt": False,
                "prompt_injection_match": [],
            }

        def redact_input(self, text):
            return text or ""

        def add_disclaimer(self, text):
            return text

        def validate_output(self, text, context=""):
            return {"valid": True, "issues": []}

try:
    from data_fetcher import fetch_web_summary
except Exception:
    def fetch_web_summary(query: str, max_chars: int = 800):
        return ""

safety = DigitalTwinSafetySystem()
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

DARK_CSS = """
<style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #061018 !important;
        color: #eaf5ff !important;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    div[data-testid="stSidebar"] {
        background: #091824 !important;
        border-right: 1px solid #1b2c3f !important;
    }

    .app-shell {
        max-width: 950px;
        margin: 0 auto;
        padding: 14px 14px 24px;
    }

    .topbar {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #233247;
        border-radius: 18px;
        padding: 12px 16px;
        margin-bottom: 14px;
    }

    .chat-card {
        background: #0d1724;
        border: 1px solid #1f2f41;
        border-radius: 18px;
        padding: 12px 12px 8px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    }

    .message-row {
        margin: 0 0 12px 0;
        display: flex;
    }

    .message-row.user {
        justify-content: flex-end;
    }

    .message-row.assistant {
        justify-content: flex-start;
    }

    .user-bubble {
        background: linear-gradient(135deg, #0f4c81, #1f7ad0);
        color: #f3fbff;
        border-radius: 18px 18px 6px 18px;
        padding: 12px 14px;
        max-width: 78%;
        box-shadow: 0 8px 16px rgba(20, 63, 113, 0.25);
    }

    .assistant-bubble {
        background: #102030;
        border: 1px solid #21384d;
        color: #ebf7ff;
        border-radius: 18px 18px 18px 6px;
        padding: 12px 14px;
        max-width: 82%;
        box-shadow: 0 8px 18px rgba(0,0,0,0.08);
    }

    .meta {
        color: #a7bfd3;
        font-size: 0.74rem;
    }

    .session-pill {
        background: #101e2d;
        border: 1px solid #243548;
        border-radius: 999px;
        padding: 8px 14px;
        color: #dfeefb;
        display: inline-block;
        font-size: 0.8rem;
        margin-right: 8px;
    }

    .profile-box {
        background: #0e1c2b;
        border: 1px solid #22384e;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 12px;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stButton > button,
    .stDownloadButton > button {
        background: #0d1726 !important;
        color: #edf6ff !important;
        border: 1px solid #243247 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    .stChatInput textarea {
        background: #0d1726 !important;
        color: #edf6ff !important;
        border: 1px solid #243247 !important;
        min-height: 68px !important;
        border-radius: 14px !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #7dd3fc !important;
        color: #ffffff !important;
    }

    .stMarkdown p, .stMarkdown li, .stMarkdown div {
        color: #edf6ff !important;
    }
</style>
"""

LIGHT_CSS = """
<style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #f2f6fb !important;
        color: #0f172a !important;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    div[data-testid="stSidebar"] {
        background: #e8eef8 !important;
        border-right: 1px solid #dfeaf7 !important;
    }

    .app-shell {
        max-width: 900px;
        margin: 0 auto;
        padding: 14px 10px 28px;
    }

    .topbar {
        background: #ffffff;
        border: 1px solid #dfeaf7;
        border-radius: 18px;
        padding: 12px 16px;
        margin-bottom: 14px;
    }

    .chat-card {
        background: #ffffff;
        border: 1px solid #e2ebf5;
        border-radius: 18px;
        padding: 12px 12px 8px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    }

    .message-row {
        margin: 0 0 12px 0;
        display: flex;
    }

    .message-row.user {
        justify-content: flex-end;
    }

    .message-row.assistant {
        justify-content: flex-start;
    }

    .user-bubble {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: #ffffff;
        border-radius: 18px 18px 6px 18px;
        padding: 12px 14px;
        max-width: 78%;
    }

    .assistant-bubble {
        background: #f6f9fc;
        border: 1px solid #e3edf8;
        color: #0f172a;
        border-radius: 18px 18px 18px 6px;
        padding: 12px 14px;
        max-width: 82%;
    }

    .meta {
        color: #5c728b;
        font-size: 0.74rem;
    }

    .session-pill {
        background: #edf4ff;
        border: 1px solid #d9e8ff;
        border-radius: 999px;
        padding: 8px 14px;
        color: #1e3a8a;
        display: inline-block;
        font-size: 0.8rem;
        margin-right: 8px;
    }

    .profile-box {
        background: #f9fbff;
        border: 1px solid #e1ebf8;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 12px;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stButton > button,
    .stDownloadButton > button {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #d5dfeb !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    .stChatInput textarea {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #d5dfeb !important;
        min-height: 68px !important;
        border-radius: 14px !important;
    }

    .stMarkdown p, .stMarkdown li, .stMarkdown div {
        color: #0f172a !important;
    }
</style>
"""

DEFAULT_PROFILE = {
    "name": "Aster",
    "persona": "calm, thoughtful, proactive, and grounded",
    "style": "warm, practical, natural, and collaborative",
    "goals": [],
    "preferences": [],
    "recurring_topics": [],
    "facts": [],
}


def default_chat_payload(title="New conversation", topic="general"):
    return {
        "id": f"chat-{uuid.uuid4().hex[:8]}",
        "title": title,
        "topic": topic,
        "messages": [{
            "role": "assistant",
            "content": "Hi — I’m your digital twin. I remember what matters to you, and I’ll keep it in context as we talk."
        }],
        "updated_at": datetime.now().isoformat(),
        "memory_summary": "The user is building a digital twin assistant and wants a conversational, human-style experience.",
        "profile": json.loads(json.dumps(DEFAULT_PROFILE)),
    }


def ensure_chat_profile(chat):
    if "profile" not in chat or not isinstance(chat["profile"], dict):
        chat["profile"] = json.loads(json.dumps(DEFAULT_PROFILE))
    for key, default_value in DEFAULT_PROFILE.items():
        chat["profile"].setdefault(key, default_value)
    if not isinstance(chat["profile"].get("goals"), list):
        chat["profile"]["goals"] = []
    if not isinstance(chat["profile"].get("preferences"), list):
        chat["profile"]["preferences"] = []
    if not isinstance(chat["profile"].get("recurring_topics"), list):
        chat["profile"]["recurring_topics"] = []
    if not isinstance(chat["profile"].get("facts"), list):
        chat["profile"]["facts"] = []
    return chat["profile"]


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_profile_from_messages(messages):
    profile = json.loads(json.dumps(DEFAULT_PROFILE))
    goals = []
    preferences = []
    topics = []
    facts = []

    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = normalize_text(msg.get("content") or "")
        if not text:
            continue

        lower = text.lower()
        if any(k in lower for k in ["goal", "want", "trying to", "need", "focus", "planning", "improve", "build", "launch", "grow", "help me"]):
            goals.append(text)
        if any(k in lower for k in ["prefer", "like", "enjoy", "love", "value", "works better", "best for"]):
            preferences.append(text)
        if len(text) > 20:
            facts.append(text)

        # simple topic extraction
        for topic in [
            "project", "business", "study", "planning", "research", "career", "product",
            "strategy", "marketing", "growth", "learning", "work", "personal", "focus"
        ]:
            if topic in lower:
                topics.append(topic)

    profile["goals"] = goals[-4:]
    profile["preferences"] = preferences[-4:]
    profile["recurring_topics"] = list(dict.fromkeys(topics[-6:]))
    profile["facts"] = facts[-5:]

    if not profile["goals"]:
        profile["goals"] = ["Build a more useful and human digital twin experience"]
    if not profile["preferences"]:
        profile["preferences"] = ["Clear, calm, practical communication"]

    return profile


def build_memory_summary(chat):
    ensure_chat_profile(chat)
    profile = chat["profile"]
    messages = chat["messages"]
    user_points = []
    for msg in messages:
        if msg.get("role") == "user":
            text = normalize_text(msg.get("content") or "")
            if len(text) > 10:
                user_points.append(text)

    summary_parts = []
    if profile.get("goals"):
        summary_parts.append("Goals: " + "; ".join(profile["goals"][-3:]))
    if profile.get("preferences"):
        summary_parts.append("Preferences: " + "; ".join(profile["preferences"][-3:]))
    if profile.get("recurring_topics"):
        summary_parts.append("Recurring topics: " + ", ".join(profile["recurring_topics"][-5:]))
    if user_points:
        recent = user_points[-5:]
        summary_parts.append("Recent context: " + "; ".join(recent))

    summary = " | ".join(summary_parts)
    chat["memory_summary"] = summary[:800]
    return chat["memory_summary"]


def get_default_system_prompt(profile_name: str, topic: str = "general", language: str = "en", memory_summary: str = ""):
    base = (
        f"You are {profile_name}, a digital twin with a calm, human personality. "
        f"You are helping in the topic: {topic}. "
        "Speak as a thoughtful personal assistant would: warm, proactive, natural, and grounded. "
        "Remember what the user has said, build on it, and connect your answers to their context. "
        "Do not sound robotic or generic. Be practical, useful, and conversational. "
        "Ask for missing detail when it is important."
    )
    if memory_summary:
        base += f"\nRemembered context: {memory_summary}"
    if language == "sw":
        return base + " Mix English and Swahili naturally when helpful."
    return base


def get_twin_brain():
    if DigitalTwinBrain is not None:
        try:
            return DigitalTwinBrain(
                api_key=os.environ.get("GROQ_API_KEY", ""),
                model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            )
        except Exception:
            pass

    if USE_REAL_TWIN and GroqClient is not None and ContextWindowManager is not None:
        try:
            return DigitalTwinBrain(
                api_key=os.environ.get("GROQ_API_KEY", ""),
                model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                context_manager=ContextWindowManager(max_tokens=8192, reserve_for_response=512),
            )
        except Exception:
            pass

    return None


def build_external_context(user_input: str, topic: str):
    text = (user_input or "").strip()
    if not text:
        return ""

    live_keywords = [
        "news", "latest", "today", "current", "market", "price", "status", "update",
        "forecast", "weather", "research", "trend", "policy", "health", "business",
        "school", "study", "project", "event", "travel", "finance", "career", "startup"
    ]

    if not any(k.lower() in text.lower() for k in live_keywords):
        return ""

    try:
        result = fetch_web_summary(f"{topic} {text}")
        if result:
            return result[:900]
    except Exception:
        return ""
    return ""


def needs_clarifying_question(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return False
    if len(lower) < 16:
        return False
    broad_markers = [
        "help me", "can you help", "what should i do", "what do you think",
        "advice", "ideas", "i need help", "what should i focus on"
    ]
    context_markers = ["because", "goal", "deadline", "trying", "already", "currently", "for my", "my situation", "what i want"]
    if any(marker in lower for marker in broad_markers) and not any(marker in lower for marker in context_markers):
        return True
    return False


def fallback_response(user_input: str, profile_name: str = "Aster", topic: str = "general", language: str = "en", memory_summary: str = ""):
    text = normalize_text(user_input)
    if not text:
        return "How can I help you today?"

    lower = text.lower()

    if needs_clarifying_question(text):
        if language == "sw":
            return (
                "Naweza kukusaidia, lakini ningependa kujua zaidi. "
                "Lengo lako ni nini, ni kwa hali gani, na umejaribu nini tayari?"
            )
        return (
            "I can help with that. Before I suggest anything, what exactly are you trying to achieve, and what context or constraints matter most right now?"
        )

    if any(word in lower for word in ["plan", "schedule", "organize", "project", "task", "deadline", "roadmap", "goal"]):
        if language == "sw":
            return (
                "Naweza kukusaidia kupanga kazi kwa njia ya maana. "
                "Kwanza, niambie lengo lako, muda unaohitaji, na hatua ya kwanza unayotaka kuchukua."
            )
        return (
            "I can help you turn this into a realistic plan. Let’s start with your goal, your timeline, and the first action that creates momentum."
        )

    if any(word in lower for word in ["research", "study", "learn", "explain", "summarize", "concept", "understand"]):
        if language == "sw":
            return (
                "Naweza kukupa ufafanuzi kwa njia ya wazi na ya kibinadamu. "
                "Niambie mada yako na kiwango cha kina unachotaka."
            )
        return (
            "I can explain it in a clear and practical way. Tell me the topic, the level you need, and what part feels confusing."
        )

    if any(word in lower for word in ["business", "strategy", "marketing", "growth", "startup", "sales"]):
        if language == "sw":
            return (
                "Naweza kukusaidia kwa mawazo ya biashara na mkakati. "
                "Tujue lengo lako, changamoto, na hatua inayofaa ya kwanza."
            )
        return (
            "I can help you think through this strategically. Start with the goal, the current bottleneck, and the simplest next move that creates momentum."
        )

    if memory_summary:
        if language == "sw":
            return (
                f"Nimekumbuka kwamba {memory_summary[:200]}. "
                f"Kulingana na hilo, ningependa kukusaidia kwa njia inayofaa zaidi na inayohusiana na lengo lako."
            )
        return (
            f"I remember the key context you’ve been working with: {memory_summary[:200]}. Based on that, I’d suggest we focus on the most relevant next step rather than jumping into a generic answer."
        )

    if language == "sw":
        return (
            f"Nimeelewa. Naweza kukusaidia katika mada ya {topic} kwa njia ya maana na yenye kutumika. "
            f"Tafadhali niambie zaidi kuhusu lengo lako au tatizo unalotaka kutatua."
        )

    return (
        f"I understand. I can help with {topic} in a practical, human way. Tell me more about your goal, the challenge, or the decision you’re trying to make."
    )


def generate_answer(user_input: str, language: str, profile_name: str, topic: str, chat_messages=None, memory_summary: str = ""):
    if not user_input or not user_input.strip():
        return "Please type a message so I can help.", {"blocked": False, "reasons": []}

    safe_state = safety.check_request(user_input, context=f"{topic} session")
    if safe_state["blocked"]:
        response = "I can’t help with that kind of request. If you want, I can help with a safe and productive alternative."
        if safe_state.get("sensitive_topic"):
            response = safety.add_disclaimer(response)
        return response, safe_state

    cleaned_input = safety.redact_input(user_input)
    external_context = build_external_context(cleaned_input, topic)
    chat_messages = chat_messages or []

    brain = get_twin_brain()
    if brain is not None:
        try:
            result = brain.route(
                user_message=cleaned_input,
                profile_name=profile_name,
                topic=topic,
                language=language,
                external_context=external_context,
                chat_history=chat_messages,
                memory_summary=memory_summary,
            )
            if result:
                reply = result.get("reply") or result.get("response") or ""
                if reply:
                    validation = safety.validate_output(reply, context=f"{topic} session")
                    if validation["valid"]:
                        return reply, safe_state
        except Exception:
            pass

    client = None
    if USE_REAL_TWIN and GroqClient is not None and ContextWindowManager is not None:
        try:
            client = GroqClient(
                api_key=os.environ.get("GROQ_API_KEY", ""),
                model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                context_manager=ContextWindowManager(max_tokens=8192, reserve_for_response=512),
            )
        except Exception:
            client = None

    if client is not None and os.environ.get("GROQ_API_KEY", ""):
        try:
            system_instruction = get_default_system_prompt(profile_name, topic, language, memory_summary)
            if external_context:
                system_instruction += "\n\nFresh external context (if relevant):\n" + external_context

            reply, _ = client.chat(
                cleaned_input,
                system_instruction=system_instruction,
                temperature=0.35,
                max_tokens=350,
                context_strategy="sliding_window",
            )
            validation = safety.validate_output(reply, context=f"{topic} session")
            if validation["valid"]:
                return reply, safe_state
        except Exception:
            pass

    fallback = fallback_response(
        cleaned_input,
        profile_name=profile_name,
        topic=topic,
        language=language,
        memory_summary=memory_summary,
    )
    validation = safety.validate_output(fallback, context=f"{topic} session")
    if not validation["valid"]:
        fallback = "I can help with a clearer, safer version of this. Tell me a little more about what you want to achieve."
    return fallback, safe_state


def init_session():
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "chat-1": default_chat_payload(title="General assistant", topic="general")
        }

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = "chat-1"

    if "language" not in st.session_state:
        st.session_state.language = "en"

    if "profile_name" not in st.session_state:
        st.session_state.profile_name = "Aster"

    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    if "draft_text" not in st.session_state:
        st.session_state.draft_text = ""

    if "rename_value" not in st.session_state:
        st.session_state.rename_value = ""

init_session()

if st.session_state.theme == "dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 Twin")
    st.caption("Personal digital twin")

    if st.button("+ New chat", use_container_width=True):
        new_id = f"chat-{uuid.uuid4().hex[:8]}"
        st.session_state.chats[new_id] = default_chat_payload(title="New conversation", topic="general")
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.markdown("### Chats")
    for chat_id, chat in st.session_state.chats.items():
        active = chat_id == st.session_state.active_chat_id
        label = f"{chat['title']} • {chat['topic']}"
        if st.button(label, key=f"chat_{chat_id}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    st.markdown("---")
    st.selectbox("Language", ["en", "sw"], index=0 if st.session_state.language == "en" else 1, key="language")
    st.text_input("Profile name", key="profile_name")
    st.radio("Theme", ["dark", "light"], index=0 if st.session_state.theme == "dark" else 1, key="theme")

    with st.expander("Chat controls"):
        current = st.session_state.chats[st.session_state.active_chat_id]
        st.text_input("Rename this chat", value=current["title"], key="rename_value")
        if st.button("Apply rename", use_container_width=True):
            current["title"] = (st.session_state.rename_value or "New conversation").strip() or "New conversation"
            st.rerun()

        st.text_input("Topic", key="topic_override")
        if st.button("Apply topic", use_container_width=True):
            current["topic"] = (st.session_state.topic_override or current["topic"]).strip() or current["topic"]
            st.rerun()

    st.download_button(
        "Export chats",
        data=json.dumps(st.session_state.chats, ensure_ascii=False, indent=2),
        file_name="digital_twin_chats.json",
        mime="application/json",
    )

with st.container():
    st.markdown('<div class="app-shell">', unsafe_allow_html=True)

    active_chat = st.session_state.chats[st.session_state.active_chat_id]
    active_chat["profile"] = ensure_chat_profile(active_chat)

    # Keep a more meaningful profile based on chat content
    if active_chat["messages"]:
        active_chat["profile"] = extract_profile_from_messages(active_chat["messages"])

    active_chat["memory_summary"] = build_memory_summary(active_chat)

    st.markdown(f'<div class="topbar"><span class="session-pill">{active_chat["topic"]}</span><span class="session-pill">{active_chat["title"]}</span></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="profile-box"><strong>{st.session_state.profile_name}</strong> · {active_chat["profile"]["persona"]}<br><small>{active_chat["memory_summary"][:220]}</small></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    for msg in active_chat["messages"]:
        role_class = "user" if msg["role"] == "user" else "assistant"
        st.markdown(f'<div class="message-row {role_class}"><div class="{role_class}-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    quick_prompts = {
        "en": [
            "Help me plan my week",
            "Explain a difficult idea simply",
            "Help me structure a project",
            "Give me a smart strategy",
            "Summarize this concept clearly",
            "What should I focus on next?",
        ],
        "sw": [
            "Nisaidie kupanga wiki yangu",
            "Fafanua dhana kwa kifupi",
            "Nisaidie kupanga mradi",
            "Nipe mkakati mzuri",
            "Fupisha dhana kwa wazi",
            "Nini kinapaswa kuzingatiwa kwa hatua inayofuata?",
        ],
    }

    cols = st.columns(2)
    for idx, prompt in enumerate(quick_prompts[st.session_state.language]):
        with cols[idx % 2]:
            if st.button(prompt, key=f"quick_{idx}", use_container_width=True):
                st.session_state.draft_text = prompt
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    draft = st.text_area(
        "Write a message",
        value=st.session_state.draft_text,
        height=120,
        key="draft_text",
        label_visibility="visible",
        placeholder="Tell me what you’re thinking about...",
    )

    send_col, clear_col = st.columns([1, 1])
    with send_col:
        if st.button("Send", use_container_width=True, type="primary"):
            if draft and draft.strip():
                active_chat["messages"].append({"role": "user", "content": draft.strip()})
                active_chat["updated_at"] = datetime.now().isoformat()
                active_chat["profile"] = extract_profile_from_messages(active_chat["messages"])
                active_chat["memory_summary"] = build_memory_summary(active_chat)
                with st.status("Aster is thinking…", expanded=True) as status:
                    st.write("I’m connecting this to what you’ve already shared and shaping a more personal response.")
                    try:
                        response, _ = generate_answer(
                            user_input=draft.strip(),
                            language=st.session_state.language,
                            profile_name=st.session_state.profile_name,
                            topic=active_chat["topic"],
                            chat_messages=active_chat["messages"],
                            memory_summary=active_chat["memory_summary"],
                        )
                    except Exception:
                        response = "I hit a temporary issue. I’ll try again in a moment."
                    status.update(label="Aster has an answer.", state="complete")

                active_chat["messages"].append({"role": "assistant", "content": response})
                active_chat["memory_summary"] = build_memory_summary(active_chat)
                active_chat["updated_at"] = datetime.now().isoformat()
                st.session_state.draft_text = ""
                st.rerun()

    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state.draft_text = ""
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("A digital twin that remembers context, understands your direction, and responds in a more human, conversational way.")
