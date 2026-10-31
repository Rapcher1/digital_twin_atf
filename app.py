import json
import os
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
        padding: 10px 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    }

    .message-row {
        margin: 0 0 12px 0;
    }

    .user-bubble {
        background: linear-gradient(135deg, #0f4c81, #1f7ad0);
        color: #f3fbff;
        border-radius: 18px 18px 6px 18px;
        padding: 12px 14px;
        max-width: 78%;
        margin-left: auto;
        box-shadow: 0 8px 16px rgba(20, 63, 113, 0.25);
    }

    .assistant-bubble {
        background: #102030;
        border: 1px solid #21384d;
        color: #ebf7ff;
        border-radius: 18px 18px 18px 6px;
        padding: 12px 14px;
        max-width: 82%;
        margin-right: auto;
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
        padding: 10px 12px;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    }

    .user-bubble {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: #ffffff;
        border-radius: 18px 18px 6px 18px;
        padding: 12px 14px;
        max-width: 78%;
        margin-left: auto;
    }

    .assistant-bubble {
        background: #f6f9fc;
        border: 1px solid #e3edf8;
        color: #0f172a;
        border-radius: 18px 18px 18px 6px;
        padding: 12px 14px;
        max-width: 82%;
        margin-right: auto;
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
    }


def get_default_system_prompt(profile_name: str, topic: str = "general", language: str = "en", memory_summary: str = ""):
    base = (
        f"You are {profile_name}, a thoughtful digital twin assistant. "
        f"You are helping in the topic: {topic}. "
        "Speak naturally, as a trusted human-like assistant would. "
        "Be warm, observant, practical, and concise. "
        "Remember what the user has said before, build on it, and refer to ongoing context when relevant. "
        "Do not sound robotic or generic. Offer useful next steps and ask thoughtful follow-ups when needed."
    )
    if memory_summary:
        base += f"\nImportant remembered context: {memory_summary}"
    if language == "sw":
        return base + " You can naturally mix English and Swahili when helpful."
    return base


def summarize_memory(chat_messages):
    if not chat_messages:
        return ""
    try:
        user_points = []
        for msg in chat_messages:
            if msg.get("role") == "user":
                text = (msg.get("content") or "").strip()
                if len(text) > 6:
                    user_points.append(text)
        if not user_points:
            return ""

        truncated = user_points[-6:]
        summary = "; ".join(truncated)
        return summary[:500]
    except Exception:
        return ""


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


def fallback_response(user_input: str, profile_name: str = "Aster", topic: str = "general", language: str = "en", memory_summary: str = ""):
    text = (user_input or "").strip()
    if not text:
        return "How can I help you today?"

    lower = text.lower()
    reference = f" Based on what you’ve shared before, " if memory_summary else " "

    if any(word in lower for word in ["plan", "schedule", "organize", "project", "task", "deadline", "roadmap", "goal"]):
        if language == "sw":
            return (
                "Naweza kukusaidia kupanga kazi kwa njia ya kutumika. "
                "Kwanza, tujue lengo lako, muda unaohitaji, na hatua ya kwanza unayotaka kuchukua."
            )
        return (
            f"I can help you shape this into a realistic plan.{reference}The best next step is usually to break it into a few concrete actions and a clear priority order."
        )

    if any(word in lower for word in ["research", "study", "learn", "explain", "summarize", "concept", "understand"]):
        if language == "sw":
            return (
                "Naweza kukupa maelezo ya wazi na kukusaidia kufahamu kwa hatua. "
                "Niambie mada yako na kiwango cha kina unachotaka."
            )
        return (
            f"I can explain it in a clearer, more natural way.{reference}The best approach is to simplify the idea first, then build from there with examples and a clear next step."
        )

    if any(word in lower for word in ["business", "strategy", "marketing", "growth", "startup", "sales"]):
        if language == "sw":
            return (
                "Naweza kukusaidia kwa mawazo ya biashara na mkakati. "
                "Tufanye hivi: sema lengo lako, changamoto, na watu wanaotakiwa kuathiriwa."
            )
        return (
            f"I can help you think through this strategically.{reference}The best response is usually to focus on the goal, the current bottleneck, and the simplest next move that creates momentum."
        )

    if language == "sw":
        return (
            f"Nimeelewa. Naweza kukusaidia katika mada ya {topic} kwa njia inayofaa na ya kibinadamu. "
            f"Tafadhali niambie zaidi kuhusu lengo lako au tatizo unalotaka kutatua."
        )

    return (
        f"I understand. I can help with {topic} in a practical, human way.{reference}Tell me a bit more about your goal, what you’ve already tried, and what success would look like."
    )


def generate_answer(user_input: str, language: str, profile_name: str, topic: str, chat_messages=None, memory_summary: str = ""):
    if not user_input or not user_input.strip():
        return "Please type a message so I can help.", {"blocked": False, "reasons": []}

    safe_state = safety.check_request(user_input, context=f"{topic} session")
    if safe_state["blocked"]:
        response = "I can’t help with that kind of request. If you want, I can help with a safe and useful alternative."
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

            recent_context = []
            for msg in chat_messages[-8:]:
                recent_context.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

            reply, _ = client.chat(
                cleaned_input,
                system_instruction=system_instruction,
                temperature=0.35,
                max_tokens=350,
                context_strategy="sliding_window",
            )
            validation = safety.validate_output(reply, context=f"{topic} session")
            if validation["valid"]:
                if memory_summary:
                    reply = reply.strip()
                    if not reply.lower().startswith("i remember") and not reply.lower().startswith("based on"):
                        reply = f"Based on what you’ve been working on, {reply[0].lower()}{reply[1:]}"
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
        fallback = "I can help with a clearer and safer version of this. Tell me a little more about what you want to achieve."
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
    st.markdown(f'<div class="topbar"><span class="session-pill">{active_chat["topic"]}</span><span class="session-pill">{active_chat["title"]}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    for msg in active_chat["messages"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="message-row"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-row"><div class="assistant-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
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
                memory_summary = active_chat.get("memory_summary") or summarize_memory(active_chat["messages"])
                with st.spinner("Thinking..."):
                    try:
                        response, _ = generate_answer(
                            user_input=draft.strip(),
                            language=st.session_state.language,
                            profile_name=st.session_state.profile_name,
                            topic=active_chat["topic"],
                            chat_messages=active_chat["messages"],
                            memory_summary=memory_summary,
                        )
                    except Exception:
                        response = "I hit a temporary issue. I’ll try again in a moment."

                active_chat["messages"].append({"role": "assistant", "content": response})
                active_chat["memory_summary"] = summarize_memory(active_chat["messages"])
                active_chat["updated_at"] = datetime.now().isoformat()
                st.session_state.draft_text = ""
                st.rerun()

    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state.draft_text = ""
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("A digital twin that remembers context and responds in a more human, conversational style.")
