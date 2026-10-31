import json
import os
import sys
import uuid
from datetime import datetime

import streamlit as st

# Ensure project root is importable when running from different working directories
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

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
    from twin import GroqClient, ContextWindowManager
    USE_REAL_TWIN = True
except Exception:
    USE_REAL_TWIN = False
    GroqClient = None
    ContextWindowManager = None

try:
    from data_fetcher import fetch_web_summary
except Exception:
    def fetch_web_summary(query: str, max_chars: int = 800):
        return ""

# ---------------------------------
# Safety
# ---------------------------------
safety = DigitalTwinSafetySystem()

# ---------------------------------
# Constants
# ---------------------------------
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------
# Helpers
# ---------------------------------
def get_default_system_prompt(profile_name: str, topic: str = "general", language: str = "en"):
    base = (
        f"You are {profile_name}, a professional digital twin assistant. "
        f"You are helping in the context: {topic}. "
        "You are general-purpose, useful, thoughtful, and reliable. "
        "Be helpful, precise, structured, and honest. "
        "Ask clarifying questions when needed. Do not pretend to have personal certainty when unsure. "
        "Keep answers practical, professional, and easy to act on."
    )

    if language == "sw":
        return (
            base + " Respond in English and Swahili when useful, and stay respectful, clear, and concise."
        )

    return base


def get_twin_client():
    if not USE_REAL_TWIN:
        return None

    try:
        cm = ContextWindowManager(max_tokens=8192, reserve_for_response=512)
        return GroqClient(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            context_manager=cm,
        )
    except Exception:
        return None


def build_external_context(user_input: str, topic: str):
    text = (user_input or "").strip()
    if not text:
        return ""

    live_keywords = [
        "news", "latest", "today", "current", "market", "price", "status", "update",
        "forecast", "weather", "research", "trend", "policy", "health", "business",
        "school", "study", "project", "event", "travel", "finance"
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


def default_chat_payload(title="New conversation", topic="general"):
    return {
        "id": f"chat-{uuid.uuid4().hex[:8]}",
        "title": title,
        "topic": topic,
        "messages": [
            {
                "role": "assistant",
                "content": "Hello! I’m your digital twin assistant. Ask me anything, organize work, brainstorm, plan, research, or help with your next decision."
            }
        ],
        "updated_at": datetime.now().isoformat(),
    }


def fallback_response(user_input: str, profile_name: str = "Aster", topic: str = "general", language: str = "en"):
    text = (user_input or "").strip()
    if not text:
        return "How can I help you today?"

    lower = text.lower()

    if any(word in lower for word in ["plan", "schedule", "organize", "project", "task", "deadline", "roadmap"]):
        if language == "sw":
            return (
                f"Naweza kukusaidia kupanga kazi, kutenga muda, na kufuatilia hatua. "
                f"Tafadhali niambie: lengo lako ni nini, muda unaopewa ni upi, na hatua zako za kwanza ni zipi?"
            )
        return (
            f"I can help you structure a plan, prioritize actions, and break the work into manageable steps. "
            f"Tell me your goal, your deadline, and the first actions you need to take."
        )

    if any(word in lower for word in ["research", "learn", "study", "explain", "summarize", "concept"]):
        if language == "sw":
            return (
                "Naweza kukusaidia kukutoa ufafanuzi, kuelezea dhana, au kufupisha habari. "
                "Tafadhali niambie mada yako na kiwango cha kina unachotaka."
            )
        return (
            "I can help explain a concept, summarize information, or guide your research. "
            "Tell me the topic and the depth or format you want."
        )

    if any(word in lower for word in ["business", "sales", "marketing", "strategy", "growth"]):
        if language == "sw":
            return (
                "Naweza kukusaidia kwa maamuzi ya biashara, mkakati, au mipango ya ukuaji. "
                "Niambie lengo lako, wafanyakazi au watumiaji, na changamoto ya msingi."
            )
        return (
            "I can help with strategy, experiments, marketing ideas, customer thinking, and execution planning. "
            "Tell me the business goal and the current challenge."
        )

    if language == "sw":
        return (
            f"Nimeelewa. Naweza kukusaidia kwenye mada ya {topic} kwa njia ya kisasa, ya kina, na inayofaa. "
            f"Tafadhali niambie zaidi kuhusu lengo lako au tatizo ambalo linaonekana."
        )

    return (
        f"I understand. I can help with the topic of {topic} in a practical and professional way. "
        f"Tell me more about your objective, challenge, or question so I can guide you clearly."
    )


def generate_answer(user_input: str, language: str, profile_name: str, topic: str):
    if not user_input or not user_input.strip():
        return "Please type a question or request so I can help.", {"blocked": False, "reasons": []}

    safe_state = safety.check_request(user_input, context=f"{topic} assistant")
    if safe_state["blocked"]:
        response = "I can’t assist with that kind of request. If you want, I can help with a safe, useful alternative."
        if safe_state.get("sensitive_topic"):
            response = safety.add_disclaimer(response)
        return response, safe_state

    cleaned_input = safety.redact_input(user_input)
    external_context = build_external_context(cleaned_input, topic)

    client = get_twin_client()
    api_key = os.environ.get("GROQ_API_KEY", "")

    if client is not None and api_key:
        try:
            system_instruction = get_default_system_prompt(profile_name, topic, language)
            if external_context:
                system_instruction += (
                    "\n\nFresh external context (use only when relevant):\n"
                    f"{external_context}"
                )

            reply, usage = client.chat(
                cleaned_input,
                system_instruction=system_instruction,
                temperature=0.35,
                max_tokens=350,
                context_strategy="sliding_window",
            )

            validation = safety.validate_output(reply, context=f"{topic} assistant")
            if not validation["valid"]:
                reply = "I can help with a safer, more practical response. Please share more detail about your goal and constraints."

            return reply, safe_state
        except Exception:
            pass

    fallback = fallback_response(cleaned_input, profile_name=profile_name, topic=topic, language=language)
    validation = safety.validate_output(fallback, context=f"{topic} assistant")
    if not validation["valid"]:
        fallback = "I can help with a safer, clearer version of this. Please provide a little more context."

    return fallback, safe_state


# ---------------------------------
# Theme
# ---------------------------------
def apply_theme(theme_name: str):
    if theme_name == "dark":
        st.markdown(
            """
            <style>
            .stApp {
                background: #0b1020;
                color: #e5eefc;
            }
            div[data-testid="stSidebar"] {
                background: #111827;
            }
            .panel {
                background: #111827;
                border: 1px solid #243244;
                border-radius: 16px;
                padding: 14px;
            }
            .soft-card {
                background: #0f172a;
                border: 1px solid #1f2a3d;
                border-radius: 12px;
                padding: 12px;
            }
            .metric-box {
                background: #101827;
                border: 1px solid #243244;
                border-radius: 12px;
                padding: 12px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
                color: #0f172a;
            }
            div[data-testid="stSidebar"] {
                background: #f1f5f9;
            }
            .panel {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 14px;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
            }
            .soft-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 12px;
            }
            .metric-box {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 12px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------
# Session setup
# ---------------------------------
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
        st.session_state.theme = "light"

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""

init_session()
apply_theme(st.session_state.theme)

# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.title("🧠 Digital Twin")
    st.caption("General-purpose AI workspace")

    if st.button("+ New chat", use_container_width=True):
        new_id = f"chat-{uuid.uuid4().hex[:8]}"
        st.session_state.chats[new_id] = default_chat_payload(title="New conversation", topic="general")
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.markdown("### Conversations")
    for chat_id, chat in st.session_state.chats.items():
        active = chat_id == st.session_state.active_chat_id
        label = f"{chat['title']} • {chat['topic']}"
        if st.button(label, key=f"chat_{chat_id}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    st.markdown("---")
    st.markdown("### Settings")

    st.selectbox("Language", ["en", "sw"], index=0 if st.session_state.language == "en" else 1, key="language")
    st.text_input("Profile name", key="profile_name")
    st.radio("Theme", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1, key="theme")

    with st.expander("Conversation settings"):
        st.text_input("Current topic", key="topic_override")
        if st.button("Apply topic", use_container_width=True):
            current = st.session_state.chats[st.session_state.active_chat_id]
            current["topic"] = (st.session_state.topic_override or current["topic"]).strip() or current["topic"]
            current["title"] = current["title"] or "New conversation"
            st.rerun()

    st.download_button(
        "Export chats",
        data=json.dumps(st.session_state.chats, ensure_ascii=False, indent=2),
        file_name="digital_twin_chats.json",
        mime="application/json",
    )

# ---------------------------------
# Active chat
# ---------------------------------
active_chat = st.session_state.chats[st.session_state.active_chat_id]

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title(active_chat["title"])
with header_col2:
    st.markdown(f"<div class='panel'>Topic: {active_chat['topic']}</div>", unsafe_allow_html=True)

for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------
# Quick actions
# ---------------------------------
quick_prompts = {
    "en": [
        "Help me plan my week",
        "Explain a concept clearly",
        "Help me structure a project",
        "Summarize this idea in simple terms",
        "Give me a smart business strategy",
        "Help me study more effectively",
    ],
    "sw": [
        "Nisaidie kupanga wiki yangu",
        "Fafanua dhana kwa undani",
        "Nisaidie kupanga mradi",
        "Fupisha wazo hili kwa kifupi",
        "Nipe mkakati mzuri wa biashara",
        "Nisaidie kujifunza kwa ufanisi",
    ],
}

actions = quick_prompts[st.session_state.language]
button_cols = st.columns(2)
for idx, prompt in enumerate(actions):
    with button_cols[idx % 2]:
        if st.button(prompt, key=f"quick_{idx}", use_container_width=True):
            st.session_state.pending_prompt = prompt
            st.rerun()

if st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = ""
else:
    user_prompt = st.chat_input("Ask your digital twin anything...")

if user_prompt:
    active_chat["messages"].append({"role": "user", "content": user_prompt})
    active_chat["updated_at"] = datetime.now().isoformat()

    with st.spinner("Thinking..."):
        try:
            response, safety_state = generate_answer(
                user_input=user_prompt,
                language=st.session_state.language,
                profile_name=st.session_state.profile_name,
                topic=active_chat["topic"],
            )
        except Exception:
            response = "I hit a temporary error. Please try again in a moment or ask a simpler question."
            safety_state = {"blocked": False, "reasons": ["error"]}

    active_chat["messages"].append({"role": "assistant", "content": response})
    active_chat["updated_at"] = datetime.now().isoformat()
    st.rerun()

# ---------------------------------
# Footer / metrics
# ---------------------------------
with st.container():
    st.markdown("---")
    st.caption("General-purpose twin assistant with Groq, session memory, safe fallback, and live external context support.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Active chat", active_chat["title"])
    m2.metric("Topic", active_chat["topic"])
    m3.metric("Messages", len(active_chat["messages"]))

    if st.button("🎙️ Voice input"):
        st.components.v1.html(
            """
            <script>
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                alert('Voice input is not supported in this browser.');
                return;
            }
            const recognition = new SpeechRecognition();
            recognition.lang = 'en-US';
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                const textInput = document.querySelector('textarea');
                if (textInput) {
                    textInput.value = transcript;
                    textInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
            recognition.start();
            </script>
            """,
            height=120,
        )
