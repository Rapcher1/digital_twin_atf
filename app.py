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

STYLES = """
<style>
    :root {
        --bg: #07111f;
        --panel: #0d1729;
        --panel-2: #111d2f;
        --panel-3: #17263d;
        --text: #edf5ff;
        --muted: #a7bdd7;
        --accent: #7dd3fc;
        --accent-2: #38bdf8;
        --line: #243247;
        --success: #22c55e;
        --warning: #fbbf24;
        --danger: #ef4444;
        --input: #0b1320;
        --chat-user: #13304c;
        --chat-assistant: #111827;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    div[data-testid="stSidebar"] {
        background: #091521 !important;
        border-right: 1px solid var(--line) !important;
    }

    .stApp .stChatMessage {
        background: transparent !important;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stChatInput textarea,
    .stSelectbox > div > div,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input, 
    .stDateInput > div > div > input {
        background: var(--input) !important;
        color: var(--text) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    .stChatInput textarea {
        min-height: 58px !important;
        color: var(--text) !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        color: var(--text) !important;
        background: #122033 !important;
        border: 1px solid var(--line) !important;
    }

    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: white !important;
    }

    .stDownloadButton > button {
        background: #122033 !important;
        color: var(--text) !important;
        border: 1px solid var(--line) !important;
    }

    .panel {
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        padding: 14px !important;
        color: var(--text) !important;
    }

    .soft-card {
        background: var(--panel-2) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        padding: 10px 12px !important;
        color: var(--text) !important;
    }

    .metric-box {
        background: var(--panel-2) !important;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    .label-text {
        color: var(--muted) !important;
    }

    .stChatMessage[data-testid="stChatMessage"] {
        background: transparent !important;
    }

    .stChatMessage [data-testid="chat-message-content"] {
        background: transparent !important;
        color: var(--text) !important;
    }

    .stChatMessage [data-testid="chat-avatar"] {
        background: linear-gradient(135deg, #38bdf8, #0ea5e9) !important;
        color: white !important;
    }

    .stMarkdown {
        color: var(--text) !important;
    }

    .stMarkdown p, .stMarkdown li, .stMarkdown div {
        color: var(--text) !important;
    }

    .stDataFrame, .stTable {
        background: var(--panel) !important;
    }
</style>
"""

st.set_page_config(page_title="Digital Twin Workspace", page_icon="🧠", layout="wide")

safety = DigitalTwinSafetySystem()
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# apply the dark styling globally
st.markdown(STYLES, unsafe_allow_html=True)


def default_chat_payload(title="New conversation", topic="general"):
    return {
        "id": f"chat-{uuid.uuid4().hex[:8]}",
        "title": title,
        "topic": topic,
        "messages": [
            {
                "role": "assistant",
                "content": "Hello! I’m your digital twin assistant. Ask me anything, refine an idea, plan your work, research a topic, or organize your next move."
            }
        ],
        "updated_at": datetime.now().isoformat(),
    }


def get_default_system_prompt(profile_name: str, topic: str = "general", language: str = "en"):
    base = (
        f"You are {profile_name}, a calm, capable, and personable digital twin assistant. "
        f"You are helping in the topic: {topic}. "
        "You are not a generic bot. You are thoughtful, conversational, warm, and practical. "
        "Speak naturally, with a human tone, and keep responses useful and easy to follow. "
        "Ask clarifying questions when needed, stay honest about uncertainty, and avoid robotic phrasing. "
        "Keep the answer structured when useful, but still sound like a real assistant talking to a person."
    )
    if language == "sw":
        return base + " Use English and Swahili naturally when helpful."
    return base


def get_twin_client():
    if not USE_REAL_TWIN or GroqClient is None:
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


def fallback_response(user_input: str, profile_name: str = "Aster", topic: str = "general", language: str = "en"):
    text = (user_input or "").strip()
    if not text:
        return "How can I help you today?"

    lower = text.lower()

    if any(word in lower for word in ["plan", "schedule", "organize", "project", "task", "roadmap", "deadline"]):
        if language == "sw":
            return (
                f"Naweza kukusaidia kupanga kazi kwa njia inayofaa. "
                f"Tafadhali niambie lengo lako, muda uliopo, na hatua za kwanza unazotaka kukitengeneza."
            )
        return (
            f"I can help structure this into a realistic plan. "
            f"Tell me your goal, the time you have, and the first steps you want to take."
        )

    if any(word in lower for word in ["research", "study", "learn", "explain", "summarize", "concept", "understand"]):
        if language == "sw":
            return (
                "Naweza kukupa ufafanuzi wa wazi, kufupisha habari, au kukusaidia kuelewa dhana kwa hatua. "
                "Tafadhali niambie mada yako na kiwango cha kina unachotaka."
            )
        return (
            "I can help explain the concept clearly, summarize the material, and guide you through it step by step. "
            "Tell me the topic and the depth you want."
        )

    if any(word in lower for word in ["business", "marketing", "strategy", "sales", "growth", "startup"]):
        if language == "sw":
            return (
                "Naweza kukusaidia kwa mawazo ya biashara na mkakati. "
                "Niambie lengo lako, changamoto yako, na jinsi unavyotaka kukua."
            )
        return (
            "I can help with business thinking, strategy, and practical next steps. "
            "Tell me what you are trying to achieve and what challenge you are facing."
        )

    if language == "sw":
        return (
            f"Nimeelewa. Naweza kukusaidia kwenye mada ya {topic} kwa njia ya kutumia akili na maelekezo yenye maana. "
            f"Tafadhali niambie zaidi kuhusu lengo lako au tatizo unalotaka kutatua."
        )

    return (
        f"I understand. I can help with the topic of {topic} in a practical and thoughtful way. "
        f"Tell me more about your goal, question, or the decision you are trying to make."
    )


def generate_answer(user_input: str, language: str, profile_name: str, topic: str, chat_history=None):
    if not user_input or not user_input.strip():
        return "Please type a message so I can help.", {"blocked": False, "reasons": []}

    safe_state = safety.check_request(user_input, context=f"{topic} session")
    if safe_state["blocked"]:
        response = "I can’t assist with that kind of request. If you want, I can help with a safe and productive alternative."
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
                system_instruction += "\n\nFresh external context (if relevant):\n" + external_context

            reply, usage = client.chat(
                cleaned_input,
                system_instruction=system_instruction,
                temperature=0.35,
                max_tokens=350,
                context_strategy="sliding_window",
            )

            validation = safety.validate_output(reply, context=f"{topic} session")
            if not validation["valid"]:
                reply = "I can help with a safer, more practical version of that. Tell me a bit more about what you need."

            return reply, safe_state
        except Exception:
            pass

    fallback = fallback_response(cleaned_input, profile_name=profile_name, topic=topic, language=language)
    validation = safety.validate_output(fallback, context=f"{topic} session")
    if not validation["valid"]:
        fallback = "I can help with a clearer and safer version of this. Give me a bit more context."

    return fallback, safe_state


# ---------------------------------
# Session state setup
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
        st.session_state.theme = "dark"

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""

    if "rename_value" not in st.session_state:
        st.session_state.rename_value = ""

init_session()

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

    with st.container():
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
    st.radio("Theme", ["dark", "light"], index=0 if st.session_state.theme == "dark" else 1, key="theme")

    with st.expander("Conversation settings"):
        if st.session_state.chats:
            current = st.session_state.chats[st.session_state.active_chat_id]
            st.text_input("Rename chat", value=current["title"], key="rename_value")
            if st.button("Apply rename", use_container_width=True):
                current["title"] = (st.session_state.rename_value or "New conversation").strip() or "New conversation"
                st.rerun()

        st.text_input("Current topic", key="topic_override")
        if st.button("Apply topic", use_container_width=True):
            current = st.session_state.chats[st.session_state.active_chat_id]
            current["topic"] = (st.session_state.topic_override or current["topic"]).strip() or current["topic"]
            st.rerun()

    st.download_button(
        "Export chats",
        data=json.dumps(st.session_state.chats, ensure_ascii=False, indent=2),
        file_name="digital_twin_chats.json",
        mime="application/json",
    )

# ---------------------------------
# Dark UI update based on theme
# ---------------------------------
if st.session_state.theme == "dark":
    st.markdown(STYLES, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #f4f7fb !important;
            color: #0f172a !important;
        }
        div[data-testid="stSidebar"] {
            background: #e5edf7 !important;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stChatInput textarea,
        .stSelectbox > div > div,
        .stButton > button {
            background: #f8fafc !important;
            color: #0f172a !important;
            border: 1px solid #d7e2ef !important;
            border-radius: 12px !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown li { color: #0f172a !important; }
        </style>
        """,
        unsafe_allow_html=True,
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

# Quick prompts
quick_prompts = {
    "en": [
        "Help me plan my week",
        "Explain a difficult idea simply",
        "Help me structure a project",
        "Give me a smart business idea",
        "Summarize this concept in plain English",
        "Help me study more effectively",
    ],
    "sw": [
        "Nisaidie kupanga wiki yangu",
        "Fafanua dhana kwa kifupi na wazi",
        "Nisaidie kupanga mradi",
        "Nipe wazo la biashara linalofaa",
        "Fupisha dhana hii kwa lugha ya kawaida",
        "Nisaidie kujifunza kwa ufanisi",
    ],
}

action_cols = st.columns(2)
for idx, prompt in enumerate(quick_prompts[st.session_state.language]):
    with action_cols[idx % 2]:
        if st.button(prompt, key=f"quick_{idx}", use_container_width=True):
            st.session_state.pending_prompt = prompt
            st.rerun()

if st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = ""
else:
    user_prompt = st.chat_input("Ask your digital twin anything...", key="chatinput")

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
                chat_history=active_chat["messages"],
            )
        except Exception:
            response = "I hit a temporary error. Please try again in a moment or ask a simpler question."
            safety_state = {"blocked": False, "reasons": ["error"]}

    active_chat["messages"].append({"role": "assistant", "content": response})
    active_chat["updated_at"] = datetime.now().isoformat()
    st.rerun()

# Footer info
with st.container():
    st.markdown("---")
    st.caption("General-purpose digital twin with Groq, session memory, and practical conversational responses.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chat", active_chat["title"])
    c2.metric("Topic", active_chat["topic"])
    c3.metric("Messages", len(active_chat["messages"]))

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
