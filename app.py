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
    from digital_twin_safety import DigitalTwinSafetySystem
except Exception:
    class DigitalTwinSafetySystem:
        def __init__(self, *args, **kwargs):
            pass

        def check_request(self, text, context=""):
            return {"blocked": False, "reasons": [], "pii": {}, "content_categories": [], "sensitive_topic": False, "adversarial_prompt": False, "prompt_injection_match": []}

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

from data_fetcher import fetch_web_summary


st.set_page_config(page_title="Digital Twin Workspace", page_icon="🧠", layout="wide")


# ---------------------------------
# Safety system
# ---------------------------------
safety = DigitalTwinSafetySystem()


# ---------------------------------
# Core utility functions
# ---------------------------------
def get_default_system_prompt(language: str, profile_name: str, topic: str = "general"):
    if language == "sw":
        return (
            f"You are {profile_name}, a helpful African digital twin assistant. "
            f"You are currently helping in the topic: {topic}. "
            "Respond in Swahili and English when useful. Give safe, practical, and empathetic advice. "
            "Ask clarifying questions when needed. Never diagnose. Use a calm and supportive tone."
        )
    return (
        f"You are {profile_name}, a helpful African digital twin assistant. "
        f"You are currently helping in the topic: {topic}. "
        "Respond in English, and if helpful, include brief Swahili phrases. "
        "Give safe, practical, and empathetic guidance. Ask clarifying questions when needed. "
        "Never diagnose. Use a calm and supportive tone."
    )


# ---------------------------------
# Real model integration
# ---------------------------------
def get_twin_client():
    if not USE_REAL_TWIN:
        return None

    try:
        cm = ContextWindowManager(max_tokens=8192, reserve_for_response=512)
        return GroqClient(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            model="llama3.3",
            context_manager=cm,
        )
    except Exception:
        return None


# ---------------------------------
# Offline fallback logic
# ---------------------------------
def local_twin_response(user_input: str, language: str = "en", profile_name: str = "Aster", topic: str = "general"):
    text = (user_input or "").strip()
    if not text:
        return "How can I help you today?"

    lower = text.lower()

    if any(word in lower for word in ["headache", "fever", "cough", "pain", "sick", "dizzy"]):
        if language == "sw":
            return (
                "Ninapenda kukusaidia. Kwanza, ni muhimu kujua umri wako, muda wa dalili, "
                "na kama una ishara za dharura kama kupumua kwa shida, maumivu makali ya kifua, au kupoteza fahamu. "
                "Kama unahisi hali ni ya dharura, piga huduma ya dharura au tembelea kliniki mara moja."
            )
        return (
            "I can help with that. First, tell me your age, symptoms, duration, and whether you have red flags such as trouble breathing, severe chest pain, fainting, or a high fever. "
            "If this is an emergency, contact emergency services immediately."
        )

    if any(word in lower for word in ["pregnant", "pregnancy", "baby", "prenatal"]):
        if language == "sw":
            return (
                "Kama una maswali kuhusu ujauzito, ni vizuri kujadili na daktari au mtoa huduma wa afya. "
                "Tafadhali sema kama una maumivu makali ya tumbo, kutokwa na damu, kufahamu kupoteza fahamu, au kupumua kwa shida."
            )
        return (
            "For pregnancy questions, it is best to consult a clinician or qualified healthcare provider. "
            "If you have severe pain, bleeding, fainting, or breathing issues, seek urgent care."
        )

    if any(word in lower for word in ["mental", "stress", "anxiety", "sad", "depressed", "panic"]):
        if language == "sw":
            return (
                "Unahitaji msaada wa karibu. Unaweza kuzungumza na mtu wa karibu, rafiki, au mtoa huduma wa afya. "
                "Kama una mawazo ya kujidhuru au huna usalama, piga huduma ya dharura au wasiliana na hotline ya dharura."
            )
        return (
            "It is important to seek support from someone you trust and a qualified health professional. "
            "If you feel at risk of harming yourself or someone else, contact emergency services or a crisis hotline immediately."
        )

    if any(word in lower for word in ["weather", "rain", "forecast", "temperature", "climate", "market", "price", "harvest", "farming", "soil"]):
        if language == "sw":
            return (
                "Kwa maswali ya kilimo au hali ya hewa, angalia taarifa za karibu na eneo lako. "
                "Vifaa vya kilimo vinaweza kuonyesha mvua, udongo na wakati wa kupanda au kuvuna. "
                "Kama unahitaji, niweza kukusaidia kuangalia hatua za msingi za kilimo kwa mazingira ya eneo lako."
            )
        return (
            "For agricultural or weather questions, check local forecasts and field conditions in your area first. "
            "I can help you reason through planting, watering, pest management, and local seasonal patterns."
        )

    if language == "sw":
        return (
            f"Nimeelewa. Nitakusaidia kwa njia inayolenga {topic} na afya yako kwa ujumla. "
            f"Tafadhali niambie zaidi kuhusu hali yako, dalili zako, au jambo la kutaka kujadili."
        )
    return (
        f"I understand. I can help with the topic of {topic} and your broader wellbeing. "
        f"Please share a little more detail about what is happening and what you need help with."
    )


# ---------------------------------
# Context fetcher for live information
# ---------------------------------
def build_external_context(user_input: str, topic: str):
    text = (user_input or "").strip()
    if not text:
        return ""

    keywords = [
        "weather", "forecast", "market", "price", "latest", "today", "news", "current",
        "malaria", "health", "disease", "agriculture", "farming", "policy",
    ]
    if not any(k in text.lower() for k in keywords):
        return ""

    try:
        result = fetch_web_summary(f"{topic} {text}")
        if not result:
            return ""
        return result[:800]
    except Exception:
        return ""


# ---------------------------------
# Main answer generator
# ---------------------------------
def generate_answer(user_input: str, language: str, profile_name: str, topic: str, chat_messages=None):
    if not user_input or not user_input.strip():
        return "Please type your message so I can help.", {"blocked": False, "reasons": []}

    safe_state = safety.check_request(user_input, context=f"{topic} conversation")
    if safe_state["blocked"]:
        response = (
            "I can't assist with that request. If this is a safety issue, contact emergency services or a local crisis hotline immediately."
        )
        if safe_state.get("sensitive_topic"):
            response = safety.add_disclaimer(response)
        return response, safe_state

    cleaned_input = safety.redact_input(user_input)
    external_context = build_external_context(cleaned_input, topic)

    client = get_twin_client()
    api_key = os.environ.get("GROQ_API_KEY", "")

    if client is not None and api_key:
        try:
            system_instruction = get_default_system_prompt(language, profile_name, topic)
            if external_context:
                system_instruction += (
                    "\n\nFresh external context (use only when relevant and clearly labeled):\n"
                    f"{external_context}"
                )

            reply, usage = client.chat(
                cleaned_input,
                system_instruction=system_instruction,
                temperature=0.3,
                max_tokens=300,
                context_strategy="sliding_window",
            )

            validation = safety.validate_output(reply, context=f"{topic} conversation")
            if not validation["valid"]:
                reply = "I can help with general wellness guidance and safe next steps. Please share more detail about your symptoms."

            return reply, safe_state
        except Exception:
            pass

    fallback = local_twin_response(cleaned_input, language=language, profile_name=profile_name, topic=topic)
    validation = safety.validate_output(fallback, context=f"{topic} conversation")
    if not validation["valid"]:
        fallback = "I can help with general wellness guidance and safe next steps. Please share more detail about your symptoms."
    return fallback, safe_state


# ---------------------------------
# Session state chat management
# ---------------------------------
def default_chat_payload(title="General health", topic="health"):
    return {
        "id": f"chat-{uuid.uuid4().hex[:8]}",
        "title": title,
        "topic": topic,
        "messages": [{"role": "assistant", "content": "Hello! I am your digital twin assistant. How can I help you today?"}],
        "updated_at": datetime.now().isoformat(),
    }


def init_session():
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "chat-1": {
                "id": "chat-1",
                "title": "General health",
                "topic": "health",
                "messages": [{"role": "assistant", "content": "Hello! I am your digital twin assistant. Ask me about symptoms, routines, or next steps."}],
                "updated_at": datetime.now().isoformat(),
            }
        }

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = "chat-1"

    if "language" not in st.session_state:
        st.session_state.language = "en"

    if "profile_name" not in st.session_state:
        st.session_state.profile_name = "Aster"

    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    if "twin_clients" not in st.session_state:
        st.session_state.twin_clients = {}


init_session()


# ---------------------------------
# UI theme
# ---------------------------------
def apply_theme(theme_name: str):
    if theme_name == "dark":
        st.markdown(
            """
            <style>
            .stApp { background: #0f172a; color: #e2e8f0; }
            div[data-testid=stSidebar] { background: #111827; }
            .chat-card { background: #111827; border-radius: 12px; padding: 12px; border: 1px solid #334155; }
            .safe-box { background: #0b1120; border: 1px solid #334155; border-radius: 10px; padding: 12px; }
            .sidebar-button { width: 100%; text-align: left; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background: #f8fafc; color: #0f172a; }
            div[data-testid=stSidebar] { background: #f1f5f9; }
            .chat-card { background: white; border-radius: 12px; padding: 12px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08); }
            .safe-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; }
            .sidebar-button { width: 100%; text-align: left; }
            </style>
            """,
            unsafe_allow_html=True,
        )


apply_theme(st.session_state.theme)


# ---------------------------------
# Sidebar / chat selection
# ---------------------------------
with st.sidebar:
    st.title("🧠 Twin Workspace")
    st.caption("Multi-session digital twin")

    if st.button("+ New chat", use_container_width=True):
        new_id = f"chat-{uuid.uuid4().hex[:8]}"
        st.session_state.chats[new_id] = default_chat_payload(title="New conversation", topic="general")
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.markdown("### Conversations")
    for chat_id, chat in st.session_state.chats.items():
        active = chat_id == st.session_state.active_chat_id
        button_label = f"{chat['title']} • {chat['topic']}"
        if st.button(button_label, key=f"sidebar_{chat_id}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    st.markdown("---")
    st.selectbox("Language", ["en", "sw"], index=0 if st.session_state.language == "en" else 1, key="language")
    st.text_input("Twin profile name", key="profile_name")
    st.radio("Theme", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1, key="theme")

    with st.expander("Session settings"):
        st.text_input("Current topic", key="topic_override")
        if st.button("Apply topic", use_container_width=True):
            chat = st.session_state.chats[st.session_state.active_chat_id]
            chat["topic"] = st.session_state.topic_override.strip() or chat["topic"]
            chat["title"] = (chat["title"] or "New conversation")
            st.rerun()

    st.download_button(
        "Export all chats",
        data=json.dumps(st.session_state.chats, ensure_ascii=False, indent=2),
        file_name="digital_twin_chats.json",
        mime="application/json",
    )


# ---------------------------------
# Active chat view
# ---------------------------------
active_chat = st.session_state.chats[st.session_state.active_chat_id]

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title(f"{active_chat['title']}")
with header_col2:
    st.markdown(f"<div class='safe-box'>Topic: {active_chat['topic']}</div>", unsafe_allow_html=True)

for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# quick actions
quick_prompts = {
    "en": [
        "I have a fever and headache",
        "What should I eat to stay healthy?",
        "I feel stressed and overwhelmed",
        "What are emergency warning signs?",
    ],
    "sw": [
        "Ninaumwa na kichwa na homa",
        "Ninahitaji chakula cha afya",
        "Ninahisi stressed na nimeshakuwa na wasiwasi",
        "Niambie ishara za dharura",
    ],
}

actions = quick_prompts[st.session_state.language]
button_cols = st.columns(2)
for idx, prompt in enumerate(actions):
    with button_cols[idx % 2]:
        if st.button(prompt, key=f"qa_{idx}", use_container_width=True):
            user_prompt = prompt
            st.session_state.pending_prompt = user_prompt
            st.rerun()

if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = ""
else:
    user_prompt = st.chat_input("Ask the twin about this topic...")

if user_prompt:
    active_chat["messages"].append({"role": "user", "content": user_prompt})
    active_chat["updated_at"] = datetime.now().isoformat()

    with st.spinner("Thinking..."):
        try:
            response, safety_state = generate_answer(
                user_prompt,
                language=st.session_state.language,
                profile_name=st.session_state.profile_name,
                topic=active_chat["topic"],
                chat_messages=active_chat["messages"],
            )
        except Exception:
            response = "I hit a temporary error while generating a response. Please try a simpler question or ask again in a moment."
            safety_state = {"blocked": False, "reasons": ["error"]}

    active_chat["messages"].append({"role": "assistant", "content": response})
    active_chat["updated_at"] = datetime.now().isoformat()
    st.rerun()


# footer and status
with st.container():
    st.markdown("---")
    st.caption("Multi-chat digital twin with Groq if configured, with safe local fallback and external context support.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Active chat", active_chat["title"])
    c2.metric("Topic", active_chat["topic"])
    c3.metric("Messages", len(active_chat["messages"]))

    if st.button("Voice input", use_container_width=False):
        st.components.v1.html(
            """
            <script>
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                alert('Voice input is not supported here.');
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

# Final app note
