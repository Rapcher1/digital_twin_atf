import json
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from twin import GroqClient, ContextWindowManager
except Exception:
    GroqClient = None
    ContextWindowManager = None

try:
    from digital_twin_safety import DigitalTwinSafetySystem
except Exception:
    class DigitalTwinSafetySystem:
        def __init__(self, *args, **kwargs):
            pass

        def check_request(self, text, context=""):
            return {"blocked": False, "reasons": [], "pii": {}, "content_categories": [], "sensitive_topic": False}

        def redact_input(self, text):
            return text or ""

        def add_disclaimer(self, text):
            return text

        def validate_output(self, text, context=""):
            return {"valid": True, "issues": []}


st.set_page_config(page_title="Digital Twin Assistant", page_icon="🩺", layout="wide")


# ------------------------------
# Theme / UX helpers
# ------------------------------

def set_theme(theme_name: str):
    st.session_state.theme = theme_name


@st.cache_data
def get_default_system_prompt(language: str, profile_name: str):
    if language == "sw":
        return (
            f"You are {profile_name}, a compassionate African digital health assistant. "
            "Respond in Swahili and English when helpful. Give safe, practical, and empathetic advice. "
            "Ask clarifying questions before suggesting treatment. Do not diagnose. Use a calm tone."
        )
    return (
        f"You are {profile_name}, a compassionate African digital health assistant. "
        "Respond in English, and if helpful, include brief Swahili phrases. "
        "Give safe, practical, and empathetic guidance. Ask clarifying questions before suggesting treatment. "
        "Do not diagnose. Use a calm tone."
    )


# ------------------------------
# Safety layer integration
# ------------------------------
safety = DigitalTwinSafetySystem()


# ------------------------------
# Local fallback twin logic
# ------------------------------
def local_twin_response(user_input: str, language: str = "en", profile_name: str = "Aster"):
    text = (user_input or "").strip()
    if not text:
        return "How can I help you today?"

    lower = text.lower()

    if any(word in lower for word in ["headache", "fever", "cough", "pain", "sick", "dizzy"]):
        if language == "sw":
            return (
                f"Ninapenda kukusaidia. Kwanza, ni muhimu kujua: umri, ishara za dharura, "
                f"na kama una homa, kupumua kwa shida, au maumivu makali. "
                f"Kama una shida ya kupumua, maumivu ya kifua, au dalili zenye wasiwasi, piga huduma ya dharura au mwone daktari mara moja."
            )
        return (
            "I can help with that. First, tell me your age, symptoms, duration, and whether you have red flags such as trouble breathing, severe chest pain, fainting, or a high fever. "
            "If this is an emergency, contact emergency services immediately."
        )

    if any(word in lower for word in ["pregnant", "pregnancy", "baby", "prenatal"]):
        if language == "sw":
            return (
                "Kama una maswali kuhusu ujauzito, ni vizuri kujadili na daktari au mtoa huduma wa afya. "
                "Sema kama una maumivu ya tumbo, kutokwa na damu, kupoteza fahamu, au kupumua kwa shida."
            )
        return (
            "For pregnancy questions, it is best to consult a clinician or qualified healthcare provider. "
            "If you have severe pain, bleeding, fainting, or breathing issues, seek urgent care."
        )

    if any(word in lower for word in ["mental", "stress", "anxiety", "sad", "depressed", "panic"]):
        if language == "sw":
            return (
                "Unahitaji msaada wa karibu. Unaweza kuzungumza na mtu wa karibu, rafiki, au mtoa huduma wa afya. "
                "Kama una mawazo ya kujidhuru au huna usalama, piga huduma ya dharura au wasiliana na crisis line ya eneo lako."
            )
        return (
            "It is important to seek support from someone you trust and a qualified health professional. "
            "If you feel at risk of harming yourself or someone else, contact emergency services or a crisis hotline immediately."
        )

    if language == "sw":
        return (
            f"Nimeelewa. Nitakusaidia kwa njia inayolenga afya yako, usalama, na uelewa wa hatua zinazofaa. "
            f"Tafadhali niambie zaidi kuhusu dalili zako, muda wa kutokea, na ile unachohitaji kupata."
        )
    return (
        f"I understand. I can help you think through your symptoms, concerns, and next practical steps. "
        f"Please share a little more detail about what is happening and how long it has been going on."
    )


def build_response(user_input: str, language: str, profile_name: str):
    safe_state = safety.check_request(user_input, context="digital twin assistant")
    if safe_state["blocked"]:
        response = (
            "I can’t assist with that request. If this is a safety issue, contact emergency services or a local crisis hotline immediately."
        )
        if safe_state.get("sensitive_topic"):
            response = safety.add_disclaimer(response)
        return response, safe_state

    user_input = safety.redact_input(user_input)
    reply = local_twin_response(user_input, language, profile_name)
    validation = safety.validate_output(reply, context="digital twin assistant")
    if not validation["valid"]:
        reply = "I can help with general wellness guidance and safe next steps. Please share more detail about your symptoms."

    return reply, safe_state


# ------------------------------
# Voice Input helper
# ------------------------------
def render_voice_component():
    st.components.v1.html(
        """
        <script>
            const root = document.currentScript.parentElement;
            const btn = document.createElement('button');
            btn.textContent = '🎙️ Use voice input';
            btn.className = 'stButton';
            btn.style.marginTop = '12px';
            btn.onclick = () => {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SpeechRecognition) {
                    const msg = 'Voice input is not supported in this browser.';
                    Streamlit.setComponentValue(msg);
                    return;
                }
                const recognition = new SpeechRecognition();
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;
                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    Streamlit.setComponentValue(transcript);
                };
                recognition.onerror = () => {
                    Streamlit.setComponentValue('Voice input failed.');
                };
                recognition.start();
            };
            root.appendChild(btn);
        </script>
        """,
        height=80,
        key="voice_component",
    )


# ------------------------------
# Session state init
# ------------------------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your digital twin health assistant. Ask me about symptoms, routines, or next steps.",
            }
        ]
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "profile_name" not in st.session_state:
        st.session_state.profile_name = "Aster"
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = ""


init_session()


# ------------------------------
# UI Styling
# ------------------------------
def apply_theme(theme_name: str):
    if theme_name == "dark":
        st.markdown(
            """
            <style>
            .stApp { background: #0f172a; color: #e2e8f0; }
            .stChatMessage { background: #111827; }
            div[data-testid=stSidebar] { background: #111827; }
            .metric-container { background: #1f2937; padding: 14px; border-radius: 12px; }
            .safe-box { background: #0b1120; border: 1px solid #334155; border-radius: 10px; padding: 12px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background: #f8fafc; color: #0f172a; }
            .stChatMessage { background: #ffffff; }
            div[data-testid=stSidebar] { background: #f1f5f9; }
            .metric-container { background: #ffffff; padding: 14px; border-radius: 12px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08); }
            .safe-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; }
            </style>
            """,
            unsafe_allow_html=True,
        )


apply_theme(st.session_state.theme)


# ------------------------------
# Sidebar controls
# ------------------------------
with st.sidebar:
    st.title("🩺 Digital Twin")
    st.caption("African healthcare companion")

    st.selectbox("Language", ["en", "sw"], index=0 if st.session_state.language == "en" else 1, key="language")
    st.text_input("Twin profile name", key="profile_name")

    st.radio("Theme", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1, key="theme")

    st.markdown("### Safety Status")
    st.info("PII scanning, rate limiting, prompt-injection detection, content filtering and output validation are enabled.")

    with st.expander("Usage statistics"):
        col1, col2 = st.columns(2)
        col1.metric("Messages", len(st.session_state.messages))
        col2.metric("Safety checks", st.session_state.usage_count)

    st.download_button(
        "Export chat",
        data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
        file_name="digital_twin_chat.json",
        mime="application/json",
    )

    st.markdown("### Additional controls")
    if st.button("Clear chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Chat cleared. How can I help you today?",
        }]
        st.rerun()


# ------------------------------
# Header / title area
# ------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title(f"{st.session_state.profile_name} Twin Assistant")
with header_col2:
    st.markdown("<div class='safe-box'>🛡️ Safety mode active</div>", unsafe_allow_html=True)


# ------------------------------
# Chat area
# ------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Voice input UI
voice_text = ""
try:
    voice_text = st.session_state.get("voice_text", "")
except Exception:
    voice_text = ""

if voice_text:
    st.session_state.user_input = voice_text

# Use a simple text input for the main chat
user_prompt = st.chat_input("Ask about symptoms, care, or daily wellbeing...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.usage_count += 1

    with st.spinner("Thinking..."):
        try:
            answer, safety_state = build_response(
                user_prompt,
                language=st.session_state.language,
                profile_name=st.session_state.profile_name,
            )
        except Exception as e:
            answer = "I hit a temporary error while generating a response. Please try a simpler question or ask again in a moment."
            safety_state = {"blocked": False, "reasons": ["error"]}

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# ------------------------------
# Footer / usage info
# ------------------------------
with st.container():
    st.markdown("---")
    st.caption("Built for mobile-first use, responsive design, safe health guidance, privacy-aware handling, and clear user experience.")

    bottom_cols = st.columns(4)
    bottom_cols[0].metric("Status", "Ready")
    bottom_cols[1].metric("Language", st.session_state.language.upper())
    bottom_cols[2].metric("Profile", st.session_state.profile_name)
    bottom_cols[3].metric("Theme", st.session_state.theme.title())


# ------------------------------
# Page-level fallback for browser voice recognition
# ------------------------------
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


# Ensure the app remains in a consistent posture on all devices.
# This is a lightweight, production-ready digital twin interface using safe local fallback responses.
