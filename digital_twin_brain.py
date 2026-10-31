import os

from twin import (
    GroqClient,
    ContextWindowManager,
    PersonalDataExtractor,
    ChunkingSystem,
    SimpleRetriever,
    TokenCounter,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class DigitalTwinBrain:
    """Thin orchestration layer that reuses the project’s original twin.py utilities."""

    def __init__(self, api_key=None, model=None, context_manager=None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or DEFAULT_MODEL
        self.context_manager = context_manager or ContextWindowManager(
            max_tokens=8192,
            reserve_for_response=512,
        )
        self.client = GroqClient(
            api_key=self.api_key,
            model=self.model,
            context_manager=self.context_manager,
        )
        self.extractor = PersonalDataExtractor()
        self.chunker = ChunkingSystem()
        self.retriever = SimpleRetriever()
        self.counter = TokenCounter()

    def _normalize_text(self, text):
        if text is None:
            return ""
        return " ".join(str(text).split()).strip()

    def _memory_from_history(self, chat_history):
        if not chat_history:
            return {"preferences": [], "topics": [], "facts": []}

        memory = {"preferences": [], "topics": [], "facts": []}
        for message in chat_history:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            text = self._normalize_text(message.get("content"))
            if not text:
                continue

            extracted = self.extractor.extract(text)
            if extracted.get("preferences"):
                memory["preferences"].extend(extracted["preferences"])
            if extracted.get("hashtags"):
                memory["topics"].extend(extracted["hashtags"])
            if len(text) > 18:
                memory["facts"].append(text)

        memory["preferences"] = list(dict.fromkeys(memory["preferences"]))[-5:]
        memory["topics"] = list(dict.fromkeys(memory["topics"]))[-6:]
        memory["facts"] = memory["facts"][-5:]
        return memory

    def _build_memory_summary(self, chat_history=None, memory_summary=""):
        chat_history = chat_history or []
        extracted = self._memory_from_history(chat_history)
        parts = []

        if memory_summary:
            parts.append(memory_summary)
        if extracted.get("preferences"):
            parts.append("Preferences: " + ", ".join(extracted["preferences"]))
        if extracted.get("topics"):
            parts.append("Topics: " + ", ".join(extracted["topics"]))
        if extracted.get("facts"):
            parts.append("Recent facts: " + "; ".join(extracted["facts"][-2:]))

        summary = " | ".join(parts).strip()
        return summary[:800] if summary else ""

    def _recent_chat_context(self, chat_history, user_message):
        if not chat_history:
            return []

        cleaned = []
        for entry in chat_history:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            text = self._normalize_text(entry.get("content"))
            if role in {"user", "assistant"} and text:
                cleaned.append({"role": role, "content": text})

        if cleaned and cleaned[-1].get("role") == "user" and cleaned[-1].get("content", "") == self._normalize_text(user_message):
            cleaned = cleaned[:-1]

        return cleaned[-8:]

    def _retrieval_context(self, chat_history, user_message):
        prior = self._recent_chat_context(chat_history, user_message)
        if not prior:
            return ""

        texts = [item["content"] for item in prior if item.get("content")]
        if not texts:
            return ""

        chunks = self.chunker.chunk("\n".join(texts), strategy="smart", source="chat_history")
        self.retriever.index(chunks)
        hits = self.retriever.retrieve(user_message, top_k=3)
        if not hits:
            return ""
        return "Prior relevant context:\n" + "\n".join(f"- {item['text'][:220]}" for item in hits)

    def route(self, user_message, profile_name="Aster", topic="general", language="en", external_context="", chat_history=None, memory_summary=""):
        if not user_message or not self._normalize_text(user_message):
            return {"reply": "Please type a message so I can help.", "usage": {}}

        user_message = self._normalize_text(user_message)
        prior_context = self._recent_chat_context(chat_history or [], user_message)
        memory_summary = self._build_memory_summary(chat_history or [], memory_summary)
        retrieved = self._retrieval_context(chat_history or [], user_message)

        system_instruction = (
            f"You are {profile_name}, a digitally aware personal assistant and digital twin. "
            f"You are helping in the topic: {topic}. "
            "Respond as a thoughtful, calm, practical, and human assistant. "
            "Use the user's context and prior conversation naturally. "
            "Keep answers grounded, helpful, and conversational, not robotic. "
            f"Answer in {language}. "
            "Do not diagnose and prioritize safety when relevant."
        )

        if memory_summary:
            system_instruction += f"\nMemory summary: {memory_summary}"
        if retrieved:
            system_instruction += f"\n\n{retrieved}"
        if external_context:
            system_instruction += f"\n\nFresh external context:\n{external_context[:700]}"

        self.client.reset()
        self.client.conversation_history.append({"role": "system", "content": system_instruction})
        for item in prior_context:
            self.client.conversation_history.append({
                "role": item.get("role", "user"),
                "content": item.get("content", "")
            })

        reply, usage = self.client.chat(
            user_message,
            system_instruction=None,
            temperature=0.35,
            max_tokens=350,
            context_strategy="sliding_window",
        )
        return {"reply": reply, "usage": usage, "memory_summary": memory_summary}
