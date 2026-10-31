import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import requests
import os
import re
import time
import json
import inspect
from typing import Any, Callable, List, Dict, get_type_hints
from datetime import datetime

# ------------------------------------------------------------
# Existing twin utility code
# ------------------------------------------------------------
activity_names = {0: 'Sleep', 1: 'Work', 2: 'Exercise', 3: 'Social', 4: 'Leisure'}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.loss_history = []

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate=0.1):
        m = X.shape[0]
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            self.loss_history.append(loss)
            self.backward(X, y, output, learning_rate)
            if verbose and epoch % 200 == 0:
                print(f"  Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)


# ------------------------------------------------------------
# Groq / LLM integration
# ------------------------------------------------------------
MODEL_NAME = "llama-3.3-70b-versatile"


class UniversalLLMClient:
    """Generic OpenAI-compatible client — works with Groq, Together, local servers, etc."""
    def __init__(self, base_url, api_key, model=MODEL_NAME, context_manager=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.context_manager = context_manager

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, system_instruction, user_prompt, temperature=0.3, max_tokens=300, context_strategy="sliding_window"):
        if self.context_manager is not None:
            messages = self.context_manager.fit_to_budget(
                system_instruction,
                [{"role": "user", "content": user_prompt}],
                strategy=context_strategy,
            )
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class GroqClient:
    """
    Wrapper around the Groq (OpenAI-compatible) chat API.
    Initialize once, reuse across the whole conversation/twin session.
    Tracks conversation history (for multi-turn chat) and total tokens used.
    """
    def __init__(self, api_key, base_url="https://api.groq.com/openai/v1", model=None, context_manager=None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model or MODEL_NAME
        self.context_manager = context_manager

        self.conversation_history = []
        self.total_tokens_used = 0
        self.call_log = []

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def reset(self):
        self.conversation_history = []

    def chat(self, user_message, system_instruction=None, temperature=0.3, max_tokens=300, context_strategy="sliding_window"):
        if system_instruction and not self.conversation_history:
            self.conversation_history.append({"role": "system", "content": system_instruction})

        self.conversation_history.append({"role": "user", "content": user_message})

        if self.context_manager is not None:
            sys_msgs = [m for m in self.conversation_history if m.get('role') == 'system']
            convo_msgs = [m for m in self.conversation_history if m.get('role') != 'system']
            system_text = sys_msgs[0]['content'] if sys_msgs else None
            messages_to_send = self.context_manager.fit_to_budget(
                system_text,
                convo_msgs,
                strategy=context_strategy,
            )
        else:
            messages_to_send = self.conversation_history

        payload = {
            "model": self.model,
            "messages": messages_to_send,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        reply = data["choices"][0]["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": reply})

        usage = data.get("usage", {})
        tokens_this_call = usage.get("total_tokens", 0)
        self.total_tokens_used += tokens_this_call
        self.call_log.append(usage)

        return reply, usage


# ------------------------------------------------------------
# Existing PersonalDataExtractor, chunking, retriever, and token counter code
# ------------------------------------------------------------
class PersonalDataExtractor:
    """
    Exercise 2.1: Build a Personal Data Extractor.
    Pulls structured data out of raw, messy, real-life text (notes, chat
    logs, journal entries, typos and all): identifiers, temporal info,
    and hashtag-based preferences/activities.
    """

    EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    PHONE_RE = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{3,4}')
    HANDLE_RE = re.compile(r'(?<!\w)@([A-Za-z0-9_]{2,30})')
    HASHTAG_RE = re.compile(r'(?<!\w)#([A-Za-z0-9_]{2,40})')
    DATE_RE = re.compile(
        r'\b(\d{4}-\d{2}-\d{2}'
        r'|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        r'|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:\s+\d{2,4})?'
        r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{2,4})?)\b',
        re.IGNORECASE)
    TIME_RE = re.compile(r'\b(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?|\d{1,2}\s?(?:am|pm|AM|PM))\b')
    DURATION_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|h\b|minutes?|mins?|m\b|days?|weeks?)\b', re.IGNORECASE)

    @classmethod
    def extract(cls, text):
        text = text or ""

        emails = cls.EMAIL_RE.findall(text)
        phones = [p.strip() for p in cls.PHONE_RE.findall(text) if len(re.sub(r'\D', '', p)) >= 7]
        handles = cls.HANDLE_RE.findall(text)
        hashtags = cls.HASHTAG_RE.findall(text)
        dates = cls.DATE_RE.findall(text)
        times = cls.TIME_RE.findall(text)
        durations = [f"{amt} {unit}" for amt, unit in cls.DURATION_RE.findall(text)]
        preferences = sorted(set(h.lower() for h in hashtags))

        return {
            'emails': sorted(set(emails)),
            'phones': sorted(set(phones)),
            'social_handles': sorted(set(handles)),
            'hashtags': sorted(set(hashtags)),
            'dates': sorted(set(dates)),
            'times': sorted(set(times)),
            'durations': sorted(set(durations)),
            'preferences': preferences,
            'raw_length_chars': len(text),
        }


class ChunkingSystem:
    SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, min_chunk_words=5, max_chunk_words=120, overlap_words=15):
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words

    def _split_sentences(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return []
        return [s.strip() for s in self.SENTENCE_SPLIT_RE.split(text) if s.strip()]

    def chunk_by_sentence(self, text, sentences_per_chunk=3):
        sentences = self._split_sentences(text)
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunks.append(" ".join(sentences[i:i + sentences_per_chunk]))
        return chunks

    def chunk_by_word(self, text, overlap=None):
        overlap = self.overlap_words if overlap is None else overlap
        words = text.split()
        chunks = []
        step = max(1, self.max_chunk_words - overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.max_chunk_words]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
        return chunks

    def chunk_by_paragraph(self, text):
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        return paragraphs if paragraphs else self.chunk_by_sentence(text)

    def chunk_smart(self, text):
        sentences = self._split_sentences(text)
        chunks, current, current_words = [], [], 0

        for sentence in sentences:
            w = len(sentence.split())
            if current and current_words + w > self.max_chunk_words:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            current.append(sentence)
            current_words += w

        if current:
            chunk_text = " ".join(current)
            if chunks and current_words < self.min_chunk_words:
                chunks[-1] = chunks[-1] + " " + chunk_text
            else:
                chunks.append(chunk_text)

        return chunks

    def chunk(self, text, strategy="smart", source="notes", timestamp=None, **kwargs):
        if strategy == "sentence":
            raw_chunks = self.chunk_by_sentence(text, **kwargs)
        elif strategy == "word":
            raw_chunks = self.chunk_by_word(text, **kwargs)
        elif strategy == "paragraph":
            raw_chunks = self.chunk_by_paragraph(text)
        elif strategy == "smart":
            raw_chunks = self.chunk_smart(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        ts = timestamp or datetime.utcnow().isoformat()
        chunks = []
        for i, c in enumerate(raw_chunks):
            chunks.append({
                'text': c,
                'position': i,
                'source': source,
                'timestamp': ts,
                'word_count': len(c.split()),
                'strategy': strategy,
            })
        return chunks

    def validate_chunks(self, chunks):
        issues = []
        valid = []
        for c in chunks:
            wc = c['word_count']
            if wc < self.min_chunk_words:
                issues.append(f"Chunk {c['position']} ({c['source']}): too short ({wc} words) — possible orphaned sentence.")
            elif wc > self.max_chunk_words * 1.5:
                issues.append(f"Chunk {c['position']} ({c['source']}): too long ({wc} words) — consider splitting further.")
            else:
                valid.append(c)
        return valid, issues

    def visualize(self, chunks, title="Chunk Size Distribution"):
        word_counts = [c['word_count'] for c in chunks]
        if not word_counts:
            print("No chunks to visualize.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].hist(word_counts, bins=min(15, max(3, len(word_counts))), color='steelblue', edgecolor='white')
        axes[0].axvline(self.min_chunk_words, color='red', linestyle='--', label='min')
        axes[0].axvline(self.max_chunk_words, color='green', linestyle='--', label='max')
        axes[0].set_title(title)
        axes[0].set_xlabel("Words per chunk")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        axes[1].bar(range(len(word_counts)), word_counts, color='seagreen')
        axes[1].set_title("Chunk Size by Position")
        axes[1].set_xlabel("Chunk position")
        axes[1].set_ylabel("Word count")

        plt.tight_layout()
        plt.show()


class SimpleRetriever:
    TOKEN_RE = re.compile(r'[a-z0-9]+')

    def __init__(self):
        self.chunks = []
        self.vocab = {}
        self.doc_vectors = None

    @classmethod
    def _tokenize(cls, text):
        return cls.TOKEN_RE.findall(text.lower())

    def index(self, chunks):
        self.chunks = chunks
        token_lists = [self._tokenize(c['text']) for c in chunks]

        vocab = {}
        for tokens in token_lists:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab

        matrix = np.zeros((len(chunks), len(vocab)))
        for row, tokens in enumerate(token_lists):
            counts = Counter(tokens)
            for t, c in counts.items():
                matrix[row, vocab[t]] = c
        self.doc_vectors = matrix

    def _vectorize_query(self, query):
        vec = np.zeros(len(self.vocab))
        for t in self._tokenize(query):
            if t in self.vocab:
                vec[self.vocab[t]] += 1
        return vec

    def retrieve(self, query, top_k=3):
        if self.doc_vectors is None or len(self.chunks) == 0:
            return []

        q_vec = self._vectorize_query(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        doc_norms = np.linalg.norm(self.doc_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1e-9

        sims = (self.doc_vectors @ q_vec) / (doc_norms * q_norm)
        top_indices = np.argsort(-sims)[:top_k]

        return [{**self.chunks[i], 'similarity': float(sims[i])} for i in top_indices if sims[i] > 0]


class TokenCounter:
    """Estimates token usage for text."""
    def __init__(self, cache_limit=5000):
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None
        self._cache = {}
        self.cache_limit = cache_limit

    def count_tokens(self, text):
        if text is None:
            return 0
        if not isinstance(text, str):
            text = str(text)
        if text == "":
            return 0

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        if self.encoding is not None:
            try:
                n = len(self.encoding.encode(text))
            except Exception:
                n = max(1, len(text) // 4)
        else:
            n = max(1, len(text) // 4)

        self._cache[text] = n
        if len(self._cache) > self.cache_limit:
            self._cache = dict(list(self._cache.items())[-self.cache_limit:])
        return n


# ------------------------------------------------------------
# Context window manager used by app / Groq integration
# ------------------------------------------------------------
class ContextWindowManager:
    """Trim system + convo prompts to fit a token budget."""
    def __init__(self, max_tokens=8192, reserve_for_response=256, system_weight=1.0):
        self.max_tokens = max_tokens
        self.reserve_for_response = reserve_for_response
        self.system_weight = system_weight
        self.counter = TokenCounter()

    def fit_to_budget(self, system_instruction, messages, strategy="sliding_window"):
        if not system_instruction and not messages:
            return []

        system_text = system_instruction or ""
        budget = self.max_tokens - self.reserve_for_response

        if not messages:
            return [{"role": "system", "content": system_text}] if system_text else []

        safe_messages = []
        if system_text:
            safe_messages.append({"role": "system", "content": system_text})
        safe_messages.extend(messages)

        total_tokens = sum(self.counter.count_tokens(m.get("content", "")) for m in safe_messages)
        if total_tokens <= budget:
            return safe_messages

        if strategy == "sliding_window":
            kept = []
            current_tokens = 0
            for m in reversed(safe_messages):
                t = self.counter.count_tokens(m.get("content", ""))
                if current_tokens + t <= budget:
                    kept.insert(0, m)
                    current_tokens += t
                if current_tokens >= budget:
                    break
            return kept

        if system_text:
            safe_messages = [{"role": "system", "content": system_text}] + safe_messages[-1:]
        else:
            safe_messages = safe_messages[-2:]
        return safe_messages


class DigitalTwinBrain:
    """High-level orchestration layer for the digital twin app."""

    def __init__(self, api_key, base_url="https://api.groq.com/openai/v1", model=None, context_manager=None):
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self.model = model or MODEL_NAME
        self.context_manager = context_manager or ContextWindowManager(max_tokens=8192, reserve_for_response=512)
        self.client = GroqClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            context_manager=self.context_manager,
        )
        self.extractor = PersonalDataExtractor()
        self.chunker = ChunkingSystem()
        self.retriever = SimpleRetriever()

    def _memory_from_history(self, chat_history):
        if not chat_history:
            return {}

        memory = {"preferences": [], "topics": [], "facts": []}
        for message in chat_history:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            text = (message.get("content") or "").strip()
            if role != "user" or not text:
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
        history = chat_history or []
        extracted = self._memory_from_history(history)
        summary_parts = []

        if memory_summary:
            summary_parts.append(memory_summary)
        if extracted.get("preferences"):
            summary_parts.append("Preferences: " + ", ".join(extracted["preferences"]))
        if extracted.get("topics"):
            summary_parts.append("Topics: " + ", ".join(extracted["topics"]))
        if extracted.get("facts"):
            summary_parts.append("Recent facts: " + "; ".join(extracted["facts"][-2:]))

        summary = " | ".join(summary_parts).strip()
        return summary[:800] if summary else ""

    def _recent_chat_context(self, chat_history, user_message):
        if not chat_history:
            return []

        cleaned = []
        for entry in chat_history:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            text = (entry.get("content") or "").strip()
            if role in {"user", "assistant"} and text:
                cleaned.append({"role": role, "content": text})

        if cleaned and cleaned[-1].get("role") == "user" and cleaned[-1].get("content", "").strip() == user_message.strip():
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
        if not user_message or not str(user_message).strip():
            return {"reply": "Please type a message so I can help.", "usage": {}}

        user_message = str(user_message).strip()
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
            "Do not diagnose, and prioritize safety where relevant."
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


# ------------------------------------------------------------
# Compatibility safety hooks used elsewhere in the project
# ------------------------------------------------------------
try:
    import digital_twin_safety as safety_module
    safety = safety_module.DigitalTwinSafetySystem()
except Exception:
    safety = None


# ------------------------------------------------------------
# Datasets / examples
# ------------------------------------------------------------
# Keep the rest of the original project code as-is below this point.

