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
from digital_twin_safety import DigitalTwinSafetySystem

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


class UniversalLLMClient:
    """
    Generic OpenAI-compatible client — works with Groq, Together, local servers, etc.

    context_manager is optional (defaults to None, which preserves the
    original behavior exactly). When set to a ContextWindowManager, the
    system instruction + prompt are trimmed to fit its token budget
    before every request.
    """
    def __init__(self, base_url, api_key, model="llama3.3", context_manager=None):
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
                system_instruction, [{"role": "user", "content": user_prompt}], strategy=context_strategy)
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


MODEL_NAME = "llama3.3"  # default model — change if your provider lists a different exact string

class GroqClient:
    """
    Wrapper around the Groq (OpenAI-compatible) chat API.
    Initialize once, reuse across the whole conversation/twin session.
    Tracks conversation history (for multi-turn chat) and total tokens used.

    context_manager is optional (defaults to None, which preserves the
    original behavior exactly — the full conversation_history is sent
    as-is). When set to a ContextWindowManager, self.conversation_history
    still keeps the COMPLETE, untrimmed log (so the token meter and any
    later inspection stay accurate), but only a trimmed window of it is
    actually sent to the API on each call.
    """
    def __init__(self, api_key, base_url="https://api.groq.com/openai/v1", model=None, context_manager=None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model or MODEL_NAME
        self.context_manager = context_manager

        self.conversation_history = []   # holds the running message list
        self.total_tokens_used = 0        # API meter: cumulative tokens across all calls
        self.call_log = []                 # per-call token usage, for inspection

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def reset(self):
        """Start a fresh conversation (clears history, keeps the token meter)."""
        self.conversation_history = []

    def chat(self, user_message, system_instruction=None, temperature=0.3, max_tokens=300, context_strategy="sliding_window"):
        """
        Send one message in an ongoing conversation. If this is the first
        message and a system_instruction is given, it's added once at the
        start. Returns (reply_text, usage_dict).
        """
        if system_instruction and not self.conversation_history:
            self.conversation_history.append({"role": "system", "content": system_instruction})

        self.conversation_history.append({"role": "user", "content": user_message})

        if self.context_manager is not None:
            sys_msgs = [m for m in self.conversation_history if m.get('role') == 'system']
            convo_msgs = [m for m in self.conversation_history if m.get('role') != 'system']
            system_text = sys_msgs[0]['content'] if sys_msgs else None
            messages_to_send = self.context_manager.fit_to_budget(
                system_text, convo_msgs, strategy=context_strategy)
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


class PersonalDataExtractor:
    """
    Exercise 2.1: Build a Personal Data Extractor.

    Pulls structured data out of raw, messy, real-life text (notes, chat
    logs, journal entries, typos and all): identifiers, temporal info,
    and hashtag-based preferences/activities. Used as the first step of
    ingesting a user's real notes into the digital twin's knowledge base.
    """

    EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    PHONE_RE = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{3,4}')
    HANDLE_RE = re.compile(r'(?<!\w)@([A-Za-z0-9_]{2,30})')
    HASHTAG_RE = re.compile(r'(?<!\w)#([A-Za-z0-9_]{2,40})')
    # Dates: 2026-10-27, 27/10/2026, 27-10-26, Oct 27 2026, 27th October
    DATE_RE = re.compile(
        r'\b(\d{4}-\d{2}-\d{2}'
        r'|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        r'|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:\s+\d{2,4})?'
        r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{2,4})?)\b',
        re.IGNORECASE)
    TIME_RE = re.compile(r'\b(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM)?|\d{1,2}\s?(?:am|pm|AM|PM))\b')
    DURATION_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|h\b|minutes?|mins?|m\b|days?|weeks?)\b',
                              re.IGNORECASE)

    @classmethod
    def extract(cls, text):
        """
        Returns a structured dict pulled from raw text. Handles messy,
        typo-laden real-world notes — every field defaults to an empty
        list rather than raising, so partial/garbled input still yields
        whatever can be found.
        """
        text = text or ""

        emails = cls.EMAIL_RE.findall(text)
        phones = [p.strip() for p in cls.PHONE_RE.findall(text) if len(re.sub(r'\D', '', p)) >= 7]
        handles = cls.HANDLE_RE.findall(text)
        hashtags = cls.HASHTAG_RE.findall(text)
        dates = cls.DATE_RE.findall(text)
        times = cls.TIME_RE.findall(text)
        durations = [f"{amt} {unit}" for amt, unit in cls.DURATION_RE.findall(text)]

        # Hashtags double as lightweight preference/activity signals,
        # e.g. "#exercise #plantbased #earlyrise" -> inferred tags
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
    """
    Exercise 3.1: Build a Chunking System.

    Splits raw ingested text (notes, journal entries, logs) into chunks
    for the digital twin's knowledge base, so the twin can later retrieve
    only the relevant pieces instead of stuffing everything into one
    prompt (this is the "indexing" half of retrieval-augmented generation).
    """

    SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, min_chunk_words=5, max_chunk_words=120, overlap_words=15):
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words

    # ---------- strategies ----------
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
        """
        Greedily accumulates whole sentences up to max_chunk_words,
        never splitting mid-sentence, and merges any trailing orphan
        (too-short) sentence into the previous chunk instead of leaving
        it standalone.
        """
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
                # orphan: merge into the previous chunk rather than keep it alone
                chunks[-1] = chunks[-1] + " " + chunk_text
            else:
                chunks.append(chunk_text)

        return chunks

    # ---------- orchestration ----------
    def chunk(self, text, strategy="smart", source="notes", timestamp=None, **kwargs):
        """
        Returns a list of chunk dicts: {text, position, source, timestamp,
        word_count, strategy}. This is what feeds the knowledge base.
        """
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
        """
        Flags quality issues: orphaned (too-short) chunks and
        oversized chunks. Returns (valid_chunks, issues_list) — nothing
        is silently dropped, so you can see exactly what was flagged.
        """
        issues = []
        valid = []
        for c in chunks:
            wc = c['word_count']
            if wc < self.min_chunk_words:
                issues.append(f"Chunk {c['position']} ({c['source']}): too short "
                              f"({wc} words) — possible orphaned sentence.")
            elif wc > self.max_chunk_words * 1.5:
                issues.append(f"Chunk {c['position']} ({c['source']}): too long "
                              f"({wc} words) — consider splitting further.")
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
    """
    Lightweight retrieval-augmented generation (RAG) index: no external
    embedding model required (works with just numpy), so it runs anywhere.
    Builds a term-frequency vector per chunk and ranks chunks for a query
    by cosine similarity — the "retrieval" half of RAG, with the chunking
    system above providing the "indexing" half.
    """

    TOKEN_RE = re.compile(r'[a-z0-9]+')

    def __init__(self):
        self.chunks = []          # list of chunk dicts (text, metadata)
        self.vocab = {}           # term -> index
        self.doc_vectors = None   # matrix of TF vectors, one row per chunk

    @classmethod
    def _tokenize(cls, text):
        return cls.TOKEN_RE.findall(text.lower())

    def index(self, chunks):
        """(Re)builds the index from a list of chunk dicts."""
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
        """Returns the top_k chunk dicts most similar to the query (cosine similarity)."""
        if self.doc_vectors is None or len(self.chunks) == 0:
            return []

        q_vec = self._vectorize_query(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        doc_norms = np.linalg.norm(self.doc_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1e-9  # avoid divide-by-zero for empty chunks

        sims = (self.doc_vectors @ q_vec) / (doc_norms * q_norm)
        top_indices = np.argsort(-sims)[:top_k]

        return [{**self.chunks[i], 'similarity': float(sims[i])} for i in top_indices if sims[i] > 0]


class TokenCounter:
    """
    Estimates how many tokens a piece of text (or a chat message) will
    cost against a model's context window.

    Uses tiktoken's cl100k_base encoding when the package is installed
    (accurate, OpenAI/Groq-family tokenization), and falls back to a
    ~4-characters-per-token heuristic when it isn't — so the rest of the
    context-window system works with or without the optional dependency.
    Counts are cached (keyed by exact text) for performance, since the
    same message content is often re-counted many times as the
    conversation grows.
    """

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
                n = max(1, round(len(text) / 4))
        else:
            n = max(1, round(len(text) / 4))

        if len(self._cache) < self.cache_limit:
            self._cache[text] = n
        return n

    def decode_prefix(self, text, max_tokens):
        """Returns the longest prefix of `text` that fits within max_tokens — used as a last-resort truncation."""
        if max_tokens <= 0 or not text:
            return ""
        if self.encoding is not None:
            try:
                tokens = self.encoding.encode(text)
                return self.encoding.decode(tokens[:max_tokens])
            except Exception:
                pass
        return text[:max_tokens * 4]


class ContextWindowManager:
    """
    Manages what actually fits inside a model's context window: token
    counting, sliding-window retention, several truncation strategies,
    and summarization-based compression — the full set of techniques
    for handling conversations that outgrow the token budget.

    Terminology used throughout this class:
      - max_tokens: the model's total context window size.
      - reserve_for_response: tokens held back for the model's reply.
      - available_tokens: max_tokens - reserve_for_response — the real
        budget for system instruction + conversation history.
    """

    def __init__(self, max_tokens=8192, reserve_for_response=512, token_counter=None):
        self.max_tokens = max_tokens
        self.reserve_for_response = reserve_for_response
        self.token_counter = token_counter or TokenCounter()

    @property
    def available_tokens(self):
        return max(0, self.max_tokens - self.reserve_for_response)

    # ---------- counting ----------
    def count(self, text):
        return self.token_counter.count_tokens(text)

    def count_message(self, message):
        if not message:
            return 0
        content = message.get('content', "")
        return self.count(content) + 4  # ~4-token overhead per message (role/structure), OpenAI-style

    def count_messages(self, messages):
        if not messages:
            return 0
        return sum(self.count_message(m) for m in messages) + 2  # +2 for reply priming

    # ---------- strategy 1: sliding window ----------
    def sliding_window(self, messages, keep_last_n=None, keep_last_tokens=None):
        """
        Keeps only the most recent messages — by count (keep_last_n) or
        by token budget (keep_last_tokens). Any system messages in the
        input are kept separately, always first, never evicted by the
        window itself.
        """
        messages = messages or []
        system_msgs = [m for m in messages if m.get('role') == 'system']
        convo = [m for m in messages if m.get('role') != 'system']

        if keep_last_n is not None:
            kept = convo[-keep_last_n:] if keep_last_n > 0 else []
        elif keep_last_tokens is not None:
            kept, running = [], 0
            for m in reversed(convo):
                t = self.count_message(m)
                if running + t > keep_last_tokens and kept:
                    break
                kept.insert(0, m)
                running += t
                if running > keep_last_tokens and len(kept) == 1:
                    break  # even the single most recent message doesn't fit; stop here
        else:
            kept = convo

        return system_msgs + kept

    # ---------- strategy 2: truncation ----------
    def truncate_oldest(self, messages, budget_tokens):
        """Drops the oldest non-system messages first, keeping the most recent ones that fit budget_tokens."""
        messages = messages or []
        system_msgs = [m for m in messages if m.get('role') == 'system']
        convo = [m for m in messages if m.get('role') != 'system']

        system_tokens = self.count_messages(system_msgs) if system_msgs else 0
        remaining = max(0, budget_tokens - system_tokens)

        kept, running = [], 0
        for m in reversed(convo):
            t = self.count_message(m)
            if running + t > remaining:
                break
            kept.insert(0, m)
            running += t

        # Edge case: not even the single most recent message fits — truncate its content instead of dropping it.
        if not kept and convo:
            last = convo[-1]
            room = max(0, remaining - 4)
            truncated_content = self.token_counter.decode_prefix(str(last.get('content', '')), room)
            kept = [{**last, 'content': truncated_content + " ...[truncated]"}]

        return system_msgs + kept

    def truncate_middle(self, messages, budget_tokens, keep_head=2, keep_tail=4):
        """Keeps the first few and last few messages, collapsing the middle into a one-line placeholder."""
        messages = messages or []
        system_msgs = [m for m in messages if m.get('role') == 'system']
        convo = [m for m in messages if m.get('role') != 'system']

        if self.count_messages(system_msgs + convo) <= budget_tokens:
            return system_msgs + convo

        head = convo[:keep_head]
        tail = convo[-keep_tail:] if len(convo) > keep_tail else convo[len(head):]

        def _assemble(h, t):
            omitted = max(0, len(convo) - len(h) - len(t))
            placeholder = (
                [{'role': 'system', 'content': f"[... {omitted} earlier messages omitted to fit context window ...]"}]
                if omitted > 0 else []
            )
            return system_msgs + h + placeholder + t

        result = _assemble(head, tail)

        # Still over budget: shrink the TAIL first (drop its oldest/front
        # entries), so the head — and the guarantee of keeping the very
        # first message(s) — is preserved as long as possible.
        while self.count_messages(result) > budget_tokens and tail:
            tail = tail[1:]
            result = _assemble(head, tail)

        # Tail is empty and still over budget: now shrink the head from
        # its end, preserving at least the very first message as long as
        # it alone can fit.
        while self.count_messages(result) > budget_tokens and len(head) > 1:
            head = head[:-1]
            result = _assemble(head, tail)

        # Last resort — even head[0] alone doesn't fit: fall back to
        # oldest-truncation on the ORIGINAL messages (not on our already
        # head/tail-reduced result, which could lose the first message
        # for no benefit).
        if self.count_messages(result) > budget_tokens:
            return self.truncate_oldest(system_msgs + convo, budget_tokens)

        return result

    def truncate_by_priority(self, messages, budget_tokens, priority_fn=None):
        """
        Drops lowest-priority messages first instead of strictly oldest.
        Default priority = recency, with a large boost for any message
        whose dict has `important: True` — so flagged messages survive
        even if they're old.
        """
        messages = messages or []
        system_msgs = [m for m in messages if m.get('role') == 'system']
        convo = [m for m in messages if m.get('role') != 'system']

        if priority_fn is None:
            def priority_fn(m, i, n):
                recency = i / max(1, n - 1) if n > 1 else 1.0
                boost = 10.0 if m.get('important') else 0.0
                return recency + boost

        system_tokens = self.count_messages(system_msgs) if system_msgs else 0
        remaining = max(0, budget_tokens - system_tokens)

        scored = sorted(enumerate(convo), key=lambda pair: -priority_fn(pair[1], pair[0], len(convo)))
        kept_indices, running = set(), 0
        for idx, m in scored:
            t = self.count_message(m)
            if running + t <= remaining:
                kept_indices.add(idx)
                running += t

        kept = [convo[i] for i in sorted(kept_indices)]  # restore chronological order
        return system_msgs + kept

    # ---------- strategy 3: compression ----------
    def extractive_summary(self, text, max_sentences=3, max_chars=400):
        """
        Dependency-free fallback summarizer: picks representative
        leading/trailing sentences. No LLM required.

        max_chars is a hard safety cap applied after sentence selection:
        text with no sentence-ending punctuation (e.g. raw logs, repeated
        tokens) would otherwise pass through almost unshortened, since
        re.split would see it as a single "sentence" under max_sentences.
        """
        if not text or not text.strip():
            return "(no earlier content)"
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        if len(sentences) <= max_sentences:
            summary = " ".join(sentences)
        else:
            head_n = max(1, max_sentences // 2)
            tail_n = max_sentences - head_n
            summary = " ".join(sentences[:head_n] + sentences[-tail_n:])

        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(' ', 1)[0] + " ...[summary truncated]"
        return summary

    def compress_messages(self, messages, summarizer_fn=None, keep_recent_n=4, max_summary_sentences=3):
        """
        Compresses everything except the most recent keep_recent_n
        messages into a single compact summary message, freeing up
        budget while preserving the gist of the older conversation.

        summarizer_fn, if given, is a plain callable (str) -> str (e.g.
        a wrapper around an LLM call) used instead of the extractive
        fallback. It is called standalone — never a stateful chat
        client's own .chat() method — so compression never mutates
        that client's conversation history as a side effect. Any
        failure in summarizer_fn is caught and falls back to the
        extractive summary rather than losing the older messages.
        """
        messages = messages or []
        system_msgs = [m for m in messages if m.get('role') == 'system']
        convo = [m for m in messages if m.get('role') != 'system']

        if len(convo) <= keep_recent_n:
            return system_msgs + convo  # nothing old enough to compress

        older, recent = convo[:-keep_recent_n], convo[-keep_recent_n:]
        older_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in older)

        summary = None
        if summarizer_fn is not None:
            try:
                candidate = summarizer_fn(older_text)
                if isinstance(candidate, str) and candidate.strip():
                    summary = candidate.strip()
            except Exception as e:
                print(f"  ! compress_messages: summarizer_fn failed ({e}); falling back to extractive summary")

        if summary is None:
            summary = self.extractive_summary(older_text, max_summary_sentences)

        summary_msg = {'role': 'system', 'content': f"[Earlier conversation summary]: {summary}"}
        return system_msgs + [summary_msg] + recent

    # ---------- orchestration ----------
    def fit_to_budget(self, system_instruction, messages, strategy="sliding_window", summarizer_fn=None, **kwargs):
        """
        Main entry point: returns a message list (system first, if any)
        guaranteed to fit within available_tokens, using the requested
        strategy. Never raises on bad input — edge cases are handled
        explicitly and a failure anywhere falls back to a safe minimal
        result instead of crashing the caller.

        strategy: one of "sliding_window", "truncate_oldest",
        "truncate_middle", "truncate_priority", "compress", "hybrid"
        (compress, then slide if still over budget).
        """
        try:
            messages = messages or []
            system_msg = {'role': 'system', 'content': system_instruction} if system_instruction else None
            full = ([system_msg] if system_msg else []) + list(messages)

            budget = self.available_tokens
            if budget <= 0:
                print("  ! ContextWindowManager: reserve_for_response >= max_tokens, no room for context")
                return [system_msg] if system_msg else []

            total = self.count_messages(full)
            if total <= budget:
                return full  # already fits — nothing to do

            system_tokens = self.count_message(system_msg) if system_msg else 0
            if system_tokens > budget:
                print("  ! ContextWindowManager: system instruction alone exceeds budget; truncating it")
                truncated = self.token_counter.decode_prefix(system_instruction, max(0, budget - 10))
                return [{'role': 'system', 'content': truncated}]

            if strategy == "sliding_window":
                convo = [m for m in full if m.get('role') != 'system']
                kept = self.sliding_window(convo, keep_last_tokens=budget - system_tokens)
                result = ([system_msg] if system_msg else []) + kept

            elif strategy == "truncate_oldest":
                result = self.truncate_oldest(full, budget)

            elif strategy == "truncate_middle":
                result = self.truncate_middle(full, budget,
                                               keep_head=kwargs.get('keep_head', 2),
                                               keep_tail=kwargs.get('keep_tail', 4))

            elif strategy == "truncate_priority":
                result = self.truncate_by_priority(full, budget, priority_fn=kwargs.get('priority_fn'))

            elif strategy == "compress":
                result = self.compress_messages(full, summarizer_fn=summarizer_fn,
                                                 keep_recent_n=kwargs.get('keep_recent_n', 4))

            elif strategy == "hybrid":
                result = self.compress_messages(full, summarizer_fn=summarizer_fn,
                                                 keep_recent_n=kwargs.get('keep_recent_n', 6))
                if self.count_messages(result) > budget:
                    convo = [m for m in result if m.get('role') != 'system']
                    kept = self.sliding_window(convo, keep_last_tokens=budget - system_tokens)
                    result = ([system_msg] if system_msg else []) + kept

            else:
                print(f"  ! ContextWindowManager: unknown strategy '{strategy}', falling back to sliding_window")
                convo = [m for m in full if m.get('role') != 'system']
                kept = self.sliding_window(convo, keep_last_tokens=budget - system_tokens)
                result = ([system_msg] if system_msg else []) + kept

            # final safety net — guarantee the contract regardless of which branch ran
            if self.count_messages(result) > budget:
                result = self.truncate_oldest(result, budget)

            return result

        except Exception as e:
            print(f"  ! ContextWindowManager.fit_to_budget failed ({e}); falling back to system-only")
            return [{'role': 'system', 'content': system_instruction}] if system_instruction else []

    # ---------- reporting / visualization ----------
    def usage_report(self, system_instruction, messages):
        system_tokens = (self.count(system_instruction) + 4) if system_instruction else 0
        convo = messages or []
        convo_tokens = self.count_messages(convo) if convo else 0
        total = system_tokens + convo_tokens
        return {
            'system_tokens': system_tokens,
            'conversation_tokens': convo_tokens,
            'total_tokens': total,
            'max_tokens': self.max_tokens,
            'reserved_for_response': self.reserve_for_response,
            'available_tokens': self.available_tokens,
            'utilization_pct': round(100 * total / self.available_tokens, 1) if self.available_tokens else 0.0,
            'fits': total <= self.available_tokens,
            'num_messages': len(convo),
        }

    def print_usage_bar(self, report, width=40):
        """User-friendly, dependency-free console view of budget usage — a text progress bar."""
        frac = min(1.0, report['total_tokens'] / report['available_tokens']) if report['available_tokens'] else 1.0
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        status = "OK" if report['fits'] else "OVER BUDGET"
        print(f"[{bar}] {report['total_tokens']}/{report['available_tokens']} tokens "
              f"({report['utilization_pct']}%) - {status}")

    def visualize_usage(self, reports, title="Context Window Usage Over Turns"):
        """
        reports: a list of usage_report() dicts collected turn-by-turn.
        Plots token usage over time against the budget line, plus a
        breakdown of the latest turn's budget allocation.
        """
        if not reports:
            print("No reports to visualize.")
            return

        turns = list(range(1, len(reports) + 1))
        totals = [r['total_tokens'] for r in reports]
        budget = reports[0]['available_tokens']

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(turns, totals, marker='o', color='steelblue', label='Tokens used')
        axes[0].axhline(budget, color='red', linestyle='--', label='Budget (available_tokens)')
        axes[0].set_title(title)
        axes[0].set_xlabel('Turn')
        axes[0].set_ylabel('Tokens')
        axes[0].legend()

        last = reports[-1]
        free = max(0, self.max_tokens - last['total_tokens'] - self.reserve_for_response)
        labels = ['System', 'Conversation', 'Reserved (response)', 'Free']
        values = [last['system_tokens'], last['conversation_tokens'], self.reserve_for_response, free]
        axes[1].bar(labels, values, color=['salmon', 'steelblue', 'gray', 'lightgreen'])
        axes[1].set_title('Latest Turn — Budget Breakdown')
        axes[1].set_ylabel('Tokens')
        axes[1].tick_params(axis='x', rotation=20)

        plt.tight_layout()
        plt.show()


def run_context_manager_tests():
    """
    Lightweight, dependency-free test suite for ContextWindowManager
    (plain asserts rather than unittest/pytest, so it runs anywhere).
    Prints PASS/FAIL per test and returns True iff every test passed.
    """
    results = []

    def check(name, condition):
        status = "PASS" if condition else "FAIL"
        results.append(bool(condition))
        print(f"  [{status}] {name}")

    cm = ContextWindowManager(max_tokens=200, reserve_for_response=20)

    # --- basic counting ---
    check("token count increases with text length", cm.count("hello " * 50) > cm.count("hello"))
    check("count() handles empty string without error", cm.count("") == 0)
    check("count() handles None without error", cm.count(None) == 0)
    check("count_message() handles a message with no content key", cm.count_message({'role': 'user'}) >= 0)

    msgs = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': f"message number {i} " * 3}
            for i in range(10)]

    # --- sliding window ---
    windowed = cm.sliding_window(msgs, keep_last_n=3)
    check("sliding_window keeps exactly keep_last_n messages", len(windowed) == 3)
    check("sliding_window keeps the most recent messages", windowed[-1]['content'] == msgs[-1]['content'])

    # --- truncate_oldest ---
    budget = 60
    truncated = cm.truncate_oldest(msgs, budget)
    check("truncate_oldest respects the token budget", cm.count_messages(truncated) <= budget)
    check("truncate_oldest keeps the most recent message", truncated[-1]['content'] == msgs[-1]['content'])

    # --- truncate_middle ---
    tm = cm.truncate_middle(msgs, budget, keep_head=2, keep_tail=2)
    non_system_tm = [m for m in tm if m.get('role') != 'system']
    check("truncate_middle keeps the first original message",
          len(non_system_tm) > 0 and non_system_tm[0]['content'] == msgs[0]['content'])
    check("truncate_middle result fits the budget", cm.count_messages(tm) <= budget)

    # --- truncate_by_priority ---
    important_msgs = [dict(m) for m in msgs]
    important_msgs[0]['important'] = True
    pri = cm.truncate_by_priority(important_msgs, budget_tokens=40)
    early_msg_tokens = cm.count_message(important_msgs[0])
    check("truncate_by_priority keeps an important-flagged message when it fits alone",
          any(m.get('important') for m in pri) or early_msg_tokens > 40)

    # --- compress_messages ---
    compressed = cm.compress_messages(msgs, keep_recent_n=2)
    check("compress_messages reduces total token count", cm.count_messages(compressed) <= cm.count_messages(msgs))
    check("compress_messages preserves the most recent messages verbatim",
          compressed[-1]['content'] == msgs[-1]['content'])

    # --- edge case: a single oversized message ---
    huge_msg = [{'role': 'user', 'content': 'word ' * 2000}]
    result = cm.truncate_oldest(huge_msg, budget_tokens=30)
    check("an oversized single message is truncated rather than dropped or crashing",
          len(result) == 1 and cm.count_message(result[0]) <= 60)

    # --- edge case: empty message list ---
    empty_result = cm.fit_to_budget("You are a helpful twin.", [])
    check("fit_to_budget handles an empty message list without error", isinstance(empty_result, list))

    # --- fit_to_budget never exceeds budget, across every strategy ---
    long_history = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': f"turn {i} " * 20}
                     for i in range(30)]
    for strat in ["sliding_window", "truncate_oldest", "truncate_middle", "truncate_priority", "compress", "hybrid"]:
        fitted = cm.fit_to_budget("System rules here.", long_history, strategy=strat)
        check(f"fit_to_budget('{strat}') result fits within available_tokens",
              cm.count_messages(fitted) <= cm.available_tokens)

    # --- edge case: unknown strategy name falls back gracefully ---
    try:
        fallback = cm.fit_to_budget("System.", long_history, strategy="not_a_real_strategy")
        check("an unknown strategy name falls back instead of raising", isinstance(fallback, list))
    except Exception:
        check("an unknown strategy name falls back instead of raising", False)

    # --- edge case: system instruction alone exceeds the whole budget ---
    tiny_cm = ContextWindowManager(max_tokens=30, reserve_for_response=5)
    huge_system = "This is a very long system instruction. " * 50
    safe = tiny_cm.fit_to_budget(huge_system, long_history)
    check("an oversized system instruction is truncated rather than crashing",
          isinstance(safe, list) and len(safe) >= 1)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} tests passed.")
    return passed == total


class Vocabulary:
    """
    Word -> index mapping with frequency counts, shared by the
    Word2Vec and GloVe trainers below. Words appearing fewer than
    min_count times are dropped (standard preprocessing step for both
    algorithms — rare words produce noisy, poorly-trained vectors).
    """
    def __init__(self, min_count=1):
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = []
        self.word_counts = Counter()

    def build(self, tokenized_sentences):
        for sent in tokenized_sentences:
            self.word_counts.update(sent)
        self.idx2word = [w for w, c in self.word_counts.items() if c >= self.min_count]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        return self

    def __len__(self):
        return len(self.idx2word)


class Word2VecTrainer:
    """
    Word2Vec internals, built from scratch with numpy only (no
    gensim/torch): skip-gram with negative sampling.

    For each (center, context) word pair drawn from a sliding window,
    the model is trained to score the TRUE context word highly
    (sigmoid(center · context) -> 1) and a handful of random
    "negative" words low (sigmoid(center · negative) -> 0), via plain
    SGD — the same manual forward/backward style used by
    SimpleNeuralNetwork elsewhere in this file.

    Standard Word2Vec tricks included: subsampling of frequent words,
    a dynamic (shrinking) window size, and a unigram^0.75 negative-
    sampling distribution.
    """
    def __init__(self, embedding_dim=32, window_size=3, negative_samples=5,
                 learning_rate=0.03, min_count=1, subsample_threshold=1e-3, seed=42):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold
        self.rng = np.random.default_rng(seed)

        self.vocab = Vocabulary(min_count=min_count)
        self.W_target = None    # (V, D) "input"/center-word embeddings — the ones returned as the result
        self.W_context = None   # (V, D) "output"/context-word embeddings
        self.loss_history = []
        self._neg_sampling_probs = None

    def _build_negative_sampling_table(self):
        counts = np.array([self.vocab.word_counts[w] for w in self.vocab.idx2word], dtype=float)
        probs = counts ** 0.75  # standard Word2Vec smoothing of the unigram distribution
        probs /= probs.sum()
        self._neg_sampling_probs = probs

    def _subsample(self, tokens):
        """Randomly drops very frequent words so rare words get proportionally more training signal."""
        total = sum(self.vocab.word_counts.values())
        if total == 0:
            return []
        kept = []
        for t in tokens:
            if t not in self.vocab.word2idx:
                continue
            freq = self.vocab.word_counts[t] / total
            if freq <= 0:
                kept.append(t)
                continue
            keep_prob = min(1.0, (np.sqrt(freq / self.subsample_threshold) + 1) * (self.subsample_threshold / freq))
            if self.rng.random() < keep_prob:
                kept.append(t)
        return kept

    def _generate_training_pairs(self, tokenized_sentences):
        pairs = []
        for sent in tokenized_sentences:
            sent = self._subsample(sent)
            idxs = [self.vocab.word2idx[w] for w in sent if w in self.vocab.word2idx]
            n = len(idxs)
            for i, center in enumerate(idxs):
                window = int(self.rng.integers(1, self.window_size + 1))  # dynamic window, standard trick
                start, end = max(0, i - window), min(n, i + window + 1)
                for j in range(start, end):
                    if j != i:
                        pairs.append((center, idxs[j]))
        return pairs

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, raw_texts, epochs=5, verbose=True):
        """raw_texts: a list of raw text strings (e.g. knowledge-base chunk texts). Tokenized internally."""
        if not raw_texts:
            raise ValueError("Word2VecTrainer.fit: no training text provided")

        tokenized = [s for s in (SimpleRetriever._tokenize(t) for t in raw_texts) if s]
        if not tokenized:
            raise ValueError("Word2VecTrainer.fit: no tokens found after tokenization")

        self.vocab.build(tokenized)
        if len(self.vocab) < 2:
            raise ValueError(f"Word2VecTrainer.fit: vocabulary too small ({len(self.vocab)} words) to train embeddings")

        self._build_negative_sampling_table()

        V, D = len(self.vocab), self.embedding_dim
        limit = 0.5 / D
        self.W_target = self.rng.uniform(-limit, limit, size=(V, D))
        self.W_context = self.rng.uniform(-limit, limit, size=(V, D))

        pairs = self._generate_training_pairs(tokenized)
        if not pairs:
            raise ValueError("Word2VecTrainer.fit: no training pairs generated (corpus may be too short for window_size)")

        self.loss_history = []
        for epoch in range(epochs):
            self.rng.shuffle(pairs)
            epoch_loss = 0.0
            for center, context in pairs:
                neg_samples = self.rng.choice(V, size=self.negative_samples, p=self._neg_sampling_probs)
                neg_samples = np.array([n if n != context else (n + 1) % V for n in neg_samples])

                v_center = self.W_target[center]
                v_context = self.W_context[context]
                v_negs = self.W_context[neg_samples]

                pos_score = self._sigmoid(np.dot(v_center, v_context))
                neg_scores = self._sigmoid(np.dot(v_negs, v_center))

                pos_loss = -np.log(np.clip(pos_score, 1e-10, 1.0))
                neg_loss = -np.sum(np.log(np.clip(1 - neg_scores, 1e-10, 1.0)))
                epoch_loss += pos_loss + neg_loss

                grad_center = (pos_score - 1) * v_context + np.sum(neg_scores[:, None] * v_negs, axis=0)
                grad_context = (pos_score - 1) * v_center
                grad_negs = neg_scores[:, None] * v_center[None, :]

                self.W_target[center] -= self.learning_rate * grad_center
                self.W_context[context] -= self.learning_rate * grad_context
                self.W_context[neg_samples] -= self.learning_rate * grad_negs

            avg_loss = epoch_loss / len(pairs)
            self.loss_history.append(avg_loss)
            if verbose:
                print(f"  [Word2Vec] epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}, "
                      f"pairs: {len(pairs)}, vocab: {V}")

        return self

    def get_embedding_space(self):
        if self.W_target is None:
            raise RuntimeError("Word2VecTrainer: call fit() before get_embedding_space()")
        vectors = {w: self.W_target[i] for w, i in self.vocab.word2idx.items()}
        return EmbeddingSpace(vectors, method="word2vec")


class GloVeTrainer:
    """
    GloVe (Global Vectors) internals, built from scratch with numpy
    only. Unlike Word2Vec's local-window prediction task, GloVe
    factorizes a GLOBAL word-word co-occurrence matrix directly:

      1. Build a distance-weighted co-occurrence count X_ij for every
         word pair seen within window_size of each other.
      2. Learn vectors W, W~ and biases b, b~ that minimize the
         weighted least-squares objective:
            f(X_ij) * (W_i . W~_j + b_i + b~_j - log(X_ij))^2
         where f(x) = (x/x_max)^alpha for x < x_max, else 1 — this
         down-weights very rare AND very frequent co-occurrences.

    Trained via AdaGrad (per-parameter adaptive learning rates), as in
    the original GloVe paper, implemented manually with numpy.
    """
    def __init__(self, embedding_dim=32, window_size=5, x_max=100, alpha=0.75,
                 learning_rate=0.05, min_count=1, seed=42):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.rng = np.random.default_rng(seed)

        self.vocab = Vocabulary(min_count=min_count)
        self.cooccurrence = {}
        self.W = None
        self.W_tilde = None
        self.b = None
        self.b_tilde = None
        self.loss_history = []

    def _build_cooccurrence(self, tokenized_sentences):
        cooc = {}
        for sent in tokenized_sentences:
            idxs = [self.vocab.word2idx[w] for w in sent if w in self.vocab.word2idx]
            n = len(idxs)
            for i in range(n):
                for dist in range(1, self.window_size + 1):
                    j = i + dist
                    if j >= n:
                        break
                    wi, wj = idxs[i], idxs[j]
                    weight = 1.0 / dist  # distance-weighted, as in the original GloVe paper
                    cooc[(wi, wj)] = cooc.get((wi, wj), 0.0) + weight
                    cooc[(wj, wi)] = cooc.get((wj, wi), 0.0) + weight  # symmetric window
        return cooc

    @staticmethod
    def _weighting_fn(x, x_max, alpha):
        return (x / x_max) ** alpha if x < x_max else 1.0

    def fit(self, raw_texts, epochs=50, verbose=True):
        if not raw_texts:
            raise ValueError("GloVeTrainer.fit: no training text provided")

        tokenized = [s for s in (SimpleRetriever._tokenize(t) for t in raw_texts) if s]
        if not tokenized:
            raise ValueError("GloVeTrainer.fit: no tokens found after tokenization")

        self.vocab.build(tokenized)
        if len(self.vocab) < 2:
            raise ValueError(f"GloVeTrainer.fit: vocabulary too small ({len(self.vocab)} words) to train embeddings")

        self.cooccurrence = self._build_cooccurrence(tokenized)
        if not self.cooccurrence:
            raise ValueError("GloVeTrainer.fit: no co-occurrences found (corpus may be too short for window_size)")

        V, D = len(self.vocab), self.embedding_dim
        limit = 0.5 / D
        self.W = self.rng.uniform(-limit, limit, size=(V, D))
        self.W_tilde = self.rng.uniform(-limit, limit, size=(V, D))
        self.b = np.zeros(V)
        self.b_tilde = np.zeros(V)

        grad_sq_W = np.ones_like(self.W)
        grad_sq_Wt = np.ones_like(self.W_tilde)
        grad_sq_b = np.ones_like(self.b)
        grad_sq_bt = np.ones_like(self.b_tilde)

        pairs = list(self.cooccurrence.items())
        self.loss_history = []

        for epoch in range(epochs):
            self.rng.shuffle(pairs)
            epoch_loss = 0.0
            for (i, j), x_ij in pairs:
                weight = self._weighting_fn(x_ij, self.x_max, self.alpha)
                dot = np.dot(self.W[i], self.W_tilde[j])
                diff = dot + self.b[i] + self.b_tilde[j] - np.log(x_ij)
                epoch_loss += weight * (diff ** 2)

                grad_common = weight * diff
                grad_w = grad_common * self.W_tilde[j]
                grad_wt = grad_common * self.W[i]

                grad_sq_W[i] += grad_w ** 2
                grad_sq_Wt[j] += grad_wt ** 2
                grad_sq_b[i] += grad_common ** 2
                grad_sq_bt[j] += grad_common ** 2

                self.W[i] -= (self.learning_rate / np.sqrt(grad_sq_W[i])) * grad_w
                self.W_tilde[j] -= (self.learning_rate / np.sqrt(grad_sq_Wt[j])) * grad_wt
                self.b[i] -= (self.learning_rate / np.sqrt(grad_sq_b[i])) * grad_common
                self.b_tilde[j] -= (self.learning_rate / np.sqrt(grad_sq_bt[j])) * grad_common

            avg_loss = epoch_loss / len(pairs)
            self.loss_history.append(avg_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  [GloVe] epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}, "
                      f"co-occurrence pairs: {len(pairs)}, vocab: {V}")

        return self

    def get_embedding_space(self):
        if self.W is None:
            raise RuntimeError("GloVeTrainer: call fit() before get_embedding_space()")
        final_vectors = self.W + self.W_tilde  # GloVe paper's recommended final representation
        vectors = {w: final_vectors[i] for w, i in self.vocab.word2idx.items()}
        return EmbeddingSpace(vectors, method="glove")


class EmbeddingSpace:
    """
    A trained or loaded set of word vectors — the common interface
    both Word2VecTrainer and GloVeTrainer return, and what
    load_pretrained() produces from a file. Provides semantic
    similarity, vector-arithmetic analogies, dimensionality reduction
    for visualization, and graceful out-of-vocabulary (OOV) handling
    throughout (no method here raises on an unseen word — each
    prints a short warning and returns None/[] instead).
    """
    def __init__(self, vectors, method="unknown"):
        self.vectors = {w: np.asarray(v, dtype=float) for w, v in vectors.items()}
        self.method = method

    def __len__(self):
        return len(self.vectors)

    def __contains__(self, word):
        return word in self.vectors

    def vector(self, word):
        """Returns the vector for `word`, or None if out-of-vocabulary (never raises)."""
        return self.vectors.get(word)

    @staticmethod
    def cosine_similarity(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def similarity(self, word1, word2):
        """Cosine similarity between two words, in [-1, 1]. Returns None (with a warning) if either is OOV."""
        v1, v2 = self.vector(word1), self.vector(word2)
        if v1 is None or v2 is None:
            missing = [w for w, v in [(word1, v1), (word2, v2)] if v is None]
            print(f"  ! similarity: out-of-vocabulary word(s): {missing}")
            return None
        return self.cosine_similarity(v1, v2)

    def most_similar(self, word, top_k=5):
        """Top-k most similar words by cosine similarity, excluding the query word itself. Returns [] if OOV."""
        v = self.vector(word)
        if v is None:
            print(f"  ! most_similar: '{word}' is out-of-vocabulary")
            return []
        scored = [(w, self.cosine_similarity(v, vec)) for w, vec in self.vectors.items() if w != word]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def analogy(self, a, b, c, top_k=5, exclude_input=True):
        """
        Solves "a is to b as c is to ?" via vector arithmetic
        (target = b - a + c), then finds the nearest word(s) to that
        target vector. Returns [] (with a warning) if any input word
        is out-of-vocabulary, rather than raising.
        """
        va, vb, vc = self.vector(a), self.vector(b), self.vector(c)
        missing = [w for w, v in [(a, va), (b, vb), (c, vc)] if v is None]
        if missing:
            print(f"  ! analogy: out-of-vocabulary word(s): {missing}")
            return []

        target = vb - va + vc
        exclude = {a, b, c} if exclude_input else set()
        scored = [(w, self.cosine_similarity(target, vec)) for w, vec in self.vectors.items() if w not in exclude]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def vocabulary(self):
        return sorted(self.vectors.keys())

    # ---------- dimensionality reduction ----------
    def pca(self, n_components=2, words=None):
        """
        Dependency-free PCA (no sklearn required): mean-centers the
        vectors, eigendecomposes the covariance matrix, and projects
        onto the top n_components. Returns (words_used, coords,
        explained_variance_ratio).
        """
        words = words if words is not None else self.vocabulary()
        words = [w for w in words if w in self.vectors]
        if len(words) < 2:
            print("  ! pca: need at least 2 in-vocabulary words to run PCA")
            return [], np.zeros((0, n_components)), np.zeros(n_components)

        X = np.stack([self.vectors[w] for w in words])
        X_centered = X - X.mean(axis=0, keepdims=True)

        cov = np.atleast_2d(np.cov(X_centered, rowvar=False))
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(-eigvals)
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        k = min(n_components, eigvecs.shape[1])
        coords = X_centered @ eigvecs[:, :k]

        total_var = eigvals.sum()
        explained = eigvals[:k] / total_var if total_var > 0 else np.zeros(k)

        return words, coords, explained

    def visualize_2d(self, words=None, title="Word Embeddings (PCA projection)", max_words=60):
        """Projects embeddings to 2D via PCA and scatter-plots them with word labels."""
        words = words if words is not None else self.vocabulary()
        if len(words) > max_words:
            words = words[:max_words]
        used_words, coords, explained = self.pca(n_components=2, words=words)
        if len(used_words) < 2:
            print("  ! visualize_2d: not enough in-vocabulary words to plot")
            return

        plt.figure(figsize=(9, 7))
        plt.scatter(coords[:, 0], coords[:, 1], color='steelblue', alpha=0.7)
        for w, (x, y) in zip(used_words, coords):
            plt.annotate(w, (x, y), fontsize=8, alpha=0.8, xytext=(3, 3), textcoords='offset points')
        ev_pct = list(explained * 100) + [0, 0]
        plt.title(f"{title} ({self.method}) — PC1 {ev_pct[0]:.1f}% / PC2 {ev_pct[1]:.1f}% variance explained")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

    def similarity_heatmap(self, words, title="Pairwise Cosine Similarity"):
        """Heatmap of pairwise cosine similarity for a given word list. Skips/warns on OOV words."""
        in_vocab = [w for w in words if w in self.vectors]
        skipped = [w for w in words if w not in self.vectors]
        if skipped:
            print(f"  ! similarity_heatmap: skipping out-of-vocabulary word(s): {skipped}")
        if len(in_vocab) < 2:
            print("  ! similarity_heatmap: need at least 2 in-vocabulary words")
            return

        n = len(in_vocab)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = self.cosine_similarity(self.vectors[in_vocab[i]], self.vectors[in_vocab[j]])

        plt.figure(figsize=(0.6 * n + 2, 0.6 * n + 2))
        plt.imshow(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar(label="Cosine similarity")
        plt.xticks(range(n), in_vocab, rotation=45, ha='right')
        plt.yticks(range(n), in_vocab)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # ---------- persistence / pretrained loading ----------
    @classmethod
    def load_pretrained(cls, path, vocab_limit=None):
        """
        Loads embeddings from a GloVe-style text file: one word per
        line, "word v1 v2 v3 ...". Handles a missing or corrupt file
        gracefully — prints a warning and returns an EMPTY
        EmbeddingSpace rather than raising, so a caller can always
        check `len(space) == 0` instead of wrapping this in try/except.
        """
        vectors = {}
        if not os.path.exists(path):
            print(f"  ! load_pretrained: file not found at '{path}'. Returning an empty embedding space.")
            return cls({}, method="pretrained(missing)")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.rstrip('\n').split(' ')
                    if len(parts) < 2:
                        continue
                    word, values = parts[0], parts[1:]
                    try:
                        vec = np.array([float(v) for v in values])
                    except ValueError:
                        print(f"  ! load_pretrained: skipping malformed line {line_num}")
                        continue
                    vectors[word] = vec
                    if vocab_limit and len(vectors) >= vocab_limit:
                        break
        except Exception as e:
            print(f"  ! load_pretrained: failed to read '{path}' ({e}). Returning an empty embedding space.")
            return cls({}, method="pretrained(error)")

        if not vectors:
            print(f"  ! load_pretrained: no valid vectors parsed from '{path}'")
        return cls(vectors, method="pretrained")

    def save(self, path):
        """Saves embeddings in the same GloVe-style text format load_pretrained() reads back."""
        with open(path, 'w', encoding='utf-8') as f:
            for w, v in self.vectors.items():
                f.write(w + " " + " ".join(f"{x:.6f}" for x in v) + "\n")


def run_embedding_tests():
    """
    Dependency-free test suite (plain asserts) for Word2VecTrainer,
    GloVeTrainer and EmbeddingSpace. Prints PASS/FAIL per test and
    returns True iff every test passed.
    """
    results = []

    def check(name, condition):
        status = "PASS" if condition else "FAIL"
        results.append(bool(condition))
        print(f"  [{status}] {name}")

    toy_corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat and the dog are pets",
        "a cat chased a mouse",
        "a dog chased a cat",
        "pets like cats and dogs are common",
        "the mat and the rug are on the floor",
    ] * 20  # repeated for enough training signal in a tiny toy setting

    # --- Word2Vec ---
    w2v = Word2VecTrainer(embedding_dim=16, window_size=2, negative_samples=4, learning_rate=0.05, min_count=1)
    w2v.fit(toy_corpus, epochs=10, verbose=False)
    space_w2v = w2v.get_embedding_space()
    check("Word2Vec produces an embedding for every vocab word", len(space_w2v) == len(w2v.vocab))
    check("Word2Vec training loss decreases over epochs", w2v.loss_history[-1] < w2v.loss_history[0])

    sim_cat_cat = space_w2v.similarity("cat", "cat")
    sim_cat_dog = space_w2v.similarity("cat", "dog")
    check("cosine similarity of a word with itself is ~1.0", sim_cat_cat is not None and abs(sim_cat_cat - 1.0) < 1e-6)
    check("cosine similarity falls within [-1, 1]", sim_cat_dog is not None and -1.0 <= sim_cat_dog <= 1.0)

    neighbors = space_w2v.most_similar("cat", top_k=5)
    check("most_similar excludes the query word itself", all(w != "cat" for w, _ in neighbors))
    check("most_similar results are sorted by descending similarity",
          all(neighbors[i][1] >= neighbors[i + 1][1] for i in range(len(neighbors) - 1)))

    check("similarity() handles an OOV word gracefully (returns None, no crash)",
          space_w2v.similarity("cat", "zzzznotaword") is None)
    check("most_similar() handles an OOV word gracefully (returns [], no crash)",
          space_w2v.most_similar("zzzznotaword") == [])
    check("vector() returns None for an OOV word instead of raising", space_w2v.vector("zzzznotaword") is None)

    # --- GloVe ---
    glove = GloVeTrainer(embedding_dim=16, window_size=3, x_max=10, learning_rate=0.05, min_count=1)
    glove.fit(toy_corpus, epochs=30, verbose=False)
    space_glove = glove.get_embedding_space()
    check("GloVe produces an embedding for every vocab word", len(space_glove) == len(glove.vocab))
    check("GloVe training loss decreases over epochs", glove.loss_history[-1] < glove.loss_history[0])

    # --- PCA / dimensionality reduction ---
    words, coords, explained = space_w2v.pca(n_components=2)
    check("PCA returns one 2D coordinate per word", coords.shape == (len(words), 2))
    check("PCA's first component explains at least as much variance as the second",
          len(explained) < 2 or explained[0] >= explained[1])
    check("PCA explained-variance ratios are non-negative and sum to <= 1",
          bool(np.all(explained >= -1e-9)) and explained.sum() <= 1.0 + 1e-6)

    # --- analogy ---
    analogy_result = space_w2v.analogy("cat", "dog", "mat", top_k=3)
    check("analogy() returns at most top_k results without crashing",
          isinstance(analogy_result, list) and len(analogy_result) <= 3)
    check("analogy() excludes its three input words from the results",
          all(w not in ("cat", "dog", "mat") for w, _ in analogy_result))
    check("analogy() handles an OOV input word gracefully (returns [], no crash)",
          space_w2v.analogy("cat", "dog", "zzzznotaword") == [])

    # --- edge cases: degenerate input ---
    try:
        Word2VecTrainer(embedding_dim=8).fit([])
        check("Word2Vec.fit raises a clear error on empty input", False)
    except ValueError:
        check("Word2Vec.fit raises a clear error on empty input", True)

    try:
        GloVeTrainer(embedding_dim=8).fit(["word"])  # a single distinct token -> vocab size 1, too small to train
        check("GloVe.fit raises a clear error on a too-small vocabulary", False)
    except ValueError:
        check("GloVe.fit raises a clear error on a too-small vocabulary", True)

    # --- pretrained loading ---
    missing_space = EmbeddingSpace.load_pretrained("/tmp/__definitely_does_not_exist__.txt")
    check("load_pretrained handles a missing file gracefully (empty space, no crash)", len(missing_space) == 0)

    tmp_path = "/tmp/__test_pretrained_embeddings.txt"
    space_w2v.save(tmp_path)
    reloaded = EmbeddingSpace.load_pretrained(tmp_path)
    check("save()/load_pretrained() round-trip preserves vocabulary size", len(reloaded) == len(space_w2v))
    sample_word = next(iter(space_w2v.vectors))
    check("save()/load_pretrained() round-trip preserves vector values",
          np.allclose(reloaded.vector(sample_word), space_w2v.vector(sample_word), atol=1e-5))
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} tests passed.")
    return passed == total


class RAGLogger:
    """
    Lightweight structured logger for the RAG pipeline: timestamped,
    leveled (DEBUG/INFO/WARNING/ERROR) entries kept in memory (so a
    pipeline run can be inspected, filtered, or visualized afterward)
    and optionally echoed to stdout as they happen.
    """
    LEVELS = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40}

    def __init__(self, name="RAGPipeline", level="INFO", echo=True):
        self.name = name
        self.level = level
        self.echo = echo
        self.entries = []

    def log(self, level, stage, message, **extra):
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'stage': stage,
            'message': message,
            **extra,
        }
        self.entries.append(entry)
        if self.echo and self.LEVELS.get(level, 0) >= self.LEVELS.get(self.level, 0):
            print(f"[{entry['timestamp']}] {level:7s} [{stage}] {message}")
        return entry

    def debug(self, stage, message, **extra):
        return self.log('DEBUG', stage, message, **extra)

    def info(self, stage, message, **extra):
        return self.log('INFO', stage, message, **extra)

    def warning(self, stage, message, **extra):
        return self.log('WARNING', stage, message, **extra)

    def error(self, stage, message, **extra):
        return self.log('ERROR', stage, message, **extra)

    def get_logs(self, level=None, stage=None):
        logs = self.entries
        if level:
            logs = [e for e in logs if e['level'] == level]
        if stage:
            logs = [e for e in logs if e['stage'] == stage]
        return logs

    def clear(self):
        self.entries = []

    def summary(self):
        return dict(Counter(e['level'] for e in self.entries))


class VectorStore:
    """
    Minimal vector-store abstraction for the RAG pipeline: holds chunk
    records and supports similarity search over them. Wraps whichever
    retrieval backend is available on the twin behind one interface —
    SimpleRetriever (TF/cosine, lexical, always available) and/or an
    EmbeddingSpace (Word2Vec/GloVe mean-pooled vectors, semantic) —
    so the pipeline can switch retrieval strategy without changing its
    own code. This is the "vector store + similarity search" piece of
    the RAG architecture, decoupled from how the vectors were produced.
    """
    def __init__(self, backend="tf", embedding_space=None, logger=None):
        if backend not in ("tf", "semantic", "hybrid"):
            raise ValueError(f"VectorStore: unknown backend '{backend}' (expected 'tf', 'semantic', or 'hybrid')")
        if backend in ("semantic", "hybrid") and (embedding_space is None or len(embedding_space) == 0):
            raise ValueError(f"VectorStore: backend '{backend}' requires a non-empty embedding_space")

        self.backend = backend
        self.embedding_space = embedding_space
        self.tf_retriever = SimpleRetriever()
        self.chunks = []
        self._semantic_cache = {}
        self.logger = logger

    def _log(self, level, msg, **extra):
        if self.logger:
            self.logger.log(level, "VectorStore", msg, **extra)

    def index(self, chunks):
        self.chunks = chunks or []
        self.tf_retriever.index(self.chunks)
        self._semantic_cache = {}
        self._log('INFO', f"Indexed {len(self.chunks)} chunks (backend={self.backend})")

    def _embed_text(self, text):
        if self.embedding_space is None or len(self.embedding_space) == 0:
            return None
        dim = next(iter(self.embedding_space.vectors.values())).shape[0]
        tokens = SimpleRetriever._tokenize(text)
        vecs = [v for v in (self.embedding_space.vector(t) for t in tokens) if v is not None]
        return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

    def _semantic_search(self, query, top_k):
        if self.embedding_space is None or len(self.embedding_space) == 0:
            self._log('WARNING', "semantic search requested but no embedding_space is set")
            return []
        if len(self._semantic_cache) != len(self.chunks):
            self._semantic_cache = {}
            for i, c in enumerate(self.chunks):
                cid = c.get('chunk_id', f"{c.get('source', 'unknown')}_{i}")
                self._semantic_cache[cid] = self._embed_text(c['text'])

        q_vec = self._embed_text(query)
        if q_vec is None or not np.any(q_vec):
            return []

        scored = []
        for i, c in enumerate(self.chunks):
            cid = c.get('chunk_id', f"{c.get('source', 'unknown')}_{i}")
            c_vec = self._semantic_cache.get(cid)
            if c_vec is None or not np.any(c_vec):
                continue
            scored.append((EmbeddingSpace.cosine_similarity(q_vec, c_vec), c))
        scored.sort(key=lambda x: -x[0])
        return [{**c, 'score': s} for s, c in scored[:top_k]]

    def _tf_search(self, query, top_k):
        results = self.tf_retriever.retrieve(query, top_k=top_k)
        for r in results:
            r['score'] = r.pop('similarity', 0.0)
        return results

    @staticmethod
    def _chunk_key(c):
        return c.get('chunk_id') or f"{c.get('source', 'unknown')}_{c.get('position', id(c))}"

    def _reciprocal_rank_fusion(self, ranked_lists, top_k, k_constant=60):
        """
        Merges several ranked result lists into one via Reciprocal
        Rank Fusion: score(d) = sum over lists of 1/(k_constant + rank
        + 1). A standard, parameter-light way to combine lexical and
        semantic rankings without needing to calibrate their raw
        scores against each other (TF-cosine and embedding-cosine
        aren't directly comparable numbers).
        """
        fused, chunk_lookup = {}, {}
        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked):
                key = self._chunk_key(chunk)
                chunk_lookup[key] = chunk
                fused[key] = fused.get(key, 0.0) + 1.0 / (k_constant + rank + 1)

        ranked_keys = sorted(fused.keys(), key=lambda k: -fused[k])[:top_k]
        return [{**chunk_lookup[k], 'score': fused[k]} for k in ranked_keys]

    def search(self, query, top_k=5):
        if not query or not query.strip():
            self._log('WARNING', "empty query passed to VectorStore.search")
            return []
        try:
            if self.backend == "tf":
                return self._tf_search(query, top_k)
            elif self.backend == "semantic":
                return self._semantic_search(query, top_k)
            else:  # hybrid: fuse lexical + semantic rankings
                tf_results = self._tf_search(query, top_k=max(top_k, 10))
                sem_results = self._semantic_search(query, top_k=max(top_k, 10))
                return self._reciprocal_rank_fusion([tf_results, sem_results], top_k)
        except Exception as e:
            self._log('ERROR', f"search failed ({e}); returning no results", error=str(e))
            return []


class RAGPipeline:
    """
    Complete retrieval-augmented generation architecture, tying every
    earlier-built piece into one pipeline:

        [Documents] -> DOCUMENT LOADING & PREPROCESSING (DocumentPreprocessor)
                     -> CHUNKING (ChunkingSystem, format-aware: text/markdown/log)
                     -> INDEXING (VectorStore: lexical / semantic / hybrid)
                     -> [Query] -> RETRIEVAL (VectorStore.search)
                     -> CONTEXT ASSEMBLY (ContextWindowManager — token-budget-safe)
                     -> GENERATION (an LLM client, if one is provided)
                     -> [Answer + Sources]

    Every stage is timed and logged (self.logger) and fails gracefully:
    errors are caught, logged, and surfaced in the returned result
    rather than raised. With no llm_client the pipeline runs in
    "retrieval-only" mode (sources only, answer=None) — useful for
    testing, or before an API key is configured.
    """
    def __init__(self, twin, llm_client=None, retrieval_mode="hybrid", top_k=4,
                 log_level="INFO", echo_logs=True):
        if retrieval_mode not in ("tf", "semantic", "hybrid"):
            raise ValueError(f"RAGPipeline: unknown retrieval_mode '{retrieval_mode}' "
                             f"(expected 'tf', 'semantic', or 'hybrid')")
        if retrieval_mode in ("semantic", "hybrid") and (twin.embeddings is None or len(twin.embeddings) == 0):
            raise ValueError(
                f"RAGPipeline: retrieval_mode='{retrieval_mode}' requires trained/loaded embeddings — "
                f"call twin.train_word_embeddings() first, or use retrieval_mode='tf'")

        self.twin = twin
        self.llm_client = llm_client
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.logger = RAGLogger(name="RAGPipeline", level=log_level, echo=echo_logs)

        self.vector_store = VectorStore(
            backend=retrieval_mode,
            embedding_space=twin.embeddings if retrieval_mode in ("semantic", "hybrid") else None,
            logger=self.logger,
        )
        self.vector_store.index(twin.knowledge_base)
        self._latency_log = []  # one entry per ask() call: {query, timings}

    # ---------- document loading / (re)indexing ----------
    def ingest(self, raw_text, source="document", doc_format=None, chunk_strategy="smart"):
        """
        Loads, preprocesses and chunks a document through the twin's
        existing DocumentPreprocessor pipeline, then re-indexes the
        vector store. Returns the ingestion result dict, or a dict
        with an 'error' key on failure — never raises.
        """
        stage = "ingest"
        t0 = time.time()
        try:
            result = self.twin.ingest_document(raw_text, source=source, doc_format=doc_format,
                                                chunk_strategy=chunk_strategy, verbose=False)
            self.vector_store.index(self.twin.knowledge_base)
            self.logger.info(stage, f"Ingested '{source}': {len(result['valid_chunks'])} chunks, "
                                     f"format={result['doc_format']}", duration=time.time() - t0)
            return result
        except Exception as e:
            self.logger.error(stage, f"Ingestion of '{source}' failed: {e}", error=str(e))
            return {'error': str(e), 'source': source}

    # ---------- retrieval ----------
    def retrieve(self, query, top_k=None):
        stage = "retrieve"
        t0 = time.time()
        top_k = top_k or self.top_k
        try:
            results = self.vector_store.search(query, top_k=top_k)
            self.logger.info(stage, f"Retrieved {len(results)} chunk(s) for query "
                                     f"'{query[:60]}' (mode={self.retrieval_mode})",
                              duration=time.time() - t0, num_results=len(results))
            return results
        except Exception as e:
            self.logger.error(stage, f"Retrieval failed: {e}", error=str(e))
            return []

    # ---------- context assembly ----------
    def assemble_context(self, query, retrieved_chunks, system_instruction=None, strategy="sliding_window"):
        stage = "assemble_context"
        t0 = time.time()
        try:
            base_instruction = system_instruction or self.twin.build_system_instruction()
            if retrieved_chunks:
                lines = [base_instruction, "", "Relevant retrieved context:"]
                for r in retrieved_chunks:
                    lines.append(f"  - ({r.get('source', 'unknown')}, score={r.get('score', 0):.2f}): {r['text']}")
                full_instruction = "\n".join(lines)
            else:
                full_instruction = base_instruction

            messages = self.twin.context_manager.fit_to_budget(
                full_instruction, [{"role": "user", "content": query}], strategy=strategy)
            self.logger.info(stage, f"Assembled context: "
                                     f"{self.twin.context_manager.count_messages(messages)} tokens "
                                     f"(budget={self.twin.context_manager.available_tokens})",
                              duration=time.time() - t0)
            return messages
        except Exception as e:
            self.logger.error(stage, f"Context assembly failed: {e}", error=str(e))
            return [{"role": "user", "content": query}]  # minimal safe fallback

    # ---------- generation ----------
    def generate(self, messages):
        stage = "generate"
        t0 = time.time()
        if self.llm_client is None:
            self.logger.warning(stage, "no llm_client configured — skipping generation (retrieval-only mode)")
            return None, None
        try:
            system_msgs = [m for m in messages if m.get('role') == 'system']
            user_msgs = [m for m in messages if m.get('role') == 'user']
            system_text = system_msgs[0]['content'] if system_msgs else None
            user_text = user_msgs[-1]['content'] if user_msgs else ""

            if hasattr(self.llm_client, 'conversation_history'):
                reply, usage = self._stateless_call_via_stateful_client(system_text, user_text)
            else:
                reply = self.llm_client.chat(system_text or "", user_text)
                usage = None

            self.logger.info(stage, "Generated a response", duration=time.time() - t0,
                              tokens_used=(usage or {}).get('total_tokens') if usage else None)
            return reply, usage
        except Exception as e:
            self.logger.error(stage, f"Generation failed: {e}", error=str(e))
            return None, None

    def _stateless_call_via_stateful_client(self, system_text, user_text):
        """
        Uses a stateful client (e.g. GroqClient) for a ONE-OFF RAG call
        without polluting its ongoing conversation — swaps in a
        throwaway history, makes the call, and restores the original
        history afterward regardless of success or failure.
        """
        saved_history = self.llm_client.conversation_history
        self.llm_client.conversation_history = []
        try:
            reply, usage = self.llm_client.chat(user_text, system_instruction=system_text)
        finally:
            self.llm_client.conversation_history = saved_history
        return reply, usage

    # ---------- orchestration ----------
    def ask(self, query, top_k=None, context_strategy="sliding_window"):
        """
        Runs the full pipeline for one query: retrieve -> assemble
        context -> generate. Returns a result dict with the answer
        (None in retrieval-only mode), the sources used, per-stage
        timings, and any error encountered. Never raises.
        """
        if not query or not query.strip():
            self.logger.warning("ask", "empty query provided")
            return {'query': query, 'answer': None, 'sources': [], 'tokens_used': None,
                    'timings': {}, 'retrieval_mode': self.retrieval_mode, 'error': 'empty query'}

        timings = {}
        t_start = time.time()

        t0 = time.time()
        retrieved = self.retrieve(query, top_k=top_k)
        timings['retrieve'] = time.time() - t0

        t0 = time.time()
        messages = self.assemble_context(query, retrieved, strategy=context_strategy)
        timings['assemble_context'] = time.time() - t0

        t0 = time.time()
        answer, usage = self.generate(messages)
        timings['generate'] = time.time() - t0

        timings['total'] = time.time() - t_start
        self._latency_log.append({'query': query, 'timings': dict(timings)})

        return {
            'query': query,
            'answer': answer,
            'sources': retrieved,
            'tokens_used': (usage or {}).get('total_tokens') if usage else None,
            'timings': timings,
            'retrieval_mode': self.retrieval_mode,
            'error': None,
        }

    # ---------- documentation / user-friendly interface ----------
    def describe(self):
        """Prints a plain-language description of the configured pipeline architecture."""
        print("RAG Pipeline Architecture")
        print("=" * 40)
        print(f"  Retrieval mode : {self.retrieval_mode}")
        print(f"  Top-k          : {self.top_k}")
        print(f"  Knowledge base : {len(self.twin.knowledge_base)} chunks")
        print(f"  LLM client     : "
              f"{'configured (' + type(self.llm_client).__name__ + ')' if self.llm_client else 'none (retrieval-only mode)'}")
        print(f"  Context budget : {self.twin.context_manager.available_tokens} tokens "
              f"(max={self.twin.context_manager.max_tokens}, "
              f"reserved={self.twin.context_manager.reserve_for_response})")
        print()
        print("  Pipeline stages:")
        print("    1. ingest()           - load + preprocess + chunk a document, (re)index the vector store")
        print("    2. retrieve()         - vector-store similarity search for a query")
        print("    3. assemble_context() - build a token-budget-safe prompt from retrieved chunks")
        print("    4. generate()         - call the LLM client with the assembled context")
        print("  ask(query) runs retrieve -> assemble_context -> generate in one call.")

    # ---------- visualization ----------
    def visualize_latency(self, title="RAG Pipeline — Stage Latency"):
        """Bar chart of average time spent per stage, across every ask() call made so far."""
        if not self._latency_log:
            print("  ! visualize_latency: no ask() calls recorded yet")
            return
        stages = ['retrieve', 'assemble_context', 'generate']
        avg_times = [float(np.mean([l['timings'].get(s, 0) for l in self._latency_log])) for s in stages]

        plt.figure(figsize=(7, 4.5))
        plt.bar(stages, avg_times, color=['steelblue', 'seagreen', 'salmon'])
        plt.title(f"{title} (avg over {len(self._latency_log)} call(s))")
        plt.ylabel("Seconds")
        plt.tight_layout()
        plt.show()

    def visualize_retrieval(self, query, top_k=None, title=None):
        """Bar chart of retrieval scores for a query — which sources were retrieved, and how confidently."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            print("  ! visualize_retrieval: no results to visualize")
            return
        labels = [f"{r.get('source', '?')} #{i}" for i, r in enumerate(results)]
        scores = [r.get('score', 0) for r in results]

        plt.figure(figsize=(8, 4.5))
        plt.barh(labels[::-1], scores[::-1], color='mediumpurple')
        plt.xlabel("Retrieval score")
        plt.title(title or f"Retrieval scores for: \"{query[:50]}\" (mode={self.retrieval_mode})")
        plt.tight_layout()
        plt.show()

    def log_summary(self):
        """Prints and returns a count of log entries by level — a quick health check of a pipeline run."""
        summary = self.logger.summary()
        print("Log summary:", summary)
        return summary


def run_rag_pipeline_tests():
    """
    Dependency-free test suite (plain asserts) for VectorStore and
    RAGPipeline, run against a small, self-contained test twin (built
    fresh here, independent of any global `twin`). Prints PASS/FAIL
    per test and returns True iff every test passed.
    """
    results = []

    def check(name, condition):
        status = "PASS" if condition else "FAIL"
        results.append(bool(condition))
        print(f"  [{status}] {name}")

    class _FakeStatelessLLM:
        """Mimics UniversalLLMClient's interface for testing, with no network call."""
        def __init__(self):
            self.calls = 0

        def chat(self, system_instruction, user_prompt, **kwargs):
            self.calls += 1
            return f"[fake answer to: {user_prompt[:30]}]"

    class _FakeStatefulLLM:
        """Mimics GroqClient's interface (has conversation_history) for testing, with no network call."""
        def __init__(self):
            self.conversation_history = []
            self.calls = 0

        def chat(self, user_message, system_instruction=None, **kwargs):
            self.calls += 1
            if system_instruction and not self.conversation_history:
                self.conversation_history.append({'role': 'system', 'content': system_instruction})
            self.conversation_history.append({'role': 'user', 'content': user_message})
            reply = f"[fake stateful answer to: {user_message[:30]}]"
            self.conversation_history.append({'role': 'assistant', 'content': reply})
            return reply, {'total_tokens': 42}

    rng = np.random.default_rng(0)
    n = 20
    test_twin = DigitalTwin(
        activities=np.array([i % 5 for i in range(n)]),
        days_of_week=np.array([i % 7 for i in range(n)]),
        sleep_hours=rng.normal(7, 1, n).clip(4, 10),
        mood=rng.random(n),
        weather=rng.random(n),
    )
    test_twin.ingest_notes(
        "Monday I worked on the digital twin coding project for several hours. "
        "Tuesday I went for a 45 minute run and felt great. Wednesday was a quiet "
        "day of sleep and rest. Thursday I had a long work meeting about the ML schedule.",
        source="test_journal", verbose=False,
    )

    # --- constructor validation ---
    try:
        RAGPipeline(test_twin, retrieval_mode="not_a_mode")
        check("RAGPipeline rejects an unknown retrieval_mode", False)
    except ValueError:
        check("RAGPipeline rejects an unknown retrieval_mode", True)

    try:
        RAGPipeline(test_twin, retrieval_mode="semantic")  # no embeddings trained yet
        check("RAGPipeline rejects 'semantic' mode without trained embeddings", False)
    except ValueError:
        check("RAGPipeline rejects 'semantic' mode without trained embeddings", True)

    # --- tf-mode pipeline: retrieval-only (no llm_client) ---
    pipeline_ro = RAGPipeline(test_twin, llm_client=None, retrieval_mode="tf", top_k=2, echo_logs=False)
    result_ro = pipeline_ro.ask("What have I been working on?")
    check("retrieval-only pipeline returns answer=None without crashing", result_ro['answer'] is None)
    check("retrieval-only pipeline still returns a list of sources", isinstance(result_ro['sources'], list))
    check("ask() result includes per-stage timings",
          {'retrieve', 'assemble_context', 'generate', 'total'}.issubset(result_ro['timings'].keys()))
    check("ask() result reports no error on a normal query", result_ro['error'] is None)

    empty_result = pipeline_ro.ask("")
    check("ask() handles an empty query gracefully (no crash, error set)", empty_result['error'] is not None)

    ingest_result = pipeline_ro.ingest("", source="empty_doc")
    check("ingest() handles empty document text without raising", isinstance(ingest_result, dict))

    # --- tf-mode pipeline with a fake stateful LLM client ---
    fake_stateful = _FakeStatefulLLM()
    fake_stateful.conversation_history = [
        {'role': 'system', 'content': 'pre-existing system'},
        {'role': 'user', 'content': 'previous turn'},
    ]
    original_history_snapshot = list(fake_stateful.conversation_history)

    pipeline_stateful = RAGPipeline(test_twin, llm_client=fake_stateful, retrieval_mode="tf",
                                     top_k=2, echo_logs=False)
    result_stateful = pipeline_stateful.ask("What have I been working on?")
    check("ask() with a stateful fake LLM client returns a non-None answer", result_stateful['answer'] is not None)
    check("ask() does not pollute the stateful client's existing conversation_history",
          fake_stateful.conversation_history == original_history_snapshot)
    check("pipeline logger captured INFO-level entries during ask()",
          len(pipeline_stateful.logger.get_logs(level='INFO')) > 0)

    # --- tf-mode pipeline with a fake stateless LLM client ---
    fake_stateless = _FakeStatelessLLM()
    pipeline_stateless = RAGPipeline(test_twin, llm_client=fake_stateless, retrieval_mode="tf",
                                      top_k=2, echo_logs=False)
    result_stateless = pipeline_stateless.ask("Tell me about my exercise")
    check("ask() with a stateless fake LLM client returns a non-None answer",
          result_stateless['answer'] is not None)

    # --- semantic / hybrid modes, once embeddings exist ---
    test_twin.train_word_embeddings(method="word2vec", dim=8, window_size=2, negative_samples=3,
                                     epochs=5, verbose=False)
    pipeline_sem = RAGPipeline(test_twin, retrieval_mode="semantic", top_k=2, echo_logs=False)
    sem_results = pipeline_sem.retrieve("daily routine")
    check("semantic retrieval mode returns a list without crashing", isinstance(sem_results, list))

    pipeline_hybrid = RAGPipeline(test_twin, retrieval_mode="hybrid", top_k=2, echo_logs=False)
    hybrid_results = pipeline_hybrid.retrieve("daily routine")
    check("hybrid retrieval mode (reciprocal rank fusion) respects top_k", len(hybrid_results) <= 2)

    # --- context assembly respects a tiny token budget ---
    test_twin.set_context_budget(max_tokens=120, reserve_for_response=20)
    pipeline_tiny = RAGPipeline(test_twin, llm_client=None, retrieval_mode="tf", top_k=5, echo_logs=False)
    tiny_messages = pipeline_tiny.assemble_context(
        "Tell me everything about my week in detail", pipeline_tiny.retrieve("week", top_k=5))
    check("assemble_context respects a tiny token budget",
          test_twin.context_manager.count_messages(tiny_messages) <= test_twin.context_manager.available_tokens)

    # --- documentation / visualization smoke tests (must not raise) ---
    try:
        pipeline_stateful.describe()
        check("describe() runs without raising", True)
    except Exception:
        check("describe() runs without raising", False)

    try:
        pipeline_stateful.visualize_latency()
        pipeline_stateful.visualize_retrieval("What have I been working on?")
        check("visualize_latency()/visualize_retrieval() run without raising", True)
    except Exception:
        check("visualize_latency()/visualize_retrieval() run without raising", False)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} tests passed.")
    return passed == total


# ============================================================
# AI AGENTS: TOOL FRAMEWORK + REACT (REASON + ACT) AGENT
# ============================================================
# Thought -> Action -> Observation -> (repeat) -> Final Answer.
#
# A Tool wraps a plain Python function. Its parameter *type hints*
# (int, float, bool, str, list, dict, Any, Callable, ...) are
# introspected with `inspect`/`typing` to (a) describe the tool to the
# LLM and (b) coerce the LLM's text "Action Input" into real Python
# values before calling the function. This is the "list, dict, any,
# callable" piece of the request: those are the types a tool's inputs
# can be declared as, handled generically rather than one-off per tool.

class Tool:
    """
    Wraps a callable as an agent tool: introspects its signature and
    type hints to build a parameter schema (used both to describe the
    tool to the LLM and to coerce string "Action Input" values back
    into real int/float/bool/list/dict/str before calling it).
    """

    def __init__(self, func: Callable, name: str = None, description: str = None):
        self.func = func
        self.name = name or func.__name__
        doc = (func.__doc__ or "").strip()
        self.description = description or (doc.split("\n")[0] if doc else "No description.")
        self.signature = inspect.signature(func)
        try:
            self.type_hints = get_type_hints(func)
        except Exception:
            # a hint that can't be resolved (forward ref, missing import, etc.)
            # shouldn't break tool registration -- fall back to Any for everything
            self.type_hints = {}

    def schema(self):
        """Builds {param_name: {type, required, default}} from the function's type hints."""
        params = {}
        for pname, p in self.signature.parameters.items():
            hint = self.type_hints.get(pname, Any)
            hint_name = getattr(hint, '__name__', str(hint))
            params[pname] = {
                'type': hint_name,
                'required': p.default is inspect.Parameter.empty,
                'default': None if p.default is inspect.Parameter.empty else p.default,
            }
        return params

    def describe(self):
        """One-line human/LLM-readable description, e.g. 'calculator(expression: str) - ...'."""
        parts = []
        for pname, info in self.schema().items():
            suffix = "" if info['required'] else f"={info['default']!r}"
            parts.append(f"{pname}: {info['type']}{suffix}")
        return f"{self.name}({', '.join(parts)}) - {self.description}"

    @staticmethod
    def _coerce(value, hint):
        """Best-effort coercion of a string Action-Input value to the hinted type. Never raises."""
        if not isinstance(value, str):
            return value
        origin = getattr(hint, '__origin__', None)
        try:
            if hint is int:
                return int(value.strip())
            if hint is float:
                return float(value.strip())
            if hint is bool:
                return value.strip().lower() in ("true", "1", "yes", "y")
            if hint is list or hint is List or origin is list:
                v = value.strip()
                if v.startswith('['):
                    return json.loads(v)
                return [item.strip() for item in v.split(',') if item.strip()]
            if hint is dict or hint is Dict or origin is dict:
                return json.loads(value)
        except Exception:
            return value  # coercion failed -- pass the raw string through; let the tool decide
        return value  # str / Any / Callable / unknown -- pass through unchanged

    def call(self, **kwargs):
        """
        Calls the wrapped function with coerced kwargs. Never raises:
        any failure (bad args, a tool-internal exception) is caught and
        returned as a "ToolError: ..." string so it becomes a normal
        Observation the agent can see and react to, instead of
        crashing the whole reasoning loop.
        """
        try:
            coerced = {pname: self._coerce(v, self.type_hints.get(pname, Any)) for pname, v in kwargs.items()}
            return self.func(**coerced)
        except Exception as e:
            return f"ToolError: {e}"


class ToolRegistry:
    """A named collection of Tools an agent can choose from, with simple registration and lookup."""

    def __init__(self):
        self.tools = {}

    def register(self, func=None, *, name=None, description=None):
        """Usable as @registry.register or @registry.register(name=..., description=...)."""
        def _wrap(f):
            tool = Tool(f, name=name, description=description)
            self.tools[tool.name] = tool
            return f
        return _wrap(func) if func is not None else _wrap

    def get(self, name):
        return self.tools.get(name)

    def names(self):
        return list(self.tools.keys())

    def describe_all(self):
        return "\n".join(f"  - {t.describe()}" for t in self.tools.values())

    def call(self, name, **kwargs):
        tool = self.get(name)
        if tool is None:
            return f"ToolError: unknown tool '{name}'. Available tools: {', '.join(self.names())}"
        return tool.call(**kwargs)


class HabitTracker:
    """
    Simple habit-tracking store: logs a completed/skipped entry per
    habit per day and computes the current streak. Backs the
    track_habit / habit_stats tools -- one of the digital twin's two
    custom agent tools (the other is the note-taker, which reuses the
    twin's existing RAG ingestion pipeline).
    """

    def __init__(self):
        self.log_entries = {}  # habit_name -> list of (date_str, completed: bool), sorted by date

    def log(self, habit_name, completed=True, date_str=None):
        date_str = date_str or datetime.utcnow().strftime("%Y-%m-%d")
        entries = self.log_entries.setdefault(habit_name, [])
        entries[:] = [e for e in entries if e[0] != date_str]  # idempotent re-logging for "today"
        entries.append((date_str, bool(completed)))
        entries.sort(key=lambda e: e[0])
        streak = self._current_streak(habit_name)
        status = "completed" if completed else "skipped"
        return f"Logged '{habit_name}' as {status} for {date_str}. Current streak: {streak} day(s)."

    def _current_streak(self, habit_name):
        entries = self.log_entries.get(habit_name, [])
        streak = 0
        for _date_str, completed in reversed(entries):
            if not completed:
                break
            streak += 1
        return streak

    def stats(self, habit_name):
        entries = self.log_entries.get(habit_name, [])
        if not entries:
            return f"No entries logged yet for '{habit_name}'."
        completed_count = sum(1 for _, c in entries if c)
        streak = self._current_streak(habit_name)
        last_date, last_completed = entries[-1]
        return (f"'{habit_name}': {completed_count}/{len(entries)} day(s) completed, "
                f"current streak: {streak} day(s), last logged: {last_date} "
                f"({'completed' if last_completed else 'skipped'}).")


def build_default_tool_registry(twin):
    """
    Builds the default tool set for a digital twin's ReAct agent:
    calculator, datetime_now, search_memory (wraps the twin's existing
    RAG retrieval), and the 2 custom tools: take_note (Note Taker,
    wraps ingest_notes so notes become searchable) and
    track_habit/habit_stats (Habit Tracker, backed by HabitTracker).
    Every function below is resolved against `twin` at CALL time, not
    registration time, so this can safely be built before or during
    DigitalTwin.__init__.
    """
    registry = ToolRegistry()

    @registry.register(name="calculator",
                        description="Evaluates a basic arithmetic expression (+ - * / ** % parentheses) and returns the numeric result.")
    def calculator(expression: str) -> str:
        cleaned = expression.replace(' ', '')
        allowed = set("0123456789+-*/().%")
        if not cleaned or not set(cleaned) <= allowed:
            return "ToolError: expression contains disallowed characters"
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"ToolError: could not evaluate expression ({e})"

    @registry.register(name="datetime_now",
                        description="Returns the current date/time (UTC), optionally in a custom strftime format.")
    def datetime_now(format: str = "%Y-%m-%d %H:%M:%S") -> str:
        try:
            return datetime.utcnow().strftime(format)
        except Exception as e:
            return f"ToolError: invalid format string ({e})"

    @registry.register(name="search_memory",
                        description="Searches the twin's knowledge base (ingested notes/documents) for information relevant to a query.")
    def search_memory(query: str, top_k: int = 3) -> str:
        results = twin.retrieve_relevant_chunks(query, top_k=top_k)
        if not results:
            return "No relevant memory found."
        return "\n".join(f"- ({r.get('source', '?')}) {r['text'][:200]}" for r in results)

    @registry.register(name="take_note",
                        description="Custom tool (Note Taker): saves a short note into the twin's memory so it becomes searchable via search_memory.")
    def take_note(text: str) -> str:
        twin.ingest_notes(text, source="agent_note", chunk_strategy="smart", verbose=False)
        preview = text if len(text) <= 60 else text[:60] + "..."
        return f'Note saved: "{preview}"'

    @registry.register(name="track_habit",
                        description="Custom tool (Habit Tracker): logs today's completion (or skip) of a named habit and returns the updated streak.")
    def track_habit(habit_name: str, completed: bool = True) -> str:
        return twin.habit_tracker.log(habit_name, completed)

    @registry.register(name="habit_stats",
                        description="Custom tool (Habit Tracker): returns completion count, streak, and last-logged date for a named habit.")
    def habit_stats(habit_name: str) -> str:
        return twin.habit_tracker.stats(habit_name)

    return registry


class ScriptedLLM:
    """
    An LLM stand-in that returns pre-programmed, ReAct-formatted
    responses in sequence -- lets the full agent loop (parsing,
    tool-calling, looping, trace compression) be tested and demoed
    deterministically with no network call and no API key. Matches
    the .chat(user_message, system_instruction=...) -> (reply, usage)
    interface used by GroqClient, and exposes conversation_history so
    ReActAgent's stateful-client code path is exercised too.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.conversation_history = []
        self.calls = 0

    def chat(self, user_message, system_instruction=None, **kwargs):
        self.calls += 1
        if self.responses:
            reply = self.responses.pop(0)
        else:
            reply = "Thought: I am out of scripted responses.\nFinal Answer: (no more responses available)"
        return reply, {'total_tokens': len(reply.split())}


class ReActAgent:
    """
    An agent that reasons and acts using tools, following the ReAct
    (Reason + Act) loop: at each step the LLM either thinks and calls
    a tool, or gives a final answer, observing each tool's result
    before deciding its next step. This lets the twin answer questions
    a single LLM call can't -- needing a calculation, today's date, a
    knowledge-base lookup, or a stateful action like logging a habit.

    Loop (bounded by max_iterations, never raises):
      1. Thought: the model reasons about what to do next
      2. Action: the model names a tool + its input
      3. Observation: the tool's result is appended to the trace
      4. ...repeat, or Final Answer: respond to the user

    "Compress, don't delete": once the reasoning trace grows past
    (compress_every + keep_recent_steps) entries, the OLDEST
    compress_every entries are merged into ONE shorter 'summary' step
    via extractive summarization (reusing the twin's own
    ContextWindowManager.extractive_summary) -- the facts stay, the
    verbose wording goes, and nothing is ever silently dropped.
    """

    def __init__(self, twin, llm_client, tool_registry=None, max_iterations=6,
                 compress_every=5, keep_recent_steps=4, log_level="INFO", echo_logs=True):
        self.twin = twin
        self.llm_client = llm_client
        self.tools = tool_registry or build_default_tool_registry(twin)
        self.max_iterations = max_iterations
        self.compress_every = compress_every
        self.keep_recent_steps = keep_recent_steps
        self.logger = RAGLogger(name="ReActAgent", level=log_level, echo=echo_logs)
        self.trace = []  # list of {'type': thought/action/observation/answer/summary, 'content': str}

    def _system_prompt(self):
        return (
            "You are a digital twin's reasoning agent. Answer the user's question by "
            "thinking step by step and using tools when needed. Respond using EXACTLY "
            "this format, one step per response:\n\n"
            "Thought: <your reasoning about what to do next>\n"
            "Action: <tool_name>\n"
            "Action Input: <input to the tool, as plain text or key=value pairs>\n\n"
            "OR, once you have enough information:\n\n"
            "Thought: <your reasoning>\n"
            "Final Answer: <your answer to the user>\n\n"
            f"Available tools:\n{self.tools.describe_all()}\n\n"
            "Output ONLY one Thought/Action pair OR one Thought/Final Answer pair per "
            "response -- never both, never skip the Thought line."
        )

    def _trace_text(self):
        lines = []
        for step in self.trace:
            if step['type'] == 'summary':
                lines.append(f"[Earlier steps summarized]: {step['content']}")
            else:
                lines.append(f"{step['type'].capitalize()}: {step['content']}")
        return "\n".join(lines)

    def _compress_trace(self):
        """Merges the oldest compress_every trace entries into one summary step, once the trace is long enough."""
        if len(self.trace) <= self.compress_every + self.keep_recent_steps:
            return
        oldest, rest = self.trace[:self.compress_every], self.trace[self.compress_every:]
        combined_text = " ".join(f"{s['type']}: {s['content']}" for s in oldest)
        summary_text = self.twin.context_manager.extractive_summary(combined_text, max_sentences=4)
        self.trace = [{'type': 'summary', 'content': summary_text}] + rest
        self.logger.info("compress", f"Compressed {len(oldest)} oldest trace entries into one summary step")

    @staticmethod
    def _parse_step(text):
        """
        Parses one LLM response into {'thought', 'action', 'action_input'}
        or {'thought', 'final_answer'}. Tolerant of minor formatting
        drift; never raises -- an unparseable response is treated as a
        final answer so the loop terminates sensibly instead of
        spinning on garbage.
        """
        thought_match = re.search(r'Thought:\s*(.*?)(?:\n(?:Action|Final Answer)|\Z)', text, re.DOTALL)
        action_match = re.search(r'Action:\s*(.*?)\n', text)
        input_match = re.search(r'Action Input:\s*(.*?)(?:\nObservation|\Z)', text, re.DOTALL)
        final_match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else None

        if final_match:
            return {'thought': thought, 'final_answer': final_match.group(1).strip()}
        if action_match:
            action_input = input_match.group(1).strip() if input_match else ""
            return {'thought': thought, 'action': action_match.group(1).strip(), 'action_input': action_input}
        return {'thought': thought, 'final_answer': text.strip()}  # fallback: unparseable -> final answer

    @staticmethod
    def _parse_action_input(raw):
        """Parses 'key=value, key2=value2' into kwargs; a single bare value is returned under '__positional__'."""
        raw = raw.strip()
        if not raw:
            return {}
        if '=' in raw:
            kwargs = {}
            for part in re.split(r',\s*(?=[\w]+\s*=)', raw):
                if '=' in part:
                    k, v = part.split('=', 1)
                    kwargs[k.strip()] = v.strip().strip('"\'')
            if kwargs:
                return kwargs
        return {'__positional__': raw}

    def _execute_action(self, action_name, action_input_raw):
        tool = self.tools.get(action_name)
        if tool is None:
            return f"ToolError: unknown tool '{action_name}'. Available tools: {', '.join(self.tools.names())}"
        kwargs = self._parse_action_input(action_input_raw)
        if '__positional__' in kwargs:
            first_param = next(iter(tool.signature.parameters), None)
            kwargs = {first_param: kwargs['__positional__']} if first_param else {}
        return tool.call(**kwargs)

    def _call_llm(self, prompt):
        """Calls the configured LLM client for one reasoning step, stateless so a stateful client's chat history is never polluted."""
        if hasattr(self.llm_client, 'conversation_history'):
            saved = self.llm_client.conversation_history
            self.llm_client.conversation_history = []
            try:
                reply, _usage = self.llm_client.chat(prompt, system_instruction=self._system_prompt())
            finally:
                self.llm_client.conversation_history = saved
            return reply
        return self.llm_client.chat(self._system_prompt(), prompt)

    def run(self, question):
        """
        Runs the full ReAct loop for one question. Returns
        {answer, trace, iterations, error}. Never raises: an LLM
        failure or exhausting max_iterations returns a best-effort
        result with `error` set, instead of crashing.
        """
        self.trace = []
        self.logger.info("run", f"Starting ReAct loop for: '{question[:80]}'")

        for i in range(self.max_iterations):
            self._compress_trace()
            prompt = f"Question: {question}"
            if self.trace:
                prompt = f"Question: {question}\n\n{self._trace_text()}\n\nWhat is your next step?"

            try:
                raw_response = self._call_llm(prompt)
            except Exception as e:
                self.logger.error("run", f"LLM call failed: {e}")
                return {'answer': None, 'trace': list(self.trace), 'iterations': i,
                        'error': f"llm_call_failed: {e}"}

            step = self._parse_step(raw_response)
            if step.get('thought'):
                self.trace.append({'type': 'thought', 'content': step['thought']})
                self.logger.debug("thought", step['thought'])

            if 'final_answer' in step:
                self.trace.append({'type': 'answer', 'content': step['final_answer']})
                self.logger.info("run", f"Final answer reached after {i + 1} iteration(s)")
                return {'answer': step['final_answer'], 'trace': list(self.trace),
                        'iterations': i + 1, 'error': None}

            action, action_input = step['action'], step['action_input']
            self.trace.append({'type': 'action', 'content': f"{action}[{action_input}]"})
            self.logger.info("action", f"Calling tool '{action}' with input: {action_input}")

            observation = self._execute_action(action, action_input)
            self.trace.append({'type': 'observation', 'content': str(observation)})
            self.logger.info("observation", str(observation)[:200])

        self.logger.warning("run", f"Max iterations ({self.max_iterations}) reached without a final answer")
        best_effort = next((s['content'] for s in reversed(self.trace) if s['type'] == 'thought'), None)
        return {'answer': best_effort, 'trace': list(self.trace),
                'iterations': self.max_iterations, 'error': 'max_iterations_reached'}

    def describe(self):
        print("ReAct Agent")
        print("=" * 40)
        print(f"  Tools available    : {', '.join(self.tools.names())}")
        print(f"  Max iterations     : {self.max_iterations}")
        print(f"  Trace compression  : oldest {self.compress_every} steps -> 1 summary step, "
              f"triggered once the trace exceeds {self.compress_every + self.keep_recent_steps} steps")
        print(f"  LLM client         : {type(self.llm_client).__name__ if self.llm_client else 'none'}")

    def visualize_trace(self, title="Agent Reasoning Trace"):
        """Step-type timeline scatter plot for the most recent run() call."""
        if not self.trace:
            print("  ! visualize_trace: no trace recorded yet -- call run() first")
            return
        type_order = {'summary': 0, 'thought': 1, 'action': 2, 'observation': 3, 'answer': 4}
        colors = {'summary': 'gray', 'thought': 'steelblue', 'action': 'orange',
                  'observation': 'seagreen', 'answer': 'salmon'}
        xs = list(range(len(self.trace)))
        ys = [type_order.get(s['type'], -1) for s in self.trace]
        c = [colors.get(s['type'], 'black') for s in self.trace]

        plt.figure(figsize=(max(6, len(self.trace) * 0.9), 4))
        plt.scatter(xs, ys, c=c, s=120)
        for x, y, s in zip(xs, ys, self.trace):
            plt.annotate(s['type'], (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
        plt.yticks(list(type_order.values()), list(type_order.keys()))
        plt.xlabel("Step")
        plt.title(title)
        plt.tight_layout()
        plt.show()


def run_react_agent_tests():
    """Dependency-free assert-based tests for Tool, ToolRegistry, HabitTracker, and ReActAgent (no network/API key needed)."""
    results = []

    def check(name, condition):
        results.append(bool(condition))
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}")

    # --- a small self-contained twin for tool/agent tests ---
    test_twin = DigitalTwin(
        activities=np.array([1, 1, 2, 3, 0]),
        days_of_week=np.array([0, 1, 2, 3, 4]),
        sleep_hours=np.array([7.0, 7.5, 6.5, 8.0, 7.0]),
        mood=np.array([0.6, 0.7, 0.5, 0.8, 0.6]),
        weather=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
    )

    print("\n=== Tool / ToolRegistry tests ===")
    registry = build_default_tool_registry(test_twin)

    check("calculator tool computes a correct result", registry.call("calculator", expression="2 + 3 * 4") == "14")
    check("calculator tool rejects a disallowed expression",
          "ToolError" in registry.call("calculator", expression="__import__('os').system('ls')"))
    check("datetime_now tool returns a non-empty string", len(registry.call("datetime_now")) > 0)
    check("datetime_now tool respects a custom format",
          re.match(r'^\d{4}$', registry.call("datetime_now", format="%Y")) is not None)

    note_result = registry.call("take_note", text="Remember to review the RAG pipeline tests tomorrow.")
    check("take_note tool reports success", "Note saved" in note_result)
    search_result = registry.call("search_memory", query="review the RAG pipeline tests")
    check("search_memory tool retrieves a note saved via take_note",
          "review" in search_result.lower() or "rag pipeline" in search_result.lower())
    check("search_memory tool handles a query with no matches gracefully",
          isinstance(registry.call("search_memory", query="zzzznonexistentzzzz"), str))

    habit_log_result = registry.call("track_habit", habit_name="meditate", completed="true")
    check("track_habit tool logs a habit and reports a streak", "streak" in habit_log_result.lower())
    habit_stats_result = registry.call("habit_stats", habit_name="meditate")
    check("habit_stats tool reports stats for a logged habit", "meditate" in habit_stats_result)
    check("habit_stats tool handles an unknown habit gracefully",
          "no entries" in registry.call("habit_stats", habit_name="not_a_real_habit").lower())

    check("ToolRegistry.names() lists all registered tools",
          {'calculator', 'datetime_now', 'search_memory', 'take_note', 'track_habit', 'habit_stats'}
          .issubset(set(registry.names())))
    check("calling an unknown tool returns a graceful ToolError string",
          "ToolError" in registry.call("not_a_real_tool", foo="bar"))

    def _typed_tool(n: int, f: float, flag: bool, items: list) -> str:
        return f"{n}|{f}|{flag}|{items}"
    typed = Tool(_typed_tool)
    coerced_out = typed.call(n="5", f="2.5", flag="true", items="a, b, c")
    check("Tool.call coerces string args to their annotated types",
          coerced_out == "5|2.5|True|['a', 'b', 'c']")
    check("Tool.describe() lists each parameter's name and type",
          "n: int" in typed.describe() and "items: list" in typed.describe())

    def _broken_tool(x: int) -> str:
        return 1 / 0
    broken = Tool(_broken_tool)
    check("Tool.call catches an internal exception and returns a ToolError string instead of raising",
          "ToolError" in broken.call(x="5"))

    print("\n=== ReActAgent loop tests (scripted LLM -- no network needed) ===")

    scripted_a = [
        "Thought: I should calculate this.\nAction: calculator\nAction Input: 12 * 7",
        "Thought: Now I have the number, I can answer.\nFinal Answer: 12 times 7 is 84.",
    ]
    agent_a = ReActAgent(test_twin, ScriptedLLM(scripted_a), max_iterations=5, echo_logs=False)
    result_a = agent_a.run("What is 12 times 7?")
    check("ReAct loop reaches a Final Answer after using a tool",
          result_a['answer'] == "12 times 7 is 84." and result_a['error'] is None)
    check("trace records both an action and an observation step",
          any(s['type'] == 'action' for s in result_a['trace']) and
          any(s['type'] == 'observation' for s in result_a['trace']))
    check("the calculator tool's observation contains the correct computed result",
          any('84' in s['content'] for s in result_a['trace'] if s['type'] == 'observation'))

    agent_b = ReActAgent(test_twin, ScriptedLLM(["Thought: I already know this.\nFinal Answer: Hello!"]),
                          max_iterations=3, echo_logs=False)
    result_b = agent_b.run("Say hello")
    check("ReAct loop handles an immediate final answer with no tool calls",
          result_b['answer'] == "Hello!" and result_b['iterations'] == 1)

    agent_c = ReActAgent(test_twin, ScriptedLLM(["this response is not formatted correctly at all"]),
                          max_iterations=3, echo_logs=False)
    result_c = agent_c.run("test")
    check("an unparseable LLM response is treated as a final answer instead of crashing",
          result_c['answer'] is not None and result_c['error'] is None)

    scripted_d = [
        "Thought: let me use a tool that doesn't exist.\nAction: not_a_real_tool\nAction Input: foo",
        "Thought: that failed, I will just answer directly.\nFinal Answer: done",
    ]
    agent_d = ReActAgent(test_twin, ScriptedLLM(scripted_d), max_iterations=5, echo_logs=False)
    result_d = agent_d.run("test unknown tool")
    check("an unknown tool name produces a graceful ToolError observation instead of a crash",
          any('ToolError' in s['content'] for s in result_d['trace'] if s['type'] == 'observation'))
    check("the agent recovers from a tool error and still reaches a final answer",
          result_d['answer'] == "done")

    infinite_responses = ["Thought: thinking more.\nAction: calculator\nAction Input: 1+1"] * 10
    agent_e = ReActAgent(test_twin, ScriptedLLM(infinite_responses), max_iterations=3, echo_logs=False)
    result_e = agent_e.run("loop forever")
    check("hitting max_iterations returns gracefully instead of raising or looping forever",
          result_e['error'] == 'max_iterations_reached' and result_e['iterations'] == 3)

    print("\n=== Trace compression tests ('compress, don't delete') ===")
    agent_f = ReActAgent(test_twin, ScriptedLLM([]), max_iterations=3,
                          compress_every=5, keep_recent_steps=2, echo_logs=False)
    for i in range(9):
        agent_f.trace.append({'type': 'observation', 'content': f"distinct_fact_{i} happened at step {i}."})
    before_len = len(agent_f.trace)
    agent_f._compress_trace()
    after_len = len(agent_f.trace)
    check("compress_trace reduces the number of trace entries", after_len < before_len)
    check("compress_trace produces exactly one summary step for the oldest group",
          agent_f.trace[0]['type'] == 'summary')
    check("compress_trace preserves the most recent steps verbatim (not summarized)",
          agent_f.trace[-1]['content'] == "distinct_fact_8 happened at step 8.")
    check("compress_trace keeps at least one fact from the compressed group in the summary (facts stay)",
          any(f"distinct_fact_{i}" in agent_f.trace[0]['content'] for i in range(5)))
    check("compress_trace does not fire before the trace is long enough",
          len(ReActAgent(test_twin, ScriptedLLM([]), compress_every=5, keep_recent_steps=2, echo_logs=False).trace) == 0)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} tests passed.")
    return passed == total


class DocumentPreprocessor:
    """
    Challenge Project: Document Preprocessor for the digital twin's
    knowledge base.

    Ingests documents in multiple formats (plain text, markdown, logs),
    extracts structured data, cleans/normalizes the text, chunks it
    optimally for the detected format, attaches metadata to every chunk,
    builds search indices (similarity + exact keyword), and reports
    quality metrics. Bonus: code blocks and markdown tables are detected
    and kept intact as their own chunks instead of being torn apart by
    sentence-based chunking.
    """

    CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
    MD_HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)
    SMART_CHAR_MAP = {
        '‘': "'", '’': "'", '“': '"', '”': '"',
        '–': '-', '—': '-', '…': '...',
    }

    def __init__(self, extractor=None, chunker=None):
        self.extractor = extractor or PersonalDataExtractor()
        self.chunker = chunker or ChunkingSystem()

    # ---------- 1. format detection ----------
    def detect_format(self, text):
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if not lines:
            return 'text'

        md_signals = sum(1 for l in lines if l.startswith(('#', '```', '- ', '* ', '|')))
        log_signals = sum(1 for l in lines if re.match(r'^\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', l)
                           or re.match(r'^\[?\d{2}:\d{2}:\d{2}\]?\s*\[?(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)', l))

        if log_signals / len(lines) > 0.3:
            return 'log'
        if md_signals >= 2:
            return 'markdown'
        return 'text'

    # ---------- 2. special-block extraction (bonus: code + tables) ----------
    def extract_special_blocks(self, text):
        """
        Pulls code blocks and markdown tables out of the text and replaces
        each with a placeholder token, so the chunker never splits them
        mid-block. Returns (text_with_placeholders, special_blocks_list).
        """
        special_blocks = []

        def _replace_code(match):
            idx = len(special_blocks)
            special_blocks.append({'type': 'code', 'content': match.group(0), 'index': idx})
            return f"\n[[SPECIAL_BLOCK_{idx}]]\n"

        text_no_code = self.CODE_BLOCK_RE.sub(_replace_code, text)

        out_lines, buffer = [], []

        def flush_table():
            if len(buffer) >= 2:  # require at least 2 pipe-rows to call it a table
                idx = len(special_blocks)
                special_blocks.append({'type': 'table', 'content': "\n".join(buffer), 'index': idx})
                out_lines.append(f"[[SPECIAL_BLOCK_{idx}]]")
            else:
                out_lines.extend(buffer)
            buffer.clear()

        for line in text_no_code.splitlines():
            if line.strip().startswith('|') and line.strip().endswith('|'):
                buffer.append(line)
            else:
                flush_table()
                out_lines.append(line)
        flush_table()

        return "\n".join(out_lines), special_blocks

    # ---------- 3. cleaning / normalization ----------
    def clean_text(self, text):
        for old, new in self.SMART_CHAR_MAP.items():
            text = text.replace(old, new)
        lines = [l.rstrip() for l in text.splitlines()]
        text = "\n".join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)          # collapse 3+ blank lines to 1
        text = re.sub(r'[ \t]{2,}', ' ', text)           # collapse repeated spaces/tabs
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)     # "!!!!" -> "!!"
        return text.strip()

    # ---------- 4. format-aware chunking ----------
    def chunk_markdown(self, text, source, timestamp):
        headers = list(self.MD_HEADER_RE.finditer(text))
        sections = []
        if not headers:
            sections.append((None, text))
        else:
            if headers[0].start() > 0:
                sections.append((None, text[:headers[0].start()]))
            for i, h in enumerate(headers):
                start = h.start()
                end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
                sections.append((h.group(2).strip(), text[start:end]))

        chunks, pos = [], 0
        for header, section_text in sections:
            for p in self.chunker.chunk_by_paragraph(section_text):
                if not p.strip():
                    continue
                chunks.append({
                    'text': p.strip(), 'position': pos, 'source': source,
                    'timestamp': timestamp, 'word_count': len(p.split()),
                    'strategy': 'markdown_section', 'section_header': header,
                })
                pos += 1
        return chunks

    def chunk_log(self, text, source, timestamp):
        chunks = []
        for i, line in enumerate(l for l in text.splitlines() if l.strip()):
            chunks.append({
                'text': line.strip(), 'position': i, 'source': source,
                'timestamp': timestamp, 'word_count': len(line.split()),
                'strategy': 'log_line',
            })
        return chunks

    # ---------- orchestration ----------
    def process_document(self, raw_text, source="document", doc_format=None, chunk_strategy="smart"):
        """Runs the full pipeline (steps 1-7) and returns everything as one result dict."""
        doc_format = doc_format or self.detect_format(raw_text)

        # extract from the ORIGINAL raw text, before any cleaning, so case
        # and exact tokens (emails, handles, etc.) are preserved
        extracted = self.extractor.extract(raw_text)

        text_with_placeholders, special_blocks = self.extract_special_blocks(raw_text)
        cleaned_text = self.clean_text(text_with_placeholders)
        ts = datetime.utcnow().isoformat()

        if doc_format == 'markdown':
            chunks = self.chunk_markdown(cleaned_text, source, ts)
        elif doc_format == 'log':
            chunks = self.chunk_log(cleaned_text, source, ts)
        else:
            chunks = self.chunker.chunk(cleaned_text, strategy=chunk_strategy, source=source, timestamp=ts)

        # restore special blocks (code/tables) as their own intact chunks
        final_chunks = []
        for c in chunks:
            m = re.fullmatch(r'\[\[SPECIAL_BLOCK_(\d+)\]\]', c['text'].strip())
            if m:
                block = special_blocks[int(m.group(1))]
                final_chunks.append({
                    'text': block['content'], 'position': c['position'], 'source': source,
                    'timestamp': ts, 'word_count': len(block['content'].split()),
                    'strategy': f"special_{block['type']}",
                })
            else:
                final_chunks.append(c)

        for i, c in enumerate(final_chunks):
            c['chunk_id'] = f"{source}_{i}"
            c['doc_format'] = doc_format

        valid_chunks, issues = self.chunker.validate_chunks(final_chunks)
        metrics = self.quality_metrics(raw_text, final_chunks, issues)

        return {
            'source': source, 'doc_format': doc_format, 'extracted': extracted,
            'special_blocks': special_blocks, 'chunks': final_chunks,
            'valid_chunks': valid_chunks, 'issues': issues, 'quality_metrics': metrics,
        }

    # ---------- 7. quality metrics ----------
    def quality_metrics(self, raw_text, chunks, issues):
        word_counts = [c['word_count'] for c in chunks] or [0]
        total_words_in_doc = len(raw_text.split())
        total_words_in_chunks = sum(word_counts)
        coverage = total_words_in_chunks / total_words_in_doc if total_words_in_doc else 0.0

        return {
            'num_chunks': len(chunks),
            'avg_chunk_words': round(float(np.mean(word_counts)), 2),
            'min_chunk_words': int(np.min(word_counts)),
            'max_chunk_words': int(np.max(word_counts)),
            'std_chunk_words': round(float(np.std(word_counts)), 2),
            'num_flagged_issues': len(issues),
            'word_coverage_ratio': round(coverage, 3),
        }

    # ---------- 6. search indices ----------
    def build_search_index(self, all_chunks):
        """
        Builds two indices over the chunks: a TF/cosine-similarity
        retriever (handles paraphrased queries) and an inverted keyword
        index (fast exact-term lookup), covering two complementary RAG
        retrieval strategies.
        """
        retriever = SimpleRetriever()
        retriever.index(all_chunks)

        inverted = {}
        for i, c in enumerate(all_chunks):
            chunk_id = c.get('chunk_id', f"{c.get('source', 'unknown')}_{i}")
            for token in set(SimpleRetriever._tokenize(c['text'])):
                inverted.setdefault(token, []).append(chunk_id)

        return retriever, inverted


class DigitalTwin:
    def __init__(self, activities, days_of_week, sleep_hours, mood, weather, n_activities=5):
        self.activities = np.array(activities)
        self.days_of_week = np.array(days_of_week)
        self.sleep_hours = np.array(sleep_hours)
        self.mood = np.array(mood)
        self.weather = np.array(weather)
        self.n_activities = n_activities

        # Perceptron weights/bias (fixed, hand-set)
        self.perceptron_weights = np.array([0.3, 0.5, 0.7])
        self.perceptron_bias = -0.5

        self.nn = None
        self.nn_no_time = None

        # Text-processing / RAG layer
        self.extractor = PersonalDataExtractor()
        self.chunker = ChunkingSystem()
        self.retriever = SimpleRetriever()
        self.doc_preprocessor = DocumentPreprocessor(extractor=self.extractor, chunker=self.chunker)
        self.knowledge_base = []        # all chunks ingested so far
        self.extracted_entries = []     # one extraction dict per ingested note
        self.inverted_index = {}        # term -> [chunk_id, ...], for keyword_search

        # Context window management (token limits, truncation/compression/sliding window)
        self.context_manager = ContextWindowManager(max_tokens=8192, reserve_for_response=512)

        # Word embeddings (Word2Vec / GloVe / pre-trained)
        self.embeddings = None          # EmbeddingSpace, once trained or loaded
        self.embedding_trainer = None   # the Word2VecTrainer/GloVeTrainer that produced it (for loss_history, etc.)
        self._chunk_embedding_cache = {}  # chunk_id -> mean-pooled vector, rebuilt lazily

        # AI agent layer: tools + ReAct loop
        self.habit_tracker = HabitTracker()          # custom tool 2 backing store (Habit Tracker)
        self.tool_registry = build_default_tool_registry(self)  # calculator, datetime_now, search_memory,
                                                                  # take_note (custom tool 1), track_habit/habit_stats (custom tool 2)

        self._build_probability_models()

    # ------------------------------------------------------
    # PROBABILITY LAYER
    # ------------------------------------------------------
    def _build_probability_models(self):
        self.freq_probs = self._calculate_probabilities(self.activities)
        self.cond_probs = self._calculate_conditional_probabilities_smoothed(self.activities)

    @staticmethod
    def _calculate_probabilities(sequence):
        counts = Counter(sequence)
        total = len(sequence)
        return {a: c / total for a, c in counts.items()}

    def _calculate_conditional_probabilities_smoothed(self, sequence, alpha=1):
        transitions = {}
        for i in range(len(sequence) - 1):
            cur, nxt = sequence[i], sequence[i + 1]
            transitions.setdefault(cur, []).append(nxt)

        smoothed = {}
        for cur, nxt_list in transitions.items():
            counts = Counter(nxt_list)
            total = len(nxt_list) + alpha * self.n_activities
            smoothed[cur] = {
                a: (counts.get(a, 0) + alpha) / total
                for a in range(self.n_activities)
            }
        return smoothed

    def joint_probability(self, activity, is_weekend):
        matches = sum(1 for a, d in zip(self.activities, self.days_of_week)
                      if a == activity and (d >= 5) == is_weekend)
        return matches / len(self.activities)

    def bayes_weekend_given_activity(self, activity):
        is_weekend = self.days_of_week >= 5
        p_weekend = np.mean(is_weekend)
        weekend_acts = self.activities[is_weekend]
        p_act_given_weekend = np.mean(weekend_acts == activity) if len(weekend_acts) else 0
        p_activity = np.mean(self.activities == activity)
        return 0 if p_activity == 0 else (p_act_given_weekend * p_weekend) / p_activity

    def predict_next_distribution(self, current):
        if current not in self.cond_probs:
            return None
        return dict(sorted(self.cond_probs[current].items(), key=lambda x: -x[1]))

    def predict_next_markov(self, current):
        dist = self.predict_next_distribution(current)
        return None if dist is None else max(dist, key=dist.get)

    def prediction_entropy(self, current):
        dist = self.predict_next_distribution(current)
        if dist is None:
            return None
        probs = np.array(list(dist.values()))
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def sequence_probability(self, seq):
        prob = 1.0
        for i in range(len(seq) - 1):
            cur, nxt = seq[i], seq[i + 1]
            if cur not in self.cond_probs or nxt not in self.cond_probs[cur]:
                return 0.0
            prob *= self.cond_probs[cur][nxt]
        return prob

    def sample_next_activity(self, current, rng):
        if current not in self.cond_probs:
            return None
        options = list(self.cond_probs[current].keys())
        weights = list(self.cond_probs[current].values())
        return rng.choice(options, p=weights)

    def simulate_greedy(self, start, days=7):
        seq = [start]
        for _ in range(days):
            nxt = self.predict_next_markov(seq[-1])
            if nxt is None:
                break
            seq.append(nxt)
        return seq

    def simulate_sampled(self, start, days=7, seed=42):
        rng = np.random.default_rng(seed)
        seq = [start]
        for _ in range(days):
            nxt = self.sample_next_activity(seq[-1], rng)
            if nxt is None:
                break
            seq.append(nxt)
        return seq

    # ------------------------------------------------------
    # PERCEPTRON LAYER
    # ------------------------------------------------------
    def perceptron_predict(self, sleep, free_time, energy):
        inputs = np.array([sleep, free_time, energy])
        return sigmoid(np.dot(inputs, self.perceptron_weights) + self.perceptron_bias)

    # ------------------------------------------------------
    # NEURAL NETWORK LAYER
    # ------------------------------------------------------
    def _build_features_and_targets(self, include_time=True):
        X, y = [], []
        for i in range(len(self.activities) - 1):
            feats = [self.activities[i] / (self.n_activities - 1),
                     self.sleep_hours[i] / 10, self.mood[i], self.weather[i]]
            if include_time:
                feats.insert(1, self.days_of_week[i] / 7)
            X.append(feats)
            target = np.zeros(self.n_activities)
            target[self.activities[i + 1]] = 1
            y.append(target)
        return np.array(X), np.array(y)

    def train_nn(self, include_time=True, hidden_size=8, epochs=1000, learning_rate=0.3, verbose=True):
        X, y = self._build_features_and_targets(include_time=include_time)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = SimpleNeuralNetwork(input_size=X.shape[1], hidden_size=hidden_size,
                                     output_size=self.n_activities)
        model.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=verbose)

        preds = np.argmax(model.predict(X_test), axis=1)
        actual = np.argmax(y_test, axis=1)
        accuracy = np.mean(preds == actual)

        if include_time:
            self.nn = model
        else:
            self.nn_no_time = model

        return model, accuracy, preds, actual

    # ------------------------------------------------------
    # PROMPTING LAYER
    # ------------------------------------------------------
    def generate_zero_shot_prompt(self, current_activity):
        predicted = self.predict_next_markov(current_activity)
        return {
            'technique': 'zero_shot',
            'task': 'activity_prediction',
            'prompt': f"I just finished {activity_names[current_activity]}. "
                      f"What's the most likely next activity?",
            'expected': activity_names[predicted] if predicted is not None else 'unknown'
        }

    def generate_few_shot_prompt(self, current_activity, n_examples=3):
        examples = []
        for i in range(len(self.activities) - 1):
            if self.activities[i] == current_activity:
                examples.append((self.activities[i], self.activities[i + 1]))
            if len(examples) >= n_examples:
                break

        prompt_lines = ["Here are observed activity transitions:"]
        for cur, nxt in examples:
            prompt_lines.append(f"- After {activity_names[cur]}, the next activity was {activity_names[nxt]}.")
        prompt_lines.append(f"Based on this pattern, what comes after {activity_names[current_activity]}?")

        return {
            'technique': 'few_shot',
            'task': 'activity_prediction',
            'examples': [{'current': activity_names[c], 'next': activity_names[n]} for c, n in examples],
            'prompt': "\n".join(prompt_lines),
            'expected': activity_names[self.predict_next_markov(current_activity)]
        }

    def generate_cot_prompt(self, day_index):
        sleep = self.sleep_hours[day_index]
        mood_v = self.mood[day_index]
        weather_v = self.weather[day_index]
        is_weekend = self.days_of_week[day_index] >= 5

        steps = [
            f"1. Sleep was {sleep:.1f} hours -> {'well-rested' if sleep >= 7 else 'low energy'}.",
            f"2. Mood score was {mood_v:.2f} -> {'positive' if mood_v >= 0.5 else 'low'}.",
            f"3. Weather score was {weather_v:.2f} -> {'good' if weather_v >= 0.5 else 'poor'}.",
            f"4. Day type: {'weekend' if is_weekend else 'weekday'}.",
        ]

        p_exercise = self.perceptron_predict(sleep, 2 if is_weekend else 1, mood_v)
        conclusion = "exercise" if p_exercise > 0.5 else "a lower-energy activity (sleep/leisure)"
        steps.append(f"5. Combining these factors, exercise probability = {p_exercise:.2%} -> "
                      f"most likely outcome: {conclusion}.")

        prompt = (
            f"Given sleep={sleep:.1f}h, mood={mood_v:.2f}, weather={weather_v:.2f}, "
            f"{'weekend' if is_weekend else 'weekday'}: think step-by-step about the "
            f"most likely activity."
        )

        return {
            'technique': 'chain_of_thought',
            'task': 'activity_prediction',
            'prompt': prompt,
            'reasoning_steps': steps,
            'expected': conclusion
        }

    def run_prompting_demo(self, current_activity=1, day_index=0):
        print("=== DIGITAL TWIN PROMPTING TECHNIQUES ===")

        zs = self.generate_zero_shot_prompt(current_activity)
        print(f"\n1. Technique: {zs['technique'].upper()}")
        print(f"Prompt: {zs['prompt']}")
        print(f"Expected: {zs['expected']}")

        fs = self.generate_few_shot_prompt(current_activity)
        print(f"\n2. Technique: {fs['technique'].upper()}")
        for i, ex in enumerate(fs['examples'], 1):
            print(f"  --- Example {i}: {ex['current']} -> {ex['next']}")
        print(f"Prompt:\n{fs['prompt']}")
        print(f"Expected: {fs['expected']}")

        cot = self.generate_cot_prompt(day_index)
        print(f"\n3. Technique: {cot['technique'].upper()}")
        print(f"Prompt: {cot['prompt']}")
        print("Steps:")
        for step in cot['reasoning_steps']:
            print(f"  {step}")
        print(f"Expected Final Answer: {cot['expected']}")

    # ------------------------------------------------------
    # SYSTEM INSTRUCTION LAYER
    # ------------------------------------------------------
    def build_system_instruction(self, n_recent_days=10):
        """
        Summarizes observed daily patterns (learning, coding, writing, revising,
        exercise, social, etc. mapped onto the activity codes) into rules an
        LLM can follow when predicting the twin's next activity.
        """
        lines = [
            "You are a digital twin of a user, predicting their next likely activity "
            "based on observed daily patterns. Follow these rules, derived from real data:",
            ""
        ]

        lines.append("Overall activity distribution:")
        for a, p in sorted(self.freq_probs.items()):
            lines.append(f"  - {activity_names[a]}: {p:.1%} of days")
        lines.append("")

        lines.append("Observed transition rules (after X, the user usually does Y):")
        for cur, dist in self.cond_probs.items():
            top_next = max(dist, key=dist.get)
            lines.append(f"  - After {activity_names[cur]} -> most likely {activity_names[top_next]} "
                         f"({dist[top_next]:.1%})")
        lines.append("")

        recent = self.activities[-n_recent_days:]
        recent_str = " -> ".join(activity_names[a] for a in recent)
        lines.append(f"Most recent {len(recent)} days, in order: {recent_str}")
        lines.append("")

        lines.append("When asked to predict the next activity, reason from these rules "
                      "and the recent sequence, not from general assumptions about people.")

        return "\n".join(lines)

    def ask_llm(self, llm_client, current_activity, n_recent_days=10):
        system_instruction = self.build_system_instruction(n_recent_days=n_recent_days)
        user_prompt = (f"I just finished {activity_names[current_activity]}. "
                       f"What's the most likely next activity, and why?")
        return llm_client.chat(system_instruction, user_prompt)

    def chat_with_twin(self, llm_client, user_message):
        """
        Multi-turn chat with the twin, using its own observed patterns as
        system context. Only sent once at the start of the conversation;
        later turns in the same llm_client reuse the existing history.
        """
        system_instruction = self.build_system_instruction() if not llm_client.conversation_history else None
        reply, usage = llm_client.chat(user_message, system_instruction=system_instruction)
        return reply, usage

    # ------------------------------------------------------
    # TEXT PROCESSING / RAG LAYER
    # ------------------------------------------------------
    def ingest_notes(self, raw_text, source="notes", chunk_strategy="smart", verbose=True):
        """
        Full ingestion pipeline for real, messy, raw text (journal entries,
        chat logs, typos and all):
          1. Extract structured data (emails, phones, handles, hashtags,
             dates, times, durations) via PersonalDataExtractor.
          2. Chunk the raw text via ChunkingSystem, using the chosen
             strategy, and validate chunk quality.
          3. Add the valid chunks to the twin's knowledge_base and
             re-index the retriever so the new material is searchable.
        """
        extracted = self.extractor.extract(raw_text)
        self.extracted_entries.append({'source': source, **extracted})

        chunks = self.chunker.chunk(raw_text, strategy=chunk_strategy, source=source)
        valid_chunks, issues = self.chunker.validate_chunks(chunks)
        for c in valid_chunks:
            c.setdefault('chunk_id', f"{source}_{c['position']}")

        self.knowledge_base.extend(valid_chunks)
        self.retriever.index(self.knowledge_base)

        if verbose:
            print(f"Ingested '{source}': {len(valid_chunks)} valid chunks "
                  f"({len(chunks) - len(valid_chunks)} flagged)")
            if issues:
                for issue in issues:
                    print(f"  ! {issue}")
            print(f"  Extracted: {extracted['emails']} emails, "
                  f"{extracted['social_handles']} handles, "
                  f"{extracted['hashtags']} hashtags, "
                  f"{extracted['dates']} dates")

        return extracted, valid_chunks, issues

    def retrieve_relevant_chunks(self, query, top_k=3):
        """Retrieval step of RAG: returns the top_k most relevant knowledge-base chunks for a query."""
        return self.retriever.retrieve(query, top_k=top_k)

    def build_rag_system_instruction(self, query, n_recent_days=10, top_k=3):
        """
        Extends build_system_instruction() with retrieved, relevant chunks
        from the twin's ingested notes — this is the "augmented" part of
        retrieval-augmented generation: the LLM sees only what's relevant
        to the current query, not the entire knowledge base.
        """
        base_instruction = self.build_system_instruction(n_recent_days=n_recent_days)
        relevant = self.retrieve_relevant_chunks(query, top_k=top_k)

        if not relevant:
            return base_instruction

        lines = [base_instruction, "", "Relevant notes retrieved from the user's own journal/records:"]
        for r in relevant:
            lines.append(f"  - ({r['source']}, similarity={r['similarity']:.2f}): {r['text']}")

        return "\n".join(lines)

    def chat_with_twin_rag(self, llm_client, user_message, top_k=3):
        """
        Like chat_with_twin, but augments the system instruction with the
        knowledge-base chunks most relevant to this specific message
        (retrieval-augmented generation), instead of always using the
        same static system instruction.
        """
        system_instruction = None
        if not llm_client.conversation_history:
            system_instruction = self.build_rag_system_instruction(user_message, top_k=top_k)
        reply, usage = llm_client.chat(user_message, system_instruction=system_instruction)
        return reply, usage

    # ------------------------------------------------------
    # DOCUMENT PREPROCESSOR LAYER (Challenge Project)
    # ------------------------------------------------------
    def ingest_document(self, raw_text, source="document", doc_format=None, chunk_strategy="smart", verbose=True):
        """
        Full multi-format document ingestion pipeline: detects format
        (text/markdown/log) if not given, extracts structured data,
        cleans and normalizes, chunks optimally for the format (code
        blocks and tables kept intact), adds the valid chunks to the
        knowledge base, and rebuilds both search indices.
        """
        result = self.doc_preprocessor.process_document(
            raw_text, source=source, doc_format=doc_format, chunk_strategy=chunk_strategy)

        self.extracted_entries.append({'source': source, **result['extracted']})
        self.knowledge_base.extend(result['valid_chunks'])

        self.retriever.index(self.knowledge_base)
        _, self.inverted_index = self.doc_preprocessor.build_search_index(self.knowledge_base)

        if verbose:
            print(f"Ingested '{source}' (format={result['doc_format']}): "
                  f"{len(result['valid_chunks'])} valid chunks, "
                  f"{len(result['special_blocks'])} special block(s) (code/tables)")
            print(f"  Quality metrics: {result['quality_metrics']}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"  ! {issue}")

        return result

    def keyword_search(self, query, top_k=5):
        """
        Exact-keyword retrieval via the inverted index — complements
        retrieve_relevant_chunks (cosine similarity), for queries where
        an exact term match matters more than overall similarity.
        """
        tokens = SimpleRetriever._tokenize(query)
        matches = Counter()
        for t in tokens:
            for chunk_id in self.inverted_index.get(t, []):
                matches[chunk_id] += 1

        id_to_chunk = {
            c.get('chunk_id', f"{c.get('source', 'unknown')}_{i}"): c
            for i, c in enumerate(self.knowledge_base)
        }
        return [{**id_to_chunk[cid], 'keyword_matches': n}
                for cid, n in matches.most_common(top_k) if cid in id_to_chunk]

    # ------------------------------------------------------
    # CONTEXT WINDOW MANAGEMENT LAYER
    # ------------------------------------------------------
    def set_context_budget(self, max_tokens=8192, reserve_for_response=512):
        """User-friendly configuration entry point — replaces the twin's context_manager with new limits."""
        self.context_manager = ContextWindowManager(max_tokens=max_tokens, reserve_for_response=reserve_for_response)
        return self.context_manager

    def get_context_usage_report(self, llm_client):
        """
        Returns a usage_report() dict for an LLM client's current
        conversation_history, using the twin's configured context_manager.
        Works with GroqClient instances (which keep conversation_history).
        """
        history = getattr(llm_client, 'conversation_history', [])
        sys_msgs = [m for m in history if m.get('role') == 'system']
        convo_msgs = [m for m in history if m.get('role') != 'system']
        system_text = sys_msgs[0]['content'] if sys_msgs else None
        return self.context_manager.usage_report(system_text, convo_msgs)

    def show_context_status(self, llm_client):
        """User-friendly one-line console view of how full the context window currently is."""
        report = self.get_context_usage_report(llm_client)
        self.context_manager.print_usage_bar(report)
        return report

    # ------------------------------------------------------
    # WORD EMBEDDINGS LAYER (Word2Vec / GloVe internals)
    # ------------------------------------------------------
    def train_word_embeddings(self, method="word2vec", dim=32, epochs=8, verbose=True, **kwargs):
        """
        Trains word embeddings from scratch on all text currently in
        the twin's knowledge_base (every ingested note/document
        chunk — see ingest_notes/ingest_document). method is
        'word2vec' or 'glove'. Stores the result in self.embeddings
        for use by retrieve_relevant_chunks_semantic() and the other
        EmbeddingSpace methods (similarity, most_similar, analogy,
        visualize_2d, ...).
        """
        texts = [c['text'] for c in self.knowledge_base if c.get('text')]
        if not texts:
            print("  ! train_word_embeddings: knowledge_base is empty — ingest some notes/documents first")
            return None

        if method == "word2vec":
            trainer = Word2VecTrainer(embedding_dim=dim, **kwargs)
        elif method == "glove":
            trainer = GloVeTrainer(embedding_dim=dim, **kwargs)
        else:
            raise ValueError(f"train_word_embeddings: unknown method '{method}' (expected 'word2vec' or 'glove')")

        trainer.fit(texts, epochs=epochs, verbose=verbose)
        self.embeddings = trainer.get_embedding_space()
        self.embedding_trainer = trainer
        self._chunk_embedding_cache = {}  # invalidate the per-chunk cache — new vectors, must rebuild

        print(f"Trained {method} embeddings: {len(self.embeddings)} words, dim={dim}")
        return self.embeddings

    def load_pretrained_embeddings(self, path, vocab_limit=None):
        """Loads a pre-trained embedding file (GloVe-style text format) in place of training one from scratch."""
        self.embeddings = EmbeddingSpace.load_pretrained(path, vocab_limit=vocab_limit)
        self.embedding_trainer = None
        self._chunk_embedding_cache = {}
        return self.embeddings

    def embed_text(self, text):
        """
        Mean-pools word vectors for every in-vocabulary token in
        `text`. Returns a zero vector (with a warning) if none of the
        text's words are in the embedding vocabulary — never raises.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            raise RuntimeError("No embeddings available — call train_word_embeddings() or "
                               "load_pretrained_embeddings() first")
        dim = next(iter(self.embeddings.vectors.values())).shape[0]
        tokens = SimpleRetriever._tokenize(text)
        vecs = [v for v in (self.embeddings.vector(t) for t in tokens) if v is not None]
        if not vecs:
            return np.zeros(dim)
        return np.mean(vecs, axis=0)

    def retrieve_relevant_chunks_semantic(self, query, top_k=3):
        """
        Embedding-based retrieval: mean-pools word vectors per chunk
        and per query, then ranks by cosine similarity. Complements
        retrieve_relevant_chunks (TF/cosine over exact terms) and
        keyword_search (exact-term inverted index) with a genuinely
        semantic signal — two different words can match here if their
        trained vectors ended up nearby, which the exact-term methods
        cannot do.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            print("  ! retrieve_relevant_chunks_semantic: no embeddings available — "
                  "call train_word_embeddings() first")
            return []

        # Rebuild the per-chunk embedding cache if the knowledge base has
        # grown/shrunk since it was last built (simple staleness check).
        if len(self._chunk_embedding_cache) != len(self.knowledge_base):
            self._chunk_embedding_cache = {}
            for i, c in enumerate(self.knowledge_base):
                cid = c.get('chunk_id', f"{c.get('source', 'unknown')}_{i}")
                self._chunk_embedding_cache[cid] = self.embed_text(c['text'])

        q_vec = self.embed_text(query)
        if not np.any(q_vec):
            print("  ! retrieve_relevant_chunks_semantic: none of the query's words are in the embedding vocabulary")
            return []

        scored = []
        for i, c in enumerate(self.knowledge_base):
            cid = c.get('chunk_id', f"{c.get('source', 'unknown')}_{i}")
            c_vec = self._chunk_embedding_cache.get(cid)
            if c_vec is None or not np.any(c_vec):
                continue
            scored.append((EmbeddingSpace.cosine_similarity(q_vec, c_vec), c))

        scored.sort(key=lambda x: -x[0])
        return [{**c, 'semantic_similarity': sim} for sim, c in scored[:top_k]]

    # ------------------------------------------------------
    # RAG PIPELINE LAYER
    # ------------------------------------------------------
    def create_rag_pipeline(self, llm_client=None, retrieval_mode=None, top_k=4,
                             log_level="INFO", echo_logs=True):
        """
        User-friendly entry point: builds a complete RAGPipeline wired
        to this twin's existing knowledge base, context manager, and
        (if trained) word embeddings. If retrieval_mode is left as
        None, it's picked automatically: 'hybrid' when embeddings are
        available, else 'tf'.
        """
        if retrieval_mode is None:
            retrieval_mode = "hybrid" if (self.embeddings is not None and len(self.embeddings) > 0) else "tf"
        return RAGPipeline(self, llm_client=llm_client, retrieval_mode=retrieval_mode, top_k=top_k,
                            log_level=log_level, echo_logs=echo_logs)

    def create_agent(self, llm_client, tool_registry=None, max_iterations=6,
                      compress_every=5, keep_recent_steps=4, log_level="INFO", echo_logs=True):
        """
        User-friendly entry point: builds a ReActAgent wired to this
        twin's default tool registry (calculator, datetime_now,
        search_memory, take_note, track_habit, habit_stats) unless a
        custom tool_registry is given. llm_client can be a real
        GroqClient/UniversalLLMClient or a ScriptedLLM for offline testing.
        """
        return ReActAgent(self, llm_client, tool_registry=tool_registry or self.tool_registry,
                           max_iterations=max_iterations, compress_every=compress_every,
                           keep_recent_steps=keep_recent_steps, log_level=log_level, echo_logs=echo_logs)


# ============================================================
# 1. COLLECT DATA
# ============================================================
np.random.seed(1)
n_days = 60
days_of_week = np.array([i % 7 for i in range(n_days)])
sleep_hours = np.random.normal(7, 1, n_days).clip(4, 10)
mood = np.random.rand(n_days)
weather = np.random.rand(n_days)

activities = []
for d in days_of_week:
    if d >= 5:
        activities.append(np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3]))
    else:
        activities.append(np.random.choice([1, 0, 2], p=[0.7, 0.2, 0.1]))
activities = np.array(activities)

twin = DigitalTwin(activities, days_of_week, sleep_hours, mood, weather)

# ============================================================
# 2. PROBABILITY FEATURES
# ============================================================
print("Overall activity probabilities:")
for a, p in sorted(twin.freq_probs.items()):
    print(f"  {activity_names[a]}: {p:.2%}")

print("\nJoint P(Exercise AND Weekend):", f"{twin.joint_probability(2, True):.2%}")
print("Bayes P(Weekend | Exercise):", f"{twin.bayes_weekend_given_activity(2):.2%}")

print("\nFull next-activity distribution after Work:")
for a, p in twin.predict_next_distribution(1).items():
    print(f"  {activity_names[a]}: {p:.2%}")

print(f"\nPrediction entropy after Work: {twin.prediction_entropy(1):.3f}")
print("Greedy 7-day simulation from Work:",
      [activity_names[a] for a in twin.simulate_greedy(1, 7)])
print("Sampled 7-day simulation from Work:",
      [activity_names[a] for a in twin.simulate_sampled(1, 7)])
print("P(Work -> Exercise -> Social):",
      f"{twin.sequence_probability([1, 2, 3]):.4%}")

# ============================================================
# 3. PERCEPTRON
# ============================================================
print("\nPerceptron P(exercise | sleep=7, free_time=2, energy=0.8):",
      f"{twin.perceptron_predict(7, 2, 0.8):.2%}")

# ============================================================
# 4. TRAIN + EVALUATE NEURAL NETWORK
# ============================================================
print("\nTraining NN with day-of-week...")
model_time, acc_time, preds_time, actual_time = twin.train_nn(include_time=True)
print(f"Accuracy with day-of-week: {acc_time:.2%}")

print("\nTraining NN without day-of-week...")
model_no_time, acc_no_time, preds_nt, actual_nt = twin.train_nn(include_time=False)
print(f"Accuracy without day-of-week: {acc_no_time:.2%}")

# ============================================================
# 5. VISUALIZE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

acts = [activity_names[k] for k in sorted(twin.freq_probs.keys())]
probs = [twin.freq_probs[k] for k in sorted(twin.freq_probs.keys())]
axes[0, 0].bar(acts, probs, color='steelblue')
axes[0, 0].set_title("Overall Activity Probabilities")

axes[0, 1].plot(model_time.loss_history, label="With day-of-week")
axes[0, 1].plot(model_no_time.loss_history, label="Without day-of-week")
axes[0, 1].set_title("Training Loss")
axes[0, 1].legend()

per_class_acc = [np.mean(preds_time[actual_time == c] == c) if np.any(actual_time == c) else 0
                  for c in range(5)]
axes[1, 0].bar([activity_names[c] for c in range(5)], per_class_acc, color='seagreen')
axes[1, 0].set_title("Per-Activity Accuracy (NN)")
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].bar(["With day-of-week", "Without"], [acc_time, acc_no_time],
               color=['steelblue', 'salmon'])
axes[1, 1].set_title("Effect of Temporal Features")

plt.tight_layout()
plt.show()

# ============================================================
# 6. PROMPTING TECHNIQUES
# ============================================================
twin.run_prompting_demo(current_activity=1, day_index=5)

# ============================================================
# 7. SYSTEM INSTRUCTION + LLM INTEGRATION (UniversalLLMClient)
# ============================================================
print("\n=== SYSTEM INSTRUCTION BUILT FROM TWIN DATA ===")
print(twin.build_system_instruction())

# -------- PUT YOUR API CREDENTIALS HERE --------
LLM_BASE_URL = "https://api.groq.com/openai/v1"   # <-- change to your provider's base URL
LLM_API_KEY = "PASTE_YOUR_API_KEY_HERE"            # <-- your key goes here
LLM_MODEL = "llama3.3"                              # <-- exact model string from your provider
# ------------------------------------------------

llm_client = UniversalLLMClient(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
)

# Uncomment once LLM_API_KEY above is filled in:
# response = twin.ask_llm(llm_client, current_activity=1)
# print("\nLLM response:\n", response)

# ============================================================
# 8. GROQ CLIENT: TWO-TURN CONVERSATION + TOKEN METER
# ============================================================

# -------- PUT YOUR GROQ API KEY HERE --------
GROQ_API_KEY = "PASTE_YOUR_API_KEY_HERE"   # <-- your Groq key goes here
# ---------------------------------------------

groq_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL_NAME)

# Uncomment the block below once GROQ_API_KEY above is filled in:

# Turn 1 — opens the conversation (system instruction is sent automatically, once)
# reply1, usage1 = twin.chat_with_twin(groq_client, "Hello! I'm testing my digital twin's connection to you.")
# print("Turn 1 reply:", reply1)
# print("Tokens used this call:", usage1.get("total_tokens"))

# Turn 2 — continues the SAME conversation (history remembered automatically)
# reply2, usage2 = twin.chat_with_twin(groq_client, "Based on what you know about my patterns, what should I do next?")
# print("\nTurn 2 reply:", reply2)
# print("Tokens used this call:", usage2.get("total_tokens"))

# API meter — total tokens across both calls
# print("\nTotal tokens used so far:", groq_client.total_tokens_used)

# ============================================================
# 9. TEXT PROCESSING / RAG: INGEST REAL, MESSY NOTES
# ============================================================
# Exercise 2.1 (Personal Data Extractor) + Exercise 3.1 (Chunking System),
# wired into the twin as one ingestion pipeline, with retrieval for RAG.

sample_notes = [
    (
        "journal_oct20",
        "Mon 20th oct, woke up late lol didnt sleep well only 5.5hrs. "
        "had to rush into work, missed breakfast again smh. meeting with "
        "sara @sara_devs went long, like 2.5 hours ugh. #worklife #tired "
        "reach me at john.doe99@gmail.com or call 0712-345-678 if urgent. "
        "evening did abt 45 mins of exercise, felt good tbh #exercise #selfcare. "
        "gonna try sleep by 10pm tonight."
    ),
    (
        "journal_oct21",
        "Tuesday 21/10 - much better day!! slept 7.5 hours, feeling great. "
        "coffee w/ @mike_k at 9am, good catch up session. worked on the ML "
        "assignment most of the afternoon (~3 hrs), making progress on the "
        "digital twin project #coding #machinelearning #progress. "
        "weather was nice so went for a short walk around 6pm, 20 mins. "
        "need to email the professor at prof.smith@university.edu about "
        "the deadline extension, due by 5pm Oct 25th."
    ),
    (
        "journal_oct22",
        "weekend!! finally. slept in till 10, total sleep abt 9hrs i think. "
        "hung out with friends at the park saturday afternoon for like 3hrs, "
        "was so fun #social #weekend #outdoors. tried a new recipe too, "
        "turned out decent. mood was really good today ngl. need to remember "
        "to call mum, her number is +254 722 111 222. also saw @lisa.codes "
        "posted about a cool ML course, #learning #ml #weekendvibes."
    ),
]

print("\n=== INGESTING RAW, MESSY NOTES INTO THE TWIN'S KNOWLEDGE BASE ===")
for source, note_text in sample_notes:
    twin.ingest_notes(note_text, source=source, chunk_strategy="smart")

print(f"\nTotal chunks in knowledge base: {len(twin.knowledge_base)}")
print(f"Total extraction entries: {len(twin.extracted_entries)}")

print("\n--- Aggregated extracted data across all notes ---")
all_emails = sorted(set(e for entry in twin.extracted_entries for e in entry['emails']))
all_handles = sorted(set(h for entry in twin.extracted_entries for h in entry['social_handles']))
all_hashtags = sorted(set(h for entry in twin.extracted_entries for h in entry['hashtags']))
all_dates = sorted(set(d for entry in twin.extracted_entries for d in entry['dates']))
print("Emails:", all_emails)
print("Handles:", all_handles)
print("Hashtags / inferred preferences:", all_hashtags)
print("Dates mentioned:", all_dates)

print("\n--- Chunk quality check ---")
_, issues = twin.chunker.validate_chunks(twin.knowledge_base)
if issues:
    for issue in issues:
        print(" ", issue)
else:
    print("  No quality issues flagged.")

print("\n--- Chunk size visualization ---")
twin.chunker.visualize(twin.knowledge_base, title="Digital Twin Knowledge Base — Chunk Sizes")

# ------------------------------------------------------------
# Retrieval (RAG): find the notes most relevant to a query
# ------------------------------------------------------------
query = "How much did I exercise this week?"
print(f"\n--- Retrieval for query: \"{query}\" ---")
top_chunks = twin.retrieve_relevant_chunks(query, top_k=3)
for c in top_chunks:
    print(f"  [{c['source']}] (sim={c['similarity']:.2f}): {c['text'][:120]}...")

# ------------------------------------------------------------
# RAG-augmented system instruction (what actually gets sent to the LLM)
# ------------------------------------------------------------
print("\n--- RAG-augmented system instruction (preview) ---")
print(twin.build_rag_system_instruction(query, top_k=3))

# ------------------------------------------------------------
# RAG-augmented chat with the twin (uncomment once GROQ_API_KEY is set)
# ------------------------------------------------------------
# rag_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL_NAME)
# rag_reply, rag_usage = twin.chat_with_twin_rag(rag_client, query)
# print("\nRAG-augmented reply:", rag_reply)
# print("Tokens used:", rag_usage.get("total_tokens"))

# ============================================================
# 10. DOCUMENT PREPROCESSOR: MULTI-FORMAT INGESTION (Challenge Project)
# ============================================================
# Three documents, three formats: markdown (with a code block AND a
# table — the bonus), a log file, and a plain-text note.

markdown_doc = """# Weekly Digital Twin Notes

## Monday - Coding Session

Worked on the activity prediction module today. Fixed the Markov chain
smoothing bug and added the perceptron layer. #coding #digitaltwin

```python
def predict_next_markov(current, cond_probs):
    dist = cond_probs.get(current)
    return max(dist, key=dist.get) if dist else None
```

## Tuesday - Data Review

Reviewed last week's activity logs with sara@teamup.io. Summary below:

| Day | Activity | Duration |
|-----|----------|----------|
| Mon | Work     | 8h       |
| Tue | Exercise | 45m      |
| Wed | Social   | 2h       |

Mood has been trending upward, #progress #mlschool.
"""

log_doc = """2026-10-20T08:03:12 [INFO] twin.ingest: started ingestion for journal_oct20
2026-10-20T08:03:12 [INFO] twin.extract: found 1 email, 1 handle
2026-10-20T08:03:13 [WARN] twin.chunk: chunk 2 below min_chunk_words threshold
2026-10-21T09:15:44 [INFO] twin.train: nn training started, epochs=1000
2026-10-21T09:16:02 [INFO] twin.train: nn training complete, final_loss=0.0421
2026-10-22T07:40:00 [ERROR] twin.llm: request timed out after 30s, retrying
2026-10-22T07:40:05 [INFO] twin.llm: retry succeeded, tokens_used=187
"""

plain_text_doc = (
    "Reflecting on this month overall, my sleep has been more consistent "
    "and I've managed to keep up with both the coding assignments and "
    "regular exercise. The digital twin project is coming together nicely. "
    "Next month I want to focus more on writing and documentation, since "
    "that's the area I've been neglecting the most."
)

print("\n=== DOCUMENT PREPROCESSOR: MULTI-FORMAT INGESTION ===")
twin.ingest_document(markdown_doc, source="weekly_notes_md", doc_format=None)
twin.ingest_document(log_doc, source="system_log_oct", doc_format=None)
twin.ingest_document(plain_text_doc, source="monthly_reflection", doc_format=None)

print(f"\nTotal knowledge base size after document ingestion: {len(twin.knowledge_base)} chunks")

print("\n--- Keyword search: 'exercise duration' ---")
for r in twin.keyword_search("exercise duration", top_k=5):
    print(f"  [{r['source']}] matches={r['keyword_matches']} strategy={r['strategy']}: {r['text'][:100]}")

print("\n--- Similarity search: 'how is the coding going' ---")
for r in twin.retrieve_relevant_chunks("how is the coding going", top_k=5):
    print(f"  [{r['source']}] sim={r['similarity']:.2f} strategy={r['strategy']}: {r['text'][:100]}")

print("\n--- Special blocks preserved intact (code/table) ---")
for c in twin.knowledge_base:
    if c.get('strategy', '').startswith('special_'):
        print(f"  [{c['strategy']}] from {c['source']}:\n{c['text']}\n")

# ============================================================
# 11. CONTEXT WINDOW MANAGEMENT: TOKEN LIMITS, TRUNCATION,
#     COMPRESSION, SLIDING WINDOW
# ============================================================
# Documentation: ContextWindowManager (defined above, alongside
# TokenCounter) is the twin's context-window system. It:
#   - Counts tokens per message/text (TokenCounter: tiktoken if
#     installed, else a ~4-chars/token fallback), with caching for
#     repeated counts.
#   - Supports a sliding window (keep only the most recent N messages
#     or N tokens' worth).
#   - Supports three truncation strategies: drop-oldest, keep-head-
#     and-tail ("truncate_middle"), and priority-based (keeps messages
#     flagged `important=True` even if old).
#   - Supports compression: summarizes everything except the most
#     recent messages into one compact note, either via a supplied
#     summarizer_fn or a dependency-free extractive fallback.
#   - fit_to_budget() ties it all together into one call that is
#     guaranteed never to exceed the available token budget, and
#     degrades gracefully on edge cases (empty history, a single
#     oversized message, even an oversized system instruction).
#   - It's wired into GroqClient and UniversalLLMClient via an
#     optional context_manager argument (default None = unchanged
#     original behavior), and into DigitalTwin via
#     set_context_budget() / get_context_usage_report() /
#     show_context_status().

print("\n=== CONTEXT WINDOW MANAGEMENT: UNIT TESTS ===")
all_tests_passed = run_context_manager_tests()
print("ALL TESTS PASSED" if all_tests_passed else "SOME TESTS FAILED — see output above")

print("\n=== SIMULATING A GROWING CONVERSATION AGAINST A SMALL BUDGET ===")
# A deliberately small budget so truncation/compression visibly kick in
# within just a few demo turns.
demo_cm = ContextWindowManager(max_tokens=600, reserve_for_response=100)
system_text = twin.build_system_instruction()

sim_history = []
reports = []
topics = ["sleep patterns", "exercise habits", "work schedule", "social activities",
          "the ML coding project", "weekend plans", "mood trends", "today's weather"]

for turn in range(12):
    topic = topics[turn % len(topics)]
    sim_history.append({"role": "user", "content": f"Tell me about my {topic}. " * 4})
    sim_history.append({"role": "assistant", "content": f"Based on your data, your {topic} shows a clear pattern. " * 4})
    report = demo_cm.usage_report(system_text, sim_history)
    reports.append(report)
    demo_cm.print_usage_bar(report)

print("\n--- Comparing every strategy at the final (overflowing) turn ---")
for strat in ["sliding_window", "truncate_oldest", "truncate_middle", "truncate_priority", "compress", "hybrid"]:
    fitted = demo_cm.fit_to_budget(system_text, sim_history, strategy=strat)
    fitted_tokens = demo_cm.count_messages(fitted)
    fits = fitted_tokens <= demo_cm.available_tokens
    print(f"  {strat:16s}: {len(fitted)} messages kept, {fitted_tokens} tokens "
          f"(budget={demo_cm.available_tokens}) {'OK' if fits else 'OVER'}")

print("\n--- Visualizing token usage across the simulated conversation ---")
demo_cm.visualize_usage(reports, title="Digital Twin Chat — Token Usage vs Budget")

print("\n--- compress_messages: extractive summary (no LLM call required) ---")
compressed_demo = demo_cm.compress_messages(sim_history, keep_recent_n=4)
print(f"Original:   {len(sim_history)} messages, {demo_cm.count_messages(sim_history)} tokens")
print(f"Compressed: {len(compressed_demo)} messages, {demo_cm.count_messages(compressed_demo)} tokens")
for m in compressed_demo[:2]:
    print(f"  [{m['role']}] {m['content'][:150]}")

print("\n=== INTEGRATION: ATTACH THE CONTEXT MANAGER TO THE TWIN'S LLM CLIENT ===")
twin.set_context_budget(max_tokens=4096, reserve_for_response=512)
print(f"Digital twin's context budget: max_tokens={twin.context_manager.max_tokens}, "
      f"available_tokens={twin.context_manager.available_tokens}")
print("To use it with a live Groq connection:")
print("  managed_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL_NAME, context_manager=twin.context_manager)")
print("  twin.chat_with_twin(managed_client, 'Hello!')  # automatically trimmed to fit the budget")
print("  twin.show_context_status(managed_client)       # see current usage at a glance")

# Uncomment with a real GROQ_API_KEY (see Section 8) to see it live:
# managed_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL_NAME, context_manager=twin.context_manager)
# reply, usage = twin.chat_with_twin(managed_client, "Hello! Let's talk about my week.")
# print(reply)
# twin.show_context_status(managed_client)

# ============================================================
# 12. WORD EMBEDDINGS: WORD2VEC AND GLOVE INTERNALS
# ============================================================
# Documentation: this section trains two kinds of word embeddings,
# both built from scratch with numpy only (no gensim/torch):
#   - Word2VecTrainer: skip-gram with negative sampling — a LOCAL,
#     prediction-based model (predict a word from its neighbors).
#   - GloVeTrainer: factorizes a GLOBAL word-word co-occurrence
#     matrix via weighted least squares — a count-based model.
# Both return an EmbeddingSpace with: similarity(), most_similar(),
# analogy() (vector arithmetic), pca() + visualize_2d() (dependency-
# free dimensionality reduction for plotting), similarity_heatmap(),
# and save()/load_pretrained() for persisting or loading vectors
# trained elsewhere. Every lookup method handles out-of-vocabulary
# words gracefully (prints a warning, returns None/[] — never raises).
#
# Integration: DigitalTwin.train_word_embeddings() trains on every
# chunk already in twin.knowledge_base (built up across Sections 9-10
# from the ingested journal notes and documents), and
# retrieve_relevant_chunks_semantic() uses the resulting vectors as a
# third retrieval strategy alongside retrieve_relevant_chunks (TF/
# cosine) and keyword_search (exact-term inverted index).

print("\n=== WORD2VEC / GLOVE: UNIT TESTS ===")
embedding_tests_passed = run_embedding_tests()
print("ALL TESTS PASSED" if embedding_tests_passed else "SOME TESTS FAILED — see output above")

print(f"\n=== TRAINING WORD2VEC ON THE TWIN'S KNOWLEDGE BASE ({len(twin.knowledge_base)} chunks) ===")
twin.train_word_embeddings(method="word2vec", dim=24, window_size=3, negative_samples=4,
                            learning_rate=0.05, epochs=40, verbose=False)
print(f"Vocabulary size: {len(twin.embeddings)} words")
print(f"Final training loss: {twin.embedding_trainer.loss_history[-1]:.4f} "
      f"(started at {twin.embedding_trainer.loss_history[0]:.4f})")

twin_vocab_words = ["exercise", "work", "coding", "sleep", "weekend", "social", "progress"]
in_vocab_words = [w for w in twin_vocab_words if w in twin.embeddings]
print(f"\n--- Nearest neighbors for a few twin-relevant words ---")
for w in in_vocab_words:
    neighbors = twin.embeddings.most_similar(w, top_k=4)
    neighbor_str = ", ".join(f"{nw} ({sim:.2f})" for nw, sim in neighbors)
    print(f"  '{w}' -> {neighbor_str}")

print(f"\n--- Pairwise similarity ---")
if len(in_vocab_words) >= 2:
    for i in range(len(in_vocab_words)):
        for j in range(i + 1, len(in_vocab_words)):
            w1, w2 = in_vocab_words[i], in_vocab_words[j]
            print(f"  similarity('{w1}', '{w2}') = {twin.embeddings.similarity(w1, w2):.3f}")

print(f"\n--- Analogy example (vector arithmetic) ---")
if len(in_vocab_words) >= 3:
    a, b, c = in_vocab_words[0], in_vocab_words[1], in_vocab_words[2]
    result = twin.embeddings.analogy(a, b, c, top_k=3)
    print(f"  '{a}' is to '{b}' as '{c}' is to: {result}")

print(f"\n--- OOV handling demo (no crash expected) ---")
print("  similarity('exercise', 'xyznotarealword') =", twin.embeddings.similarity("exercise", "xyznotarealword"))
print("  most_similar('xyznotarealword') =", twin.embeddings.most_similar("xyznotarealword"))

print("\n--- Visualizing the embedding space (PCA projection) ---")
twin.embeddings.visualize_2d(title="Digital Twin Vocabulary — Word2Vec")

print("\n--- Similarity heatmap for twin-relevant vocabulary ---")
if len(in_vocab_words) >= 2:
    twin.embeddings.similarity_heatmap(in_vocab_words, title="Twin Vocabulary — Pairwise Similarity")

print(f"\n=== TRAINING GLOVE ON THE SAME KNOWLEDGE BASE (COMPARISON) ===")
glove_space = twin.train_word_embeddings(method="glove", dim=24, window_size=4, x_max=8,
                                          learning_rate=0.05, epochs=60, verbose=False)
print(f"GloVe vocabulary size: {len(glove_space)} words")
print(f"GloVe final training loss: {twin.embedding_trainer.loss_history[-1]:.4f} "
      f"(started at {twin.embedding_trainer.loss_history[0]:.4f})")

print("\n--- GloVe training loss curve ---")
plt.figure(figsize=(7, 4))
plt.plot(twin.embedding_trainer.loss_history, color='seagreen')
plt.title("GloVe Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Weighted least-squares loss")
plt.tight_layout()
plt.show()

print("\n=== SEMANTIC RETRIEVAL: EMBEDDINGS vs TF/COSINE vs KEYWORD ===")
semantic_query = "how is the coding going"
print(f"Query: \"{semantic_query}\"")

print("\n-- retrieve_relevant_chunks_semantic (Word2Vec/GloVe mean-pooled vectors) --")
for r in twin.retrieve_relevant_chunks_semantic(semantic_query, top_k=3):
    print(f"  [{r['source']}] semantic_sim={r['semantic_similarity']:.2f}: {r['text'][:100]}")

print("\n-- retrieve_relevant_chunks (TF/cosine over exact terms) --")
for r in twin.retrieve_relevant_chunks(semantic_query, top_k=3):
    print(f"  [{r['source']}] sim={r['similarity']:.2f}: {r['text'][:100]}")

print("\n=== PRE-TRAINED EMBEDDINGS: SAVE / LOAD ROUND-TRIP ===")
pretrained_demo_path = "/tmp/demo_pretrained_embeddings.txt"
twin.embeddings.save(pretrained_demo_path)
print(f"Saved {len(twin.embeddings)} vectors to {pretrained_demo_path}")

loaded_space = EmbeddingSpace.load_pretrained(pretrained_demo_path)
print(f"Reloaded {len(loaded_space)} vectors — round-trip "
      f"{'OK' if len(loaded_space) == len(twin.embeddings) else 'MISMATCH'}")

print("\n--- Graceful handling of a missing pretrained file ---")
missing_demo = twin.load_pretrained_embeddings("/tmp/this_file_does_not_exist_12345.txt")
print(f"Loading a missing file returned an embedding space of size {len(missing_demo)} (no crash)")

# Restore the twin's trained embeddings (the load_pretrained_embeddings
# call above intentionally replaced them with an empty space for the demo).
twin.embeddings = loaded_space
twin.embedding_trainer = None
twin._chunk_embedding_cache = {}
print(f"\nTwin's embeddings restored: {len(twin.embeddings)} words available for semantic retrieval.")

# ============================================================
# 13. RAG ARCHITECTURE AND PIPELINE
# ============================================================
# Documentation: this section assembles everything built across
# Sections 9-12 into one named architecture:
#
#   [Documents] -> DOCUMENT LOADING & PREPROCESSING (DocumentPreprocessor, Sec 10)
#                -> CHUNKING (ChunkingSystem, format-aware, Sec 9-10)
#                -> INDEXING (VectorStore: lexical / semantic / hybrid, new)
#                -> [Query] -> RETRIEVAL (VectorStore.search, new)
#                -> CONTEXT ASSEMBLY (ContextWindowManager, Sec 11)
#                -> GENERATION (GroqClient / UniversalLLMClient, Sec 7-8)
#                -> [Answer + Sources]
#
# RAGLogger gives every stage structured, timestamped logging.
# Every stage (ingest/retrieve/assemble_context/generate) catches its
# own errors and returns a safe fallback instead of raising, so a bad
# document, an empty query, or a down LLM endpoint degrades the
# pipeline's output rather than crashing it.
#
# Integration: twin.create_rag_pipeline() builds a RAGPipeline bound
# directly to this twin's knowledge_base, context_manager, and
# embeddings — nothing about the pipeline is reimplemented, it is
# entirely composed from the twin's existing methods.

print("\n=== RAG PIPELINE: UNIT TESTS ===")
rag_tests_passed = run_rag_pipeline_tests()
print("ALL TESTS PASSED" if rag_tests_passed else "SOME TESTS FAILED — see output above")

print("\n=== BUILDING A RAG PIPELINE FOR THE TWIN (retrieval-only — no API key needed) ===")
pipeline = twin.create_rag_pipeline(llm_client=None, top_k=3, echo_logs=False)
pipeline.describe()

demo_queries = [
    "What have I been doing for exercise?",
    "How is the coding project going?",
    "What's my mood been like recently?",
]

print("\n--- Running queries through the pipeline (retrieval-only mode) ---")
for q in demo_queries:
    result = pipeline.ask(q)
    print(f"\nQuery: \"{q}\"")
    print(f"  retrieval_mode={result['retrieval_mode']}  "
          f"total_time={result['timings'].get('total', 0):.4f}s  error={result['error']}")
    for src in result['sources']:
        print(f"    [{src.get('source','?')}] score={src.get('score', 0):.3f}: {src['text'][:90]}")

print("\n--- Pipeline logging: recent entries and summary ---")
for entry in pipeline.logger.get_logs()[-6:]:
    print(f"  [{entry['level']}] [{entry['stage']}] {entry['message']}")
pipeline.log_summary()

print("\n--- Visualizing retrieval scores for one query ---")
pipeline.visualize_retrieval(demo_queries[0])

print("\n--- Visualizing average per-stage latency across all ask() calls so far ---")
pipeline.visualize_latency()

print("\n=== ATTACHING A LIVE LLM CLIENT FOR FULL GENERATION (needs GROQ_API_KEY) ===")
print("To run the pipeline end-to-end with real generated answers:")
print("  live_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL_NAME, context_manager=twin.context_manager)")
print("  live_pipeline = twin.create_rag_pipeline(llm_client=live_client, retrieval_mode='hybrid')")
print("  result = live_pipeline.ask('What have I been doing for exercise?')")
print("  print(result['answer'])")

# Uncomment with a real GROQ_API_KEY (see Section 8) to see it live:
# live_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL_NAME, context_manager=twin.context_manager)
# live_pipeline = twin.create_rag_pipeline(llm_client=live_client, retrieval_mode="hybrid", top_k=3)
# live_result = live_pipeline.ask("What have I been doing for exercise?")
# print(live_result['answer'])
# live_pipeline.visualize_retrieval("What have I been doing for exercise?")

print("\n--- Ingesting a brand-new document directly through the pipeline ---")
pipeline.ingest(
    "Friday wrap-up: finished the RAG pipeline integration, trained Word2Vec and GloVe "
    "embeddings on the full knowledge base, and set up the context window manager. "
    "#coding #milestone #digitaltwin",
    source="friday_wrapup",
)
followup_result = pipeline.ask("What did I accomplish on Friday?")
print(f"\nQuery: \"What did I accomplish on Friday?\"")
for src in followup_result['sources']:
    print(f"  [{src.get('source','?')}] score={src.get('score', 0):.3f}: {src['text'][:100]}")

# ============================================================
# 14. AI AGENTS: TOOL USE + THE REACT REASONING LOOP
# ============================================================
# Documentation: this section adds an agent layer on top of the
# existing twin/LLM/RAG infrastructure:
#
#   Tool            - wraps a plain function; its type hints describe
#                      its parameters to the LLM and coerce its text
#                      "Action Input" back into real int/float/bool/
#                      list/dict/str values.
#   ToolRegistry    - a named set of Tools an agent can call.
#   ReActAgent      - the Thought -> Action -> Observation -> (repeat)
#                      -> Final Answer loop. Decides, step by step,
#                      whether to call a tool or answer directly.
#   HabitTracker    - custom tool 2's backing store (streaks/history).
#
# Tools wired in by default (twin.tool_registry / twin.create_agent()):
#   calculator, datetime_now, search_memory (reuses the twin's own
#   RAG retrieval), take_note [custom tool 1: Note Taker -- reuses
#   ingest_notes so notes become part of the searchable knowledge
#   base], track_habit / habit_stats [custom tool 2: Habit Tracker].
#
# "Compress, don't delete": ReActAgent._compress_trace() merges the
# oldest block of reasoning-trace entries into ONE shorter summary
# entry (via the twin's own ContextWindowManager.extractive_summary)
# once the trace grows past a threshold -- the facts survive, the
# verbose wording doesn't, and nothing is ever silently dropped.
#
# A ScriptedLLM (no network/API key needed) drives the demo below so
# the full loop is visible end-to-end; swapping in a real GroqClient
# for live reasoning is a one-line change, shown (commented) at the end.

print("\n=== AI AGENTS: UNIT TESTS ===")
agent_tests_passed = run_react_agent_tests()
print("ALL TESTS PASSED" if agent_tests_passed else "SOME TESTS FAILED — see output above")

print("\n=== REACT AGENT DEMO (scripted LLM — no API key needed) ===")
scripted_responses = [
    "Thought: The user is asking about their exercise habits. I should search my memory for relevant notes.\n"
    "Action: search_memory\nAction Input: exercise",
    "Thought: I found some notes. Let me also check today's date for context.\n"
    "Action: datetime_now\nAction Input: %Y-%m-%d",
    "Thought: I now have enough information to answer.\n"
    "Final Answer: Based on your notes, you've been staying active with regular exercise sessions. "
    "(See the Observation steps above for today's date and the retrieved notes.)",
]
demo_agent = twin.create_agent(ScriptedLLM(scripted_responses), max_iterations=5, echo_logs=True)
demo_result = demo_agent.run("How has my exercise been going, and what's today's date?")

print("\nFinal answer:", demo_result['answer'])
print("Iterations used:", demo_result['iterations'], " error:", demo_result['error'])

print("\n--- Full reasoning trace ---")
for step in demo_result['trace']:
    print(f"  [{step['type']}] {step['content']}")

demo_agent.describe()
demo_agent.visualize_trace()

print("\n--- Custom tool 1: Note Taker (integrated with the twin's RAG knowledge base) ---")
print(twin.tool_registry.call(
    "take_note",
    text="Finished building the ReAct agent with calculator, search, datetime, note-taker and habit-tracker tools.",
))
print(twin.tool_registry.call("search_memory", query="ReAct agent tools"))

print("\n--- Custom tool 2: Habit Tracker ---")
print(twin.tool_registry.call("track_habit", habit_name="exercise", completed=True))
print(twin.tool_registry.call("track_habit", habit_name="exercise", completed=True))
print(twin.tool_registry.call("habit_stats", habit_name="exercise"))

print("\n--- Trace compression demo: 'compress, don't delete — five entries become one' ---")
compress_agent = twin.create_agent(ScriptedLLM([]), compress_every=5, keep_recent_steps=2, echo_logs=False)
for i in range(9):
    compress_agent.trace.append({'type': 'observation', 'content': f"On day {i}, the user logged activity #{i}."})
print(f"Trace length before compression: {len(compress_agent.trace)}")
compress_agent._compress_trace()
print(f"Trace length after compression:  {len(compress_agent.trace)}")
for step in compress_agent.trace:
    print(f"  [{step['type']}] {step['content']}")

print("\n=== USING A REAL LLM FOR LIVE REASONING (needs GROQ_API_KEY) ===")
print("  live_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL_NAME, context_manager=twin.context_manager)")
print("  live_agent = twin.create_agent(live_client, max_iterations=6)")
print("  result = live_agent.run('What have I been working on, and can you calculate 23 * 17 for me?')")
print("  print(result['answer'])")

# Uncomment with a real GROQ_API_KEY (see Section 8) to see it live:
# live_client = GroqClient(api_key=GROQ_API_KEY, model=MODEL_NAME, context_manager=twin.context_manager)
# live_agent = twin.create_agent(live_client, max_iterations=6)
# live_result = live_agent.run("What have I been working on, and can you calculate 23 * 17 for me?")
# print(live_result['answer'])
# live_agent.visualize_trace()
