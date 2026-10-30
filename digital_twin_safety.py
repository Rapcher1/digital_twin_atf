import re
import time
from collections import deque
from datetime import datetime


class SafetyViolationLogger:
    """Records safety-related events for auditing and debugging."""

    def __init__(self):
        self.events = []

    def log(self, event_type, severity, detail, metadata=None):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "detail": detail,
            "metadata": metadata or {},
        }
        self.events.append(record)
        return record

    def count(self, event_type=None):
        if event_type is None:
            return len(self.events)
        return sum(1 for e in self.events if e["event_type"] == event_type)


class RateLimiter:
    """Simple sliding-window rate limiter to prevent abuse."""

    def __init__(self, max_calls=5, window_seconds=60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def allow(self):
        now = time.time()
        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] <= cutoff:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_calls:
            return False

        self.timestamps.append(now)
        return True

    def reset(self):
        self.timestamps.clear()


class PIIProtector:
    """Detects and redacts common personal information types."""

    PATTERNS = [
        (
            "email",
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        ),
        (
            "phone",
            re.compile(
                r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{3,4}"
            ),
        ),
        (
            "dob",
            re.compile(
                r"\b(?:DOB|Date of Birth|born on|birth date)\s*[:\-]?\s*"
                r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
                r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
                r"(?:\s+\d{2,4})?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "address",
            re.compile(
                r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+\s(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Terrace|Ter|Parkway|Pkwy|Circle|Cir)\b"
                r"(?:,\s*(?:Apt|Apartment|Suite|Unit|#)\s*\w+[A-Za-z0-9-]*)?"
                r"(?:,\s*[A-Za-z .\-]+)?(?:,\s*(?:[A-Z]{2}|[A-Za-z]+))?",
                re.IGNORECASE,
            ),
        ),
        (
            "ssn",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ),
    ]

    @classmethod
    def detect(cls, text):
        text = text or ""
        found = {}
        for key, pattern in cls.PATTERNS:
            matches = pattern.findall(text)
            normalized = []
            for item in matches:
                if isinstance(item, tuple):
                    candidate = " ".join(part for part in item if part)
                else:
                    candidate = str(item).strip()
                if candidate:
                    normalized.append(candidate)
            if normalized:
                found[key] = sorted(set(normalized))
        return found

    @classmethod
    def redact(cls, text):
        text = text or ""
        redacted = text
        for _, pattern in cls.PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted


class ContentFilter:
    """Domain-specific filters for digital-twin safety."""

    DOMAIN_RULES = {
        "self_harm": [
            "kill myself",
            "hurt myself",
            "commit suicide",
            "end my life",
            "self-harm",
            "cut myself",
            "jump off",
            "poison myself",
        ],
        "violent_instructions": [
            "attack someone",
            "assault someone",
            "stab someone",
            "shoot someone",
            "burn someone",
            "fight club",
            "hurt another person",
        ],
        "hate_or_extremism": [
            "kill all immigrants",
            "hate speech",
            "attack a group",
            "dehumanize people",
            "extremist ideology",
            "ethnic cleansing",
        ],
        "sexual_explicit": [
            "explicit sexual content",
            "pornographic",
            "nude images",
            "sexually explicit",
            "graphic sexual",
        ],
        "illegal_activity": [
            "hack a bank",
            "steal identities",
            "make a bomb",
            "avoid taxes illegally",
            "launder money",
        ],
        "stalking_doxxing": [
            "find someone’s home address",
            "track their phone",
            "doxx",
            "stalk",
            "monitor their movements",
        ],
        "medical_misinfo": [
            "cure cancer with one trick",
            "stop insulin with this",
            "all vaccines are poison",
            "medical miracle",
        ],
    }

    @classmethod
    def scan(cls, text):
        text = (text or "").lower()
        matches = []
        for category, phrases in cls.DOMAIN_RULES.items():
            if any(phrase in text for phrase in phrases):
                matches.append(category)
        return matches


class OutputValidator:
    """Ensures generated output is coherent, safe, and not empty."""

    def __init__(self, max_characters=800):
        self.max_characters = max_characters

    def validate(self, text, context=""):
        text = text or ""
        issues = []

        if not text.strip():
            issues.append("empty_response")
        if len(text) > self.max_characters:
            issues.append("too_long")
        if re.search(r"\b(?:\d{1,3}\s*%|\d+\s*out\s*of\s*\d+)\b", text, re.IGNORECASE) and "according to" not in context.lower():
            issues.append("unsupported_confidence")
        if re.search(r"\b(?:100%|guaranteed|definitely|certainly)\b", text, re.IGNORECASE):
            issues.append("overconfident_claim")

        return {"valid": not issues, "issues": issues}


class DigitalTwinSafetySystem:
    """Combined safety layer for privacy, filtering, validation, and rate management."""

    def __init__(self, max_calls=5, window_seconds=60):
        self.rate_limiter = RateLimiter(max_calls=max_calls, window_seconds=window_seconds)
        self.logger = SafetyViolationLogger()
        self.content_filter = ContentFilter
        self.pii = PIIProtector
        self.validator = OutputValidator()

    def check_request(self, text, context=""):
        result = {
            "blocked": False,
            "reasons": [],
            "pii": {},
            "content_categories": [],
            "sensitive_topic": False,
        }

        if not self.rate_limiter.allow():
            result["blocked"] = True
            result["reasons"].append("rate_limited")
            self.logger.log("rate_limit_violation", "high", "Request exceeded rate limit", {"context": context})
            return result

        pii_matches = self.pii.detect(text)
        if pii_matches:
            result["pii"] = pii_matches
            result["reasons"].append("pii_detected")

        content_categories = self.content_filter.scan(text)
        if content_categories:
            result["content_categories"] = content_categories
            result["blocked"] = True
            result["reasons"].append("content_blocked")
            self.logger.log("content_blocked", "high", "Content filter triggered", {"categories": content_categories})

        sensitive = any(k in text.lower() for k in ["self-harm", "suicide", "hurt myself", "mental health crisis", "abuse", "violence", "emergency"])
        if sensitive:
            result["sensitive_topic"] = True
            result["reasons"].append("sensitive_topic")

        return result

    def redact_input(self, text):
        return self.pii.redact(text or "")

    def add_disclaimer(self, text):
        disclaimer = (
            "\n\nDisclaimer: This is general guidance, not crisis counseling. "
            "If this is an immediate emergency or you feel at risk of harming yourself or someone else, "
            "contact emergency services or a local crisis hotline immediately."
        )
        return text.rstrip() + disclaimer if text.strip() else disclaimer

    def validate_output(self, text, context=""):
        validation = self.validator.validate(text, context)
        if not validation["valid"]:
            self.logger.log("output_validation_failed", "medium", "Generated output failed validation", {"issues": validation["issues"]})
        return validation

    def sanitize_response(self, response, sensitive):
        response = response.strip()
        if not response:
            response = "I can help with general wellness, routines, and safe planning."
        if sensitive:
            response = self.add_disclaimer(response)
        return response


class SafeDigitalTwin:
    """Improved twin that includes privacy, content filtering, and output safety."""

    def __init__(self, name="Aster"):
        self.name = name
        self.safety = DigitalTwinSafetySystem()

    def _generic_reply(self, prompt):
        prompt = (prompt or "").strip()
        if not prompt:
            return "I’m here to help with routines, memory, wellness, and planning."
        if "plan" in prompt.lower():
            return "I can help you build a simple, balanced plan with sleep, work, exercise, and recovery time."
        if "wellness" in prompt.lower() or "health" in prompt.lower():
            return "I can suggest healthy routines, realistic habits, and reminders that support your energy and focus."
        return "I can help you reflect on your goals, priorities, and routines in a practical, supportive way."

    def generate(self, user_input, context=""):
        safety_state = self.safety.check_request(user_input, context)
        if safety_state["blocked"]:
            self.safety.logger.log("blocked_request", "high", "Blocked unsafe user request", {"reasons": safety_state["reasons"]})
            response = "I can’t assist with that kind of request. If this is a safety issue, contact emergency services or a local crisis hotline."
            if safety_state["sensitive_topic"]:
                response = self.safety.add_disclaimer(response)
            return {
                "response": response,
                "safe": False,
                "safety": safety_state,
            }

        sanitized_input = self.safety.redact_input(user_input)
        response = self._generic_reply(sanitized_input)

        if self.safety.content_filter.scan(response):
            response = "I can help with general wellbeing, routines, and safe planning without engaging with risky or harmful topics."

        validation = self.safety.validate_output(response, context)
        if not validation["valid"]:
            response = "I can help with general planning, routines, and healthy habits while keeping your information private and safe."

        if safety_state["sensitive_topic"]:
            response = self.safety.add_disclaimer(response)

        return {
            "response": response,
            "safe": True,
            "safety": safety_state,
            "validation": validation,
        }


def run_digital_twin_safety_suite():
    """Ten test cases covering capabilities, edge cases, and safety failures."""
    results = []

    def check(name, condition):
        ok = bool(condition)
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    twin = SafeDigitalTwin()
    safety = DigitalTwinSafetySystem(max_calls=2, window_seconds=60)

    # 1. PII detection catches email and phone
    pii_text = "Contact me at 555-123-4567 or me@example.com."
    pii_hits = PIIProtector.detect(pii_text)
    check("PII detection catches email and phone patterns", {"phone", "email"}.issubset(set(pii_hits.keys())))

    # 2. address and DOB detection
    address_text = "DOB: 04/12/1990. Address: 123 Main St, Apt 6B, Boston, MA."
    dob_hits = PIIProtector.detect(address_text)
    check("DOB and address patterns are detected", {"dob", "address"}.issubset(set(dob_hits.keys())))

    # 3. content filter blocks self-harm request
    content_hits = ContentFilter.scan("I want to hurt myself tonight and skip my medication.")
    check("Content filtering blocks self-harm instructions", "self_harm" in content_hits)

    # 4. content filter blocks violent instructions
    violent_hits = ContentFilter.scan("Tell me how to attack someone without getting caught.")
    check("Content filtering blocks violent requests", "violent_instructions" in violent_hits)

    # 5. rate limiter blocks excess requests
    limiter = RateLimiter(max_calls=2, window_seconds=60)
    first = limiter.allow()
    second = limiter.allow()
    third = limiter.allow()
    check("Rate limiting blocks repeated abuse", first and second and (not third))

    # 6. PII redaction protects privacy
    original = "Email: jane.doe@example.com. Phone: 415-555-0102."
    redacted = PIIProtector.redact(original)
    check("PII redaction masks private data before processing", "[REDACTED]" in redacted and "jane.doe@example.com" not in redacted)

    # 7. sensitive topic gets disclaimer
    result = twin.generate("Help me plan how to hurt myself")
    check("Sensitive topics always include a disclaimer", result["safe"] is False and "Disclaimer" in result["response"])

    # 8. output validation catches empty or malformed output
    validator = OutputValidator()
    validation = validator.validate("")
    check("Output validation rejects empty responses", validation["issues"] and "empty_response" in validation["issues"])

    # 9. logging records safety violations
    safety.logger.log("content_blocked", "high", "blocked unsafe request")
    check("Safety logging captures violations", safety.logger.count("content_blocked") >= 1)

    # 10. hallucination guard reduces unsupported claims
    response = "I reviewed your entire history and 92% of users in this app are depressed."
    validation = safety.validate_output(response, context="user asked a general question")
    check("Output validation flags unsupported confidence claims", "unsupported_confidence" in validation["issues"] or "overconfident_claim" in validation["issues"])

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} safety tests passed.")
    return passed == total


if __name__ == "__main__":
    run_digital_twin_safety_suite()
    print("Digital twin safety suite completed.")
    

# The twin now includes multiple safety layers:
# - PII detection and redaction
# - Custom content filtering for the digital twin domain
# - Rate limiting
# - Logging of violations
# - Output validation and disclaimers for sensitive topics
# - Privacy-preserving defaults and safe fallback behavior
