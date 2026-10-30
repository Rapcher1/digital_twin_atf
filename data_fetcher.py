import requests

def fetch_web_summary(query: str, max_chars: int = 800):
    if not query or not str(query).strip():
        return ""

    try:
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "kl": "en-us",
        }
        response = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        abstract = (data.get("Abstract") or "").strip()
        if abstract:
            return abstract[:max_chars]

        related = data.get("RelatedTopics") or []
        if isinstance(related, list):
            texts = []
            for item in related[:3]:
                if isinstance(item, dict):
                    text = item.get("Text")
                    if text:
                        texts.append(text)
            joined = " ".join(texts)
            if joined:
                return joined[:max_chars]

        return ""
    except Exception:
        return ""
