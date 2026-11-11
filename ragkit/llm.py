import time

import httpx
from . import config


class LLM:
    """Minimal OpenAI-compatible chat client. Swap via env vars."""

    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or config.LLM_API_KEY
        self.model = model or config.LLM_MODEL

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY not set")
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        for attempt in range(6):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions",
                               headers={"Authorization": f"Bearer {self.api_key}"}, json=payload, timeout=120)
            except httpx.TransportError as e:
                if attempt == 5:
                    raise
                print(f"  [llm] network error ({type(e).__name__}), retry in {2 ** attempt}s", flush=True)
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503) and attempt < 5:
                # cap the wait: a huge retry-after means a daily quota is exhausted, which is better surfaced than slept through
                wait = min(float(r.headers.get("retry-after", 2 ** attempt * 2)), 90.0)
                print(f"  [llm] HTTP {r.status_code}, waiting {wait:.0f}s (attempt {attempt + 1}/6)", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        return r.json()["choices"][0]["message"]["content"]
