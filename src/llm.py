"""Thin LLM client for Groq and Gemini with a committed on-disk cache.

Every (provider, model, prompt, params) call is hashed and stored in cache/<name>.jsonl.
With OFFLINE=1, a cache miss raises instead of calling the API, which lets reviewers
reproduce every number without API keys.
"""
import hashlib
import json
import os
import re
import time

from dotenv import load_dotenv

from config import CACHE_DIR, MIN_INTERVAL_S, ROOT

load_dotenv(ROOT / ".env")

_last_call = {}


class CacheMiss(RuntimeError):
    pass


class LLM:
    def __init__(self, provider, model, cache_name):
        self.provider, self.model = provider, model
        self.offline = os.getenv("OFFLINE") == "1"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f"{cache_name}.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry["response"]
        self._client = None

    def _key(self, system, user, temperature, max_tokens):
        payload = json.dumps([self.provider, self.model, system, user, temperature, max_tokens])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(self, system, user, temperature=0.0, max_tokens=700):
        key = self._key(system, user, temperature, max_tokens)
        if key in self.cache:
            return self.cache[key]
        if self.offline:
            raise CacheMiss(f"OFFLINE=1 and no cached response in {self.cache_path.name}")

        response = self._call_with_retry(system, user, temperature, max_tokens)
        self.cache[key] = response
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": self.model, "response": response}) + "\n")
        return response

    def complete_json(self, system, user, **kwargs):
        return parse_json(self.complete(system, user, **kwargs))

    def _call_with_retry(self, system, user, temperature, max_tokens, attempts=6):
        for attempt in range(attempts):
            self._pace()
            try:
                if self.provider == "groq":
                    return self._groq(system, user, temperature, max_tokens)
                if self.provider == "gemini":
                    return self._gemini(system, user, temperature, max_tokens)
                raise ValueError(f"Unknown provider {self.provider}")
            except Exception as exc:  # rate limits / transient server errors
                message = str(exc).lower()
                retryable = any(s in message for s in ("429", "rate", "quota", "503", "500", "overloaded", "timeout", "unavailable"))
                if not retryable or attempt == attempts - 1:
                    raise
                wait = min(90, 5 * 2 ** attempt)
                print(f"  [{self.provider}] retryable error, sleeping {wait}s: {str(exc)[:120]}")
                time.sleep(wait)

    def _pace(self):
        gap = MIN_INTERVAL_S.get(self.provider, 0)
        elapsed = time.time() - _last_call.get(self.provider, 0)
        if elapsed < gap:
            time.sleep(gap - elapsed)
        _last_call[self.provider] = time.time()

    def _groq(self, system, user, temperature, max_tokens):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            # gpt-oss models reason before answering; low effort keeps token use inside free-tier limits
            **({"reasoning_effort": "low"} if "gpt-oss" in self.model else {}),
        )
        return resp.choices[0].message.content

    def _gemini(self, system, user, temperature, max_tokens):
        from google.genai import types
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text


def parse_json(text):
    """Parse a JSON object from a model response, tolerating code fences or stray prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
