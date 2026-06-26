"""
Async multi-provider LLM router.

Routes chat-completion requests to Groq, OpenAI, Gemini, Claude, or a local
OpenClaw-compatible endpoint (Ollama/LM Studio). Used by the analysis engine.
"""

import os
import asyncio
import logging
import httpx
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

MODELS = {
    "groq": {
        "default": "llama-3.3-70b-versatile",
        "fast":    "llama-3.1-8b-instant",
    },
    "openai": {
        "default": "gpt-4o",
        "fast":    "gpt-4o-mini",
    },
    "gemini": {
        "default": "gemini-1.5-pro",
        "fast":    "gemini-1.5-flash",
    },
    "claude": {
        "default": "claude-sonnet-4-20250514",
        "fast":    "claude-haiku-4-5-20251001",
    },
    "openclaw": {
        "default": "llama-3-70b",
        "fast":    "llama-3-8b",
    },
}

COMPLEX_KEYWORDS = [
    "constitutional", "evidence", "bail", "murder", "qatl",
    "custody", "divorce", "contract", "fraud", "section",
    "appeal", "high court", "supreme court", "writ",
]


def smart_select(query: str, preferred: str = "groq") -> tuple:
    """
    Pick a provider and model tier based on query complexity.

    Short or simple queries use faster/cheaper models; longer queries or those
    containing legal complexity keywords use the default (stronger) model.

    Args:
        query: User or document text used to estimate complexity.
        preferred: Provider the caller wants to prioritize when credentials exist.

    Returns:
        Tuple of (provider_name, model_name).
    """
    is_complex = (
        len(query.split()) > 100 or
        any(kw in query.lower() for kw in COMPLEX_KEYWORDS)
    )

    if preferred == "openclaw":
        return "openclaw", MODELS["openclaw"]["default"]

    keys = {
        "groq":   "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }

    # Complex cases favour stronger models first; simple cases favour speed/cost.
    order = ["claude", "openai", "groq", "gemini"] if is_complex else ["groq", "gemini", "openai", "claude"]
    order = [preferred] + [p for p in order if p != preferred]

    for p in order:
        if p == "openclaw" or os.getenv(keys.get(p, "")):
            mode  = "default" if is_complex else "fast"
            model = MODELS[p][mode]
            return p, model

    return preferred, MODELS[preferred]["default"]


async def call_llm_async(
    system: str,
    user: str,
    provider: str        = "groq",
    model: Optional[str] = None,
    temperature: float   = 0.1,
    max_tokens: int      = 3000,
) -> str:
    """
    Send a chat completion request to the specified LLM provider.

    Args:
        system: System prompt defining model behaviour.
        user: User message / analysis prompt.
        provider: Provider key (groq, openai, gemini, claude, openclaw).
        model: Optional model name; falls back to provider default.
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens in the completion response.

    Returns:
        Raw text content from the model's first completion choice.

    Raises:
        ValueError: If provider is not recognized.
        httpx.HTTPStatusError: If a local OpenClaw request fails.
    """
    provider = provider.lower()
    selected = model or MODELS.get(provider, {}).get("default", "unknown")

    logger.info(f"[ASYNC] {provider}/{selected}")

    if provider == "openclaw":
        # OpenAI-compatible local endpoint; default port 11434 avoids clashing with uvicorn on 8000.
        url = os.getenv("OPENCLAW_BASE_URL", "http://localhost:11434/v1/chat/completions")
        headers = {"Authorization": f"Bearer {os.getenv('OPENCLAW_API_KEY', 'local-key')}"}
        payload = {
            "model": selected,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']

    elif provider == "groq":
        from groq import AsyncGroq
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        r = await client.chat.completions.create(
            model=selected,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content

    elif provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        r = await client.chat.completions.create(
            model=selected,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        m = genai.GenerativeModel(
            model_name=selected,
            system_instruction=system,
            generation_config={"temperature":temperature, "max_output_tokens":max_tokens})
        loop = asyncio.get_event_loop()
        r    = await loop.run_in_executor(None, m.generate_content, user)
        return r.text

    elif provider == "claude":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        r = await client.messages.create(
            model=selected, system=system,
            messages=[{"role":"user","content":user}],
            temperature=temperature, max_tokens=max_tokens)
        return r.content[0].text

    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm(system: str, user: str, provider: str = "groq",
             model: str = None, temperature: float = 0.1,
             max_tokens: int = 3000) -> str:
    """
    Synchronous wrapper around call_llm_async.

    Args:
        system: System prompt.
        user: User message.
        provider: Provider key.
        model: Optional model override.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        Raw completion text from the model.
    """
    return asyncio.run(call_llm_async(system, user, provider, model, temperature, max_tokens))


def check_providers() -> dict:
    """
    Report which LLM providers appear configured in the environment.

    Returns:
        Dict mapping provider name to availability flag and default model name.
        OpenClaw is always marked available (local server may be running without env vars).
    """
    keys = {
        "groq":   "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "openclaw": "OPENCLAW_BASE_URL",
    }
    return {
        p: {"available": bool(os.getenv(k)) or p == "openclaw",
            "default_model": MODELS[p]["default"]}
        for p, k in keys.items()
    }
