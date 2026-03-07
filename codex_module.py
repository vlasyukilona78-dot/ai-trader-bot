import os
import time
from functools import lru_cache
from typing import Dict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import openai as legacy_openai
except ImportError:
    legacy_openai = None

_EXPLAIN_MODEL = os.getenv("OPENAI_EXPLAIN_MODEL", "gpt-5-mini")
_CODE_MODEL = os.getenv("OPENAI_CODE_MODEL", "gpt-5-codex")
_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
_RETRY_DELAY = float(os.getenv("OPENAI_RETRY_DELAY", "1.0"))


def _get_api_key() -> str:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("openai.api_key")
        or os.getenv("OPENAI_APIKEY")
        or ""
    )


@lru_cache(maxsize=1)
def _get_client():
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    if OpenAI is not None:
        return OpenAI(api_key=api_key)

    if legacy_openai is not None:
        legacy_openai.api_key = api_key
        return legacy_openai

    raise RuntimeError("OpenAI SDK is not installed")


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for chunk in content:
                text = getattr(chunk, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()

    raise RuntimeError("Model response did not contain text output")


def _call_responses_api(model: str, instructions: str, prompt: str, max_output_tokens: int) -> str:
    client = _get_client()
    if not hasattr(client, "responses"):
        raise RuntimeError("Installed OpenAI SDK does not support Responses API")

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    return _extract_response_text(response)


def _call_legacy_chat_api(model: str, instructions: str, prompt: str, max_output_tokens: int) -> str:
    client = _get_client()
    if legacy_openai is None:
        raise RuntimeError("Legacy OpenAI SDK is unavailable")

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_output_tokens,
        )
        return response.choices[0].message.content.strip()

    response = legacy_openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_output_tokens,
    )
    return response.choices[0].message["content"].strip()


def _call_openai_text(model: str, instructions: str, prompt: str, max_output_tokens: int = 512) -> str:
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _call_responses_api(model, instructions, prompt, max_output_tokens)
        except Exception as exc:
            last_exc = exc
            try:
                return _call_legacy_chat_api(model, instructions, prompt, max_output_tokens)
            except Exception as legacy_exc:
                last_exc = legacy_exc
                time.sleep(_RETRY_DELAY * attempt)
    raise last_exc


def explain_signal(symbol: str, indicators: Dict[str, float], probability: float, direction: str) -> str:
    """Return a short explanation of a trading signal."""
    indicators_text = ", ".join(f"{key}={value:.4g}" for key, value in indicators.items()) or "no data"
    instructions = (
        "You are a crypto analyst and trader. Reply in Russian, keep it short, practical, and direct. "
        "Give 2-3 short sentences covering signal strength, key risks, and one confirmation step."
    )
    prompt = (
        f"Signal: {direction} on {symbol}. "
        f"Estimated win probability: {probability:.3f}. "
        f"Indicators: {indicators_text}."
    )

    try:
        return _call_openai_text(
            model=_EXPLAIN_MODEL,
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=180,
        )
    except Exception as exc:
        return f"(Failed to get model explanation: {exc})"


def optimize_strategy(current_code: str, feedback: str) -> str:
    """Request an updated strategy implementation and return code only."""
    instructions = (
        "You are a senior Python engineer. Return only valid Python code, with no markdown and no explanation. "
        "Fix logic issues, guard against divide-by-zero, check for minimum data length, and add a short self-test."
    )
    prompt = (
        "Below is a trading strategy file and the requested changes. "
        "Return only the full corrected Python file.\n\n"
        f"Requested changes:\n{feedback}\n\n"
        f"Current code:\n{current_code}"
    )

    try:
        return _call_openai_text(
            model=_CODE_MODEL,
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=4000,
        )
    except Exception:
        return current_code


def generate_strategy(description: str) -> str:
    """Generate a strategy implementation and return code only."""
    instructions = (
        "You are a senior Python engineer. Return only valid Python code, with no markdown and no explanation. "
        "Include compute_indicators(df), generate_signals(df), backtest(df), and a small self-test block."
    )
    prompt = (
        f"Generate a trading strategy from this description: {description}. "
        "Use pandas or numpy, keep calculations safe, and include a simple self-test on synthetic data."
    )

    try:
        return _call_openai_text(
            model=_CODE_MODEL,
            instructions=instructions,
            prompt=prompt,
            max_output_tokens=4000,
        )
    except Exception as exc:
        return f"# Strategy generation error: {exc}\n"
