"""Unified LLM bridge supporting OpenAI models and Gemini 3.1 Pro via Antigravity CLI."""

import json
import os
import subprocess
import time
from typing import Any, Dict, Optional


def call_gemini_cli(
    prompt: str,
    model: str = "gemini-3.1-pro-high",
    timeout: int = 300,
) -> Optional[Dict[str, Any]]:
    """Invoke Gemini 3.1 Pro via the local Antigravity (agy) CLI."""
    # Ensure model identifier matches agy models
    target_model = "gemini-3.1-pro-high" if "3.1" in model or "pro" in model else model
    cmd = [
        "/Users/VJ/.local/bin/agy",
        "--model",
        target_model,
        "-p",
        prompt,
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            print(f"[Warning] agy CLI exited with code {r.returncode}: {r.stderr[:200]}")
            return None

        out = r.stdout
        i, j = out.find("{"), out.rfind("}")
        if i == -1 or j <= i:
            print(f"[Warning] No JSON found in agy output: {out[:200]}")
            return None
        return json.loads(out[i : j + 1])
    except Exception as e:
        print(f"[Warning] Exception calling agy CLI: {e}")
        return None


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """Unified entry point for structured JSON calls across OpenAI and Gemini."""
    if "gemini" in model.lower():
        full_prompt = (
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
            f"USER QUERY:\n{user_prompt}\n\n"
            "CRITICAL: Output strictly a single valid JSON object. No conversational markdown, no code fences, no commentary outside the JSON."
        )
        return call_gemini_cli(full_prompt, model=model)

    # Default to OpenAI API
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"[Warning] OpenAI API call failed: {e}")
        return None
