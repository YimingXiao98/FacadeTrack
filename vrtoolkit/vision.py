"""
OpenAI Vision/Text integration helpers.

Note: Ensure `OPENAI_API_KEY` is set in the environment before using.
This module intentionally does not embed any API keys.
"""

import base64
import os
from typing import Tuple

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional
    openai = None  # Allows the repo to function without the dependency


DEFAULT_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o")
DEFAULT_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o")


def call_vision_model(image_path: str, model: str = DEFAULT_VISION_MODEL) -> Tuple[str, int | None]:
    if openai is None:
        raise RuntimeError("openai package is not installed. `pip install openai`." )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    openai.api_key = api_key

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Analyze the image and answer with a JSON object in a strict schema. "
        "Only output the JSON object, nothing else."
    )

    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output only strict JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            },
        ],
        max_tokens=200,
        temperature=0,
    )
    usage = getattr(getattr(resp, "usage", None), "total_tokens", None)
    return resp.choices[0].message.content, usage


def call_text_model(vision_output_str: str, model: str = DEFAULT_TEXT_MODEL) -> Tuple[str, int | None]:
    if openai is None:
        raise RuntimeError("openai package is not installed. `pip install openai`." )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    openai.api_key = api_key

    prompt = (
        "You are an expert in post-disaster building occupancy assessment. "
        "Decide 'Occupied' or 'Not Occupied' using the provided JSON attributes. "
        "Output only one word: Occupied or Not Occupied.\n\n"
        f"Attributes: {vision_output_str}"
    )
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Classify occupancy concisely."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5,
        temperature=0,
    )
    usage = getattr(getattr(resp, "usage", None), "total_tokens", None)
    return resp.choices[0].message.content.strip(), usage

