"""Audio-trait extraction for hybrid voice-pool retrieval.

Runtime extraction is intentionally fail-loud: no heuristic or fake model is used
when a real audio model is requested. Tests exercise prompt/schema plumbing with
mocked clients; production should configure an OpenAI-compatible client and key.
"""

from __future__ import annotations

import base64
import io
import json
import os
import wave
from pathlib import Path
from typing import Any

from .schema import AegisVoxContractError
from .voice_pool import normalize_audio_trait_value


class AudioTraitExtractionError(RuntimeError):
    """Raised when real audio trait extraction cannot be performed."""


AUDIO_TRAIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "language": {"type": ["string", "null"]},
        "accent_group": {"type": ["string", "null"]},
        "geography_region": {"type": ["string", "null"]},
        "tone": {"type": ["string", "null"]},
        "energy": {"type": ["string", "null"], "enum": ["low", "medium", "high", "calm", "animated", None]},
        "speaking_rate": {"type": ["string", "null"], "enum": ["slow", "moderate", "fast", None]},
        "pitch_band": {"type": ["string", "null"], "enum": ["low", "mid", "high", None]},
        "gender_presentation": {"type": ["string", "null"], "enum": ["female", "male", "unknown", None]},
        "vocal_weight": {"type": ["string", "null"], "enum": ["thin", "medium", "thick", "unknown", None]},
        "resonance": {"type": ["string", "null"], "enum": ["light", "balanced", "deep", "unknown", None]},
        "voice_maturity": {"type": ["string", "null"], "enum": ["youthful", "adult", "mature", "unknown", None]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": [
        "language",
        "accent_group",
        "geography_region",
        "tone",
        "energy",
        "speaking_rate",
        "pitch_band",
        "gender_presentation",
        "vocal_weight",
        "resonance",
        "voice_maturity",
        "confidence",
        "evidence",
    ],
}


AUDIO_TRAIT_PROMPT = """You are extracting operational voice-matching traits for a privacy-preserving voice-pool search.
Return JSON only. Do not transcribe private content. Do not infer identity, ethnicity, health status, or protected-class facts.
Estimate only audible matching traits useful for selecting a pseudo-speaker: language, broad accent_group, broad geography_region if audible, tone, energy, speaking_rate, pitch_band, gender_presentation if apparent, vocal_weight (thin/medium/thick), resonance (light/balanced/deep), and voice_maturity (youthful/adult/mature).
Use unknown/null when uncertain. geography_region must be broad and operational (e.g. south_asia, north_america, uk_ireland, east_asia, middle_east, unknown), not a precise location. Do not label an adult or older speaker as youthful unless the voice clearly sounds child/teen-like; this field is only to avoid child-like pseudo-speaker mismatches.
"""


def normalize_audio_traits(payload: dict[str, Any]) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise AegisVoxContractError("audio trait extraction payload must be an object")
    traits: dict[str, str] = {}
    for key in [
        "language",
        "accent_group",
        "geography_region",
        "tone",
        "energy",
        "speaking_rate",
        "pitch_band",
        "gender_presentation",
        "vocal_weight",
        "resonance",
        "voice_maturity",
    ]:
        normalized = normalize_audio_trait_value(key, payload.get(key))
        if normalized and normalized != "unknown":
            traits[key] = normalized
    return traits


def _audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"wav", "mp3", "m4a", "ogg", "webm", "flac"}:
        return suffix
    return "wav"


def _audio_bytes_for_trait_request(path: Path) -> tuple[bytes, str, dict[str, Any]]:
    audio_format = _audio_format(path)
    max_seconds = float(os.environ.get("AEGISVOX_AUDIO_TRAIT_MAX_SECONDS", "30"))
    if audio_format == "wav" and max_seconds > 0:
        try:
            with wave.open(str(path), "rb") as reader:
                max_frames = min(reader.getnframes(), max(1, int(round(max_seconds * reader.getframerate()))))
                with io.BytesIO() as buffer:
                    with wave.open(buffer, "wb") as writer:
                        writer.setnchannels(reader.getnchannels())
                        writer.setsampwidth(reader.getsampwidth())
                        writer.setframerate(reader.getframerate())
                        writer.writeframes(reader.readframes(max_frames))
                    data = buffer.getvalue()
                return data, "wav", {
                    "trait_audio_duration_sec": round(max_frames / float(reader.getframerate()), 3),
                    "trait_audio_source_duration_sec": round(reader.getnframes() / float(reader.getframerate()), 3),
                    "trait_audio_truncated": max_frames < reader.getnframes(),
                }
        except Exception:
            # Fall back to raw file bytes for non-standard WAV-like payloads used
            # in tests or externally prepared files.
            pass
    data = path.read_bytes()
    max_bytes = int(os.environ.get("AEGISVOX_AUDIO_TRAIT_MAX_BYTES", "20000000"))
    if max_bytes > 0 and len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True
    else:
        truncated = False
    return data, audio_format, {"trait_audio_truncated": truncated}


def build_gpt_audio_trait_messages(audio_path: Path) -> list[dict[str, Any]]:
    if not audio_path.exists():
        raise AudioTraitExtractionError(f"query audio not found: {audio_path}")
    data, audio_format, _metadata = _audio_bytes_for_trait_request(audio_path)
    encoded = base64.b64encode(data).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": AUDIO_TRAIT_PROMPT},
                {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}},
            ],
        }
    ]


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                raise AudioTraitExtractionError("audio trait model did not return valid JSON") from exc
        else:
            raise AudioTraitExtractionError("audio trait model did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise AudioTraitExtractionError("audio trait model returned non-object JSON")
    return payload


def extract_audio_traits_with_openai(
    audio_path: Path,
    *,
    model: str = "gpt-audio-1.5",
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "AEGISVOX_OPENAI_BASE_URL",
    client: Any | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Extract structured search traits using a real OpenAI audio model."""

    if client is None:
        api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AudioTraitExtractionError(f"missing {api_key_env}/OPENAI_API_KEY for audio trait extraction")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise AudioTraitExtractionError("openai package is required for audio trait extraction") from exc
        base_url = os.environ.get(base_url_env) or os.environ.get("OPENAI_BASE_URL")
        timeout = timeout_seconds
        if timeout is None:
            timeout_raw = os.environ.get("AEGISVOX_AUDIO_TRAIT_TIMEOUT_SEC", "45")
            try:
                timeout = float(timeout_raw)
            except ValueError as exc:
                raise AudioTraitExtractionError("AEGISVOX_AUDIO_TRAIT_TIMEOUT_SEC must be numeric") from exc
        kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
    request: dict[str, Any] = {
        "model": model,
        "messages": build_gpt_audio_trait_messages(audio_path),
        "temperature": 0,
    }
    # gpt-audio-1.5 currently accepts audio input through Chat Completions but
    # rejects response_format=json_object; enforce JSON via prompt and parse.
    if not model.startswith("gpt-audio"):
        request["response_format"] = {"type": "json_object"}
    try:
        if timeout_seconds is not None and hasattr(client.chat.completions, "with_options"):
            response = client.chat.completions.with_options(timeout=timeout_seconds).create(**request)
        else:
            response = client.chat.completions.create(**request)
    except Exception as exc:
        raise AudioTraitExtractionError(f"audio trait model request failed: {exc}") from exc
    text = response.choices[0].message.content
    if not isinstance(text, str) or not text.strip():
        raise AudioTraitExtractionError("audio trait model returned empty content")
    payload = _parse_json_text(text)
    return {
        "model": model,
        "raw": payload,
        "audio_traits": normalize_audio_traits(payload),
        "confidence": payload.get("confidence"),
        "evidence": payload.get("evidence") if isinstance(payload.get("evidence"), list) else [],
    }
