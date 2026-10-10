"""Multi-embedding voice quality/profile service for pseudo-speaker matching.

This module separates identity/privacy embeddings (ECAPA) from perceptual voice
quality signals that matter for target suitability: adult-vs-child maturity,
vocal weight, resonance, pitch, and coarse style.  Runtime model loading is
fail-loud for the real GPU backend; tests can inject deterministic backends.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import wave
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .speaker_profile import _estimate_median_f0_hz, _pitch_band_from_f0
from .voice_pool import normalize_audio_trait_value


class VoiceQualityProfileError(RuntimeError):
    """Raised when voice quality profiling cannot run or returns invalid data."""


class VoiceQualityBackend(Protocol):
    backend_name: str
    backend_version: str

    def profile_wav(self, wav_path: Path, *, include_embeddings: bool = False) -> dict[str, Any]: ...


def _wav_stats(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise VoiceQualityProfileError(f"audio WAV is missing or empty: {path}")
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            nframes = wav.getnframes()
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            frames = wav.readframes(nframes)
    except wave.Error as exc:
        raise VoiceQualityProfileError(f"expected PCM WAV for voice profiling: {path}") from exc
    duration = nframes / float(sample_rate) if sample_rate else 0.0
    if sampwidth == 2 and frames:
        import audioop

        rms = float(audioop.rms(frames, sampwidth))
    else:
        rms = None
    median_f0 = _estimate_median_f0_hz(path)
    pitch_band = _pitch_band_from_f0(median_f0)
    return {
        "duration_sec": round(duration, 3),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sampwidth,
        "rms": rms,
        "median_f0_hz": round(median_f0, 2) if median_f0 is not None else None,
        "pitch_band": pitch_band,
    }


def _heuristic_profile_from_acoustics(stats: dict[str, Any]) -> dict[str, Any]:
    pitch = str(stats.get("pitch_band") or "unknown")
    f0 = stats.get("median_f0_hz")
    vocal_weight = "unknown"
    resonance = "unknown"
    if pitch == "high":
        vocal_weight = "thin"
        resonance = "light"
    elif pitch == "mid":
        vocal_weight = "medium"
        resonance = "balanced"
    elif pitch == "low":
        vocal_weight = "thick"
        resonance = "deep"
    # This is intentionally conservative and is not a demographic claim.  It is
    # only used to prevent obvious child-like pseudo-targets when the model
    # backend is unavailable in unit tests/smokes.
    child_like_risk = 0.0
    if isinstance(f0, (int, float)) and f0 >= 245:
        child_like_risk = 0.45
    elif isinstance(f0, (int, float)) and f0 >= 220:
        child_like_risk = 0.25
    return {
        "pitch_band": pitch,
        "vocal_weight": vocal_weight,
        "resonance": resonance,
        "voice_maturity": "unknown",
        "child_like_risk": child_like_risk,
        "adult_target_suitability": round(1.0 - child_like_risk, 3),
        "gender_presentation": "unknown",
        "confidence": 0.25,
    }


def _normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = dict(profile)
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
        if key in normalized:
            value = normalize_audio_trait_value(key, normalized.get(key))
            if value:
                normalized[key] = value
    for key in ["child_like_risk", "adult_target_suitability", "confidence", "age_estimate"]:
        if key in normalized and normalized[key] is not None:
            try:
                normalized[key] = round(float(normalized[key]), 4)
            except (TypeError, ValueError):
                normalized.pop(key, None)
    return normalized


def _audio_to_tensor(path: Path, *, target_sample_rate: int, max_seconds: float | None = None) -> tuple[Any, int]:
    try:
        torch = importlib.import_module("torch")
        torchaudio = importlib.import_module("torchaudio")
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise VoiceQualityProfileError("voice quality models require torch and torchaudio") from exc
    wav, sample_rate = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    if max_seconds and max_seconds > 0:
        max_frames = int(round(max_seconds * sample_rate))
        if wav.shape[-1] > max_frames:
            wav = wav[..., :max_frames]
    return wav.squeeze(0).to(dtype=torch.float32), sample_rate


@dataclass
class HFAgeGenderVoiceBackend:
    """Hugging Face audio classifier/profile backend.

    Default model is the audEERING age/gender model.  The adapter is defensive:
    labels vary across models, so it maps any returned labels containing child,
    female, male, adult/mature/etc. into the AegisVox profile schema and stores
    raw classifier rows for audit.
    """

    model_id: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    device: str = "auto"
    max_seconds: float = 20.0
    backend_name: str = "hf-age-gender-profile"
    backend_version: str = field(init=False)
    _pipeline: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend_version = f"{self.model_id};device={self.device};max_seconds={self.max_seconds}"

    def _device_arg(self) -> int:
        if self.device == "auto":
            try:
                torch = importlib.import_module("torch")
                return 0 if bool(torch.cuda.is_available()) else -1
            except Exception:
                return -1
        if self.device.startswith("cuda"):
            return 0
        return -1

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            transformers = importlib.import_module("transformers")
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise VoiceQualityProfileError("transformers is required for HF age/gender voice profiling") from exc
        try:
            self._pipeline = transformers.pipeline("audio-classification", model=self.model_id, device=self._device_arg())
        except Exception as exc:  # pragma: no cover - external model path
            raise VoiceQualityProfileError(f"failed to load HF audio classifier {self.model_id!r}") from exc
        return self._pipeline

    @staticmethod
    def _clean_classifier_rows(rows: Any) -> list[dict[str, Any]]:
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            raise VoiceQualityProfileError("HF audio classifier returned an unsupported payload")
        clean = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("class") or "").strip().lower().replace(" ", "_")
            try:
                score = float(row.get("score"))
            except (TypeError, ValueError):
                score = None
            clean.append({"label": label, "score": score})
        return clean

    def classify_many(self, wav_paths: list[Path], *, batch_size: int = 8) -> list[list[dict[str, Any]]]:
        if not wav_paths:
            return []
        pipe = self._load()
        # Avoid the transformers pipeline's filename loader: in minimal ORD
        # model envs it requires ffmpeg even for WAVs.  Decode with torchaudio
        # and pass an array payload to keep the singleton self-contained.
        payloads = []
        for wav_path in wav_paths:
            wav, sample_rate = _audio_to_tensor(wav_path, target_sample_rate=16000, max_seconds=self.max_seconds)
            payloads.append({"array": wav.detach().cpu().numpy(), "sampling_rate": sample_rate})
        try:
            raw = pipe(payloads, top_k=None, batch_size=batch_size)
        except TypeError:
            raw = pipe(payloads)
        if len(wav_paths) == 1 and isinstance(raw, list) and (not raw or isinstance(raw[0], dict)):
            return [self._clean_classifier_rows(raw)]
        if not isinstance(raw, list) or len(raw) != len(wav_paths):
            raise VoiceQualityProfileError("HF audio classifier returned an unsupported batch payload")
        return [self._clean_classifier_rows(rows) for rows in raw]

    def classify(self, wav_path: Path) -> list[dict[str, Any]]:
        return self.classify_many([wav_path], batch_size=1)[0]

    def profile_from_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_label = {str(row.get("label")): float(row.get("score") or 0.0) for row in rows}
        female = max((score for label, score in by_label.items() if "female" in label), default=0.0)
        male = max((score for label, score in by_label.items() if "male" in label and "female" not in label), default=0.0)
        child = max((score for label, score in by_label.items() if "child" in label or "kid" in label), default=0.0)
        youth = max((score for label, score in by_label.items() if "teen" in label or "young" in label or "youth" in label), default=0.0)
        mature = max((score for label, score in by_label.items() if "mature" in label or "senior" in label or "elder" in label or "old" in label), default=0.0)
        adult = max((score for label, score in by_label.items() if "adult" in label), default=0.0)
        sorted_scores = sorted(by_label.values(), reverse=True)
        top_score = sorted_scores[0] if sorted_scores else 0.0
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        low_confidence = top_score < 0.55 or (top_score - second_score) < 0.12
        profile: dict[str, Any] = {
            "raw_classifier_rows": rows[:20],
            "confidence": top_score,
        }
        if low_confidence and rows:
            profile["classifier_low_confidence"] = True
            profile["classifier_low_confidence_reason"] = f"top={top_score:.3f};margin={top_score - second_score:.3f}"
            return profile
        profile["child_like_risk"] = max(child, youth * 0.75)
        profile["adult_target_suitability"] = max(adult, mature, 1.0 - max(child, youth * 0.75)) if rows else None
        if max(female, male) >= 0.60 and abs(female - male) >= 0.15:
            profile["gender_presentation"] = "female" if female >= male else "male"
            profile["gender_confidence"] = max(female, male)
        if child >= 0.65:
            profile["voice_maturity"] = "child"
        elif youth >= 0.55:
            profile["voice_maturity"] = "youthful"
        elif mature >= 0.55:
            profile["voice_maturity"] = "mature"
        elif adult >= 0.55:
            profile["voice_maturity"] = "adult"
        return profile

    def profile_wav(self, wav_path: Path, *, include_embeddings: bool = False) -> dict[str, Any]:
        rows = self.classify(wav_path)
        return _normalize_profile(self.profile_from_rows(rows))

    def profile_wavs(self, wav_paths: list[Path], *, batch_size: int = 8) -> list[dict[str, Any]]:
        return [_normalize_profile(self.profile_from_rows(rows)) for rows in self.classify_many(wav_paths, batch_size=batch_size)]


@dataclass
class HFWavLMStyleEmbeddingBackend:
    """WavLM-style pooled embedding backend for perceptual/style ranking."""

    model_id: str = "microsoft/wavlm-large"
    device: str = "auto"
    max_seconds: float = 20.0
    backend_name: str = "hf-wavlm-style-embedding"
    backend_version: str = field(init=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend_version = f"{self.model_id};device={self.device};max_seconds={self.max_seconds}"

    def _resolve_device(self) -> str:
        torch = importlib.import_module("torch")
        if self.device == "auto":
            return "cuda" if bool(torch.cuda.is_available()) else "cpu"
        if self.device.startswith("cuda") and not bool(torch.cuda.is_available()):
            raise VoiceQualityProfileError(f"requested {self.device} but torch.cuda.is_available() is false")
        return self.device

    def _load(self) -> tuple[Any, Any, Any, str]:
        if self._model is not None:
            return self._torch, self._processor, self._model, self._resolve_device()
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise VoiceQualityProfileError("transformers and torch are required for WavLM style embeddings") from exc
        device = self._resolve_device()
        try:
            try:
                processor = transformers.AutoProcessor.from_pretrained(self.model_id)
            except Exception:
                processor = transformers.AutoFeatureExtractor.from_pretrained(self.model_id)
            model = transformers.AutoModel.from_pretrained(self.model_id)
            model.to(device)
            model.eval()
        except Exception as exc:  # pragma: no cover - external model path
            raise VoiceQualityProfileError(f"failed to load WavLM style model {self.model_id!r}") from exc
        self._torch = torch
        self._processor = processor
        self._model = model
        return torch, processor, model, device

    def embed_wavs(self, wav_paths: list[Path], *, batch_size: int = 8) -> list[list[float]]:
        if not wav_paths:
            return []
        torch, processor, model, device = self._load()
        embeddings: list[list[float]] = []
        for start in range(0, len(wav_paths), max(1, batch_size)):
            chunk = wav_paths[start : start + max(1, batch_size)]
            arrays = []
            sample_rate = 16000
            for wav_path in chunk:
                wav, sample_rate = _audio_to_tensor(wav_path, target_sample_rate=16000, max_seconds=self.max_seconds)
                arrays.append(wav.detach().cpu().numpy())
            inputs = processor(arrays, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
            with torch.no_grad():
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state
                pooled = hidden.mean(dim=1).detach().cpu().float().tolist()
            embeddings.extend([[float(x) for x in row] for row in pooled])
        return embeddings

    def embed_wav(self, wav_path: Path) -> list[float]:
        return self.embed_wavs([wav_path], batch_size=1)[0]

    def profile_wav(self, wav_path: Path, *, include_embeddings: bool = False) -> dict[str, Any]:
        if not include_embeddings:
            return {}
        embedding = self.embed_wav(wav_path)
        return {"embeddings": {"wavlm_style": {"dim": len(embedding), "embedding": embedding}}}

    def profile_wavs(self, wav_paths: list[Path], *, include_embeddings: bool = False, batch_size: int = 8) -> list[dict[str, Any]]:
        if not include_embeddings:
            return [{} for _ in wav_paths]
        embeddings = self.embed_wavs(wav_paths, batch_size=batch_size)
        return [{"embeddings": {"wavlm_style": {"dim": len(embedding), "embedding": embedding}}} for embedding in embeddings]


@dataclass
class MultiEmbeddingVoiceQualityBackend:
    """Combine acoustics, HF classifier traits, and optional WavLM embeddings."""

    age_gender_backend: VoiceQualityBackend | None = None
    style_backend: VoiceQualityBackend | None = None
    fail_without_models: bool = True
    backend_name: str = "multi-embedding-voice-quality"
    backend_version: str = "v1"

    @classmethod
    def from_env(cls) -> "MultiEmbeddingVoiceQualityBackend":
        age_model = os.environ.get("AEGISVOX_AGE_GENDER_MODEL", "audeering/wav2vec2-large-robust-24-ft-age-gender")
        style_model = os.environ.get("AEGISVOX_STYLE_EMBEDDING_MODEL", "microsoft/wavlm-large")
        device = os.environ.get("AEGISVOX_VOICE_QUALITY_DEVICE", "auto")
        max_seconds = float(os.environ.get("AEGISVOX_VOICE_QUALITY_MAX_SECONDS", "20"))
        enable_style = os.environ.get("AEGISVOX_VOICE_QUALITY_ENABLE_STYLE_EMBEDDING", "true").lower() in {"1", "true", "yes"}
        fail_without = os.environ.get("AEGISVOX_VOICE_QUALITY_FAIL_WITHOUT_MODELS", "true").lower() in {"1", "true", "yes"}
        return cls(
            age_gender_backend=HFAgeGenderVoiceBackend(model_id=age_model, device=device, max_seconds=max_seconds),
            style_backend=HFWavLMStyleEmbeddingBackend(model_id=style_model, device=device, max_seconds=max_seconds) if enable_style else None,
            fail_without_models=fail_without,
        )

    def profile_wav(self, wav_path: Path, *, include_embeddings: bool = False) -> dict[str, Any]:
        stats = _wav_stats(wav_path)
        profile = _heuristic_profile_from_acoustics(stats)
        model_reports: dict[str, Any] = {}
        warnings: list[dict[str, Any]] = []
        if self.age_gender_backend is None:
            if self.fail_without_models:
                raise VoiceQualityProfileError("age/gender backend is required for multi-embedding voice profiling")
            warnings.append(
                {
                    "type": "age_gender_backend_missing",
                    "message": "debug/acoustic-only mode; profile is not model-backed",
                }
            )
        else:
            try:
                age_profile = self.age_gender_backend.profile_wav(wav_path, include_embeddings=False)
                profile.update({k: v for k, v in age_profile.items() if k != "raw_classifier_rows"})
                if age_profile.get("classifier_low_confidence"):
                    warnings.append(
                        {
                            "type": "age_gender_low_confidence_ignored",
                            "message": str(age_profile.get("classifier_low_confidence_reason") or "low-confidence classifier output"),
                        }
                    )
                model_reports["age_gender"] = {
                    "backend": self.age_gender_backend.backend_name,
                    "version": self.age_gender_backend.backend_version,
                    "raw_classifier_rows": age_profile.get("raw_classifier_rows"),
                }
            except Exception as exc:  # noqa: BLE001
                if self.fail_without_models:
                    raise
                warnings.append({"type": "age_gender_backend_failed", "message": str(exc)})
        embeddings: dict[str, Any] = {}
        if include_embeddings and self.style_backend is not None:
            try:
                style_profile = self.style_backend.profile_wav(wav_path, include_embeddings=True)
                embeddings.update(style_profile.get("embeddings") or {})
                model_reports["style_embedding"] = {
                    "backend": self.style_backend.backend_name,
                    "version": self.style_backend.backend_version,
                }
            except Exception as exc:  # noqa: BLE001
                if self.fail_without_models:
                    raise
                warnings.append({"type": "style_embedding_backend_failed", "message": str(exc)})
        profile = _normalize_profile(profile)
        traits = {k: v for k, v in profile.items() if k in {"gender_presentation", "voice_maturity", "vocal_weight", "resonance", "pitch_band", "speaking_rate", "tone", "energy", "language", "accent_group", "geography_region"} and v not in {None, "", "unknown"}}
        return {
            "ok": True,
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "audio": str(wav_path),
            "acoustic": stats,
            "profile": profile,
            "audio_traits": traits,
            "embeddings": embeddings,
            "model_reports": model_reports,
            "warnings": warnings,
            "claim_boundary": "Operational pseudo-speaker target/source suitability profile only; not identity, demographic, clinical, or protected-class evidence.",
        }

    def profile_wavs(self, wav_paths: list[Path], *, include_embeddings: bool = False, batch_size: int = 8) -> list[dict[str, Any]]:
        if not wav_paths:
            return []
        stats_list = [_wav_stats(path) for path in wav_paths]
        profiles = [_heuristic_profile_from_acoustics(stats) for stats in stats_list]
        model_reports: list[dict[str, Any]] = [{} for _ in wav_paths]
        warnings: list[list[dict[str, Any]]] = [[] for _ in wav_paths]
        embeddings: list[dict[str, Any]] = [{} for _ in wav_paths]

        if self.age_gender_backend is None:
            if self.fail_without_models:
                raise VoiceQualityProfileError("age/gender backend is required for multi-embedding voice profiling")
            for row_warnings in warnings:
                row_warnings.append(
                    {
                        "type": "age_gender_backend_missing",
                        "message": "debug/acoustic-only mode; profile is not model-backed",
                    }
                )
        else:
            try:
                if hasattr(self.age_gender_backend, "profile_wavs"):
                    age_profiles = self.age_gender_backend.profile_wavs(wav_paths, batch_size=batch_size)  # type: ignore[attr-defined]
                else:
                    age_profiles = [self.age_gender_backend.profile_wav(path, include_embeddings=False) for path in wav_paths]
                for idx, age_profile in enumerate(age_profiles):
                    profiles[idx].update({k: v for k, v in age_profile.items() if k != "raw_classifier_rows"})
                    if age_profile.get("classifier_low_confidence"):
                        warnings[idx].append(
                            {
                                "type": "age_gender_low_confidence_ignored",
                                "message": str(age_profile.get("classifier_low_confidence_reason") or "low-confidence classifier output"),
                            }
                        )
                    model_reports[idx]["age_gender"] = {
                        "backend": self.age_gender_backend.backend_name,
                        "version": self.age_gender_backend.backend_version,
                        "raw_classifier_rows": age_profile.get("raw_classifier_rows"),
                    }
            except Exception as exc:  # noqa: BLE001
                if self.fail_without_models:
                    raise
                for row_warnings in warnings:
                    row_warnings.append({"type": "age_gender_backend_failed", "message": str(exc)})

        if include_embeddings and self.style_backend is not None:
            try:
                if hasattr(self.style_backend, "profile_wavs"):
                    style_profiles = self.style_backend.profile_wavs(wav_paths, include_embeddings=True, batch_size=batch_size)  # type: ignore[attr-defined]
                else:
                    style_profiles = [self.style_backend.profile_wav(path, include_embeddings=True) for path in wav_paths]
                for idx, style_profile in enumerate(style_profiles):
                    embeddings[idx].update(style_profile.get("embeddings") or {})
                    model_reports[idx]["style_embedding"] = {
                        "backend": self.style_backend.backend_name,
                        "version": self.style_backend.backend_version,
                    }
            except Exception as exc:  # noqa: BLE001
                if self.fail_without_models:
                    raise
                for row_warnings in warnings:
                    row_warnings.append({"type": "style_embedding_backend_failed", "message": str(exc)})

        results = []
        for wav_path, stats, profile, row_embeddings, row_reports, row_warnings in zip(
            wav_paths, stats_list, profiles, embeddings, model_reports, warnings
        ):
            profile = _normalize_profile(profile)
            traits = {
                k: v
                for k, v in profile.items()
                if k
                in {
                    "gender_presentation",
                    "voice_maturity",
                    "vocal_weight",
                    "resonance",
                    "pitch_band",
                    "speaking_rate",
                    "tone",
                    "energy",
                    "language",
                    "accent_group",
                    "geography_region",
                }
                and v not in {None, "", "unknown"}
            }
            results.append(
                {
                    "ok": True,
                    "backend": self.backend_name,
                    "backend_version": self.backend_version,
                    "audio": str(wav_path),
                    "acoustic": stats,
                    "profile": profile,
                    "audio_traits": traits,
                    "embeddings": row_embeddings,
                    "model_reports": row_reports,
                    "warnings": row_warnings,
                    "claim_boundary": "Operational pseudo-speaker target/source suitability profile only; not identity, demographic, clinical, or protected-class evidence.",
                }
            )
        return results


def post_voice_quality_profile(base_url: str, audio: Path, *, include_embeddings: bool = False, timeout_sec: int = 180) -> dict[str, Any]:
    payload = {"audio": str(audio), "include_embeddings": include_embeddings}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + "/profile", data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - local/internal service URL
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise VoiceQualityProfileError(f"voice quality service request failed: {exc}") from exc
    decoded = json.loads(raw or "{}")
    if not isinstance(decoded, dict):
        raise VoiceQualityProfileError("voice quality service returned non-object JSON")
    return decoded


def post_voice_quality_profiles(
    base_url: str,
    audios: list[Path],
    *,
    include_embeddings: bool = False,
    batch_size: int = 8,
    timeout_sec: int = 900,
) -> list[dict[str, Any]]:
    if not audios:
        return []
    payload = {"audios": [str(audio) for audio in audios], "include_embeddings": include_embeddings, "batch_size": batch_size}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + "/profile_batch", data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - local/internal service URL
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Older service process may not have /profile_batch yet; fall back to
        # sequential calls rather than failing a long enrichment run.
        if exc.code == 404:
            return [post_voice_quality_profile(base_url, audio, include_embeddings=include_embeddings, timeout_sec=timeout_sec) for audio in audios]
        raise VoiceQualityProfileError(f"voice quality service batch request failed: {exc}") from exc
    except urllib.error.URLError as exc:
        raise VoiceQualityProfileError(f"voice quality service batch request failed: {exc}") from exc
    decoded = json.loads(raw or "{}")
    if not isinstance(decoded, dict) or not isinstance(decoded.get("results"), list):
        raise VoiceQualityProfileError("voice quality batch service returned malformed JSON")
    return [dict(row) for row in decoded["results"]]
