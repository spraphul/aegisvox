"""Speaker profiling contracts and audio grouping.

The service computes deterministic duration/audio-energy summaries and delegates
speaker embeddings to a real backend. If no embedding backend is configured, it
fails loudly instead of writing fake embeddings.
"""

from __future__ import annotations

import audioop
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

from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore

MIN_RELIABLE_PROFILE_DURATION_SEC = 1.0

_FRENCH_MARKERS = {
    " le ", " la ", " les ", " des ", " une ", " un ", " je ", " vous ", " nous ",
    " dans ", " avec ", " pour ", " est ", " et ", " pas ", " oui ", " non ",
    " docteur ", " monsieur ", " madame ", " urgence ", " secours ", " sang ",
}
_ENGLISH_MARKERS = {
    " the ", " and ", " you ", " are ", " is ", " patient ", " doctor ",
    " emergency ", " call ", " have ", " with ", " for ", " not ", " yes ", " no ",
}


def _infer_document_language_from_text(text: str) -> tuple[str, dict[str, Any]]:
    padded = " " + " ".join(str(text or "").lower().split()) + " "
    if not padded.strip():
        return "unknown", {"method": "text_marker_count", "reason": "empty_text"}
    french_hits = sum(padded.count(marker) for marker in _FRENCH_MARKERS)
    english_hits = sum(padded.count(marker) for marker in _ENGLISH_MARKERS)
    if french_hits >= 3 and french_hits >= english_hits * 2:
        return "french", {"method": "text_marker_count", "french_hits": french_hits, "english_hits": english_hits}
    if english_hits >= 3 and english_hits >= french_hits * 2:
        return "english", {"method": "text_marker_count", "french_hits": french_hits, "english_hits": english_hits}
    return "unknown", {"method": "text_marker_count", "french_hits": french_hits, "english_hits": english_hits, "reason": "insufficient_margin"}


def _infer_document_language_from_asr_artifact(artifact_store: LocalArtifactStore, output_prefix_uri: str) -> tuple[str, dict[str, Any]]:
    candidates = [
        f"{output_prefix_uri.rstrip('/')}/asr.changed_identity.json",
        f"{output_prefix_uri.rstrip('/')}/asr.json",
    ]
    for uri in candidates:
        path = artifact_store.local_path(uri)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - corrupt auxiliary artifact should not abort profiling
            return "unknown", {"method": "asr_sibling_text", "asr_uri": uri, "reason": f"read_failed:{exc}"}
        explicit = str(payload.get("language") or payload.get("detected_language") or "").strip().lower()
        if explicit in {"fr", "fra", "fre", "french", "français", "francais"}:
            return "french", {"method": "asr_explicit_language", "asr_uri": uri, "raw_language": explicit}
        if explicit in {"en", "eng", "english"}:
            return "english", {"method": "asr_explicit_language", "asr_uri": uri, "raw_language": explicit}
        text = str(payload.get("text") or "")
        if not text and isinstance(payload.get("segments"), list):
            text = " ".join(str(row.get("text") or "") for row in payload["segments"] if isinstance(row, dict))
        language, evidence = _infer_document_language_from_text(text)
        evidence["asr_uri"] = uri
        return language, evidence
    return "unknown", {"method": "asr_sibling_text", "reason": "no_asr_artifact_found"}


class SpeakerProfilingSetupError(RuntimeError):
    """Raised when speaker profiling dependencies are not configured."""


class SpeakerEmbeddingBackend(Protocol):
    backend_name: str
    backend_version: str

    def embed_wav(self, wav_path: Path) -> list[float]: ...


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    duration_sec: float
    source_audio_uri: str
    source_embedding_uri: str
    language: str = "unknown"
    gender_presentation: str = "unknown"
    age_band: str = "unknown"
    pitch_band: str = "unknown"
    speaking_rate: str = "unknown"
    energy: str = "unknown"
    accent_group: str = "unknown"
    clinical_role_guess: str = "unknown"
    embedding_backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.speaker_id:
            raise AegisVoxContractError("speaker_id is required")
        validate_artifact_uri(self.source_audio_uri)
        validate_artifact_uri(self.source_embedding_uri)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "duration_sec": round(self.duration_sec, 3),
            "source_audio_uri": self.source_audio_uri,
            "source_embedding_uri": self.source_embedding_uri,
            "language": self.language,
            "gender_presentation": self.gender_presentation,
            "age_band": self.age_band,
            "pitch_band": self.pitch_band,
            "speaking_rate": self.speaking_rate,
            "energy": self.energy,
            "accent_group": self.accent_group,
            "clinical_role_guess": self.clinical_role_guess,
            "embedding_backend": self.embedding_backend,
            "metadata": dict(self.metadata),
        }


def _post_json(url: str, payload: dict[str, Any], *, timeout_sec: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - operator-configured local/internal endpoint
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise SpeakerProfilingSetupError(f"SpeechBrain embedding service request failed: {exc}") from exc
    try:
        decoded = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise SpeakerProfilingSetupError(f"SpeechBrain embedding service returned non-JSON response: {raw[:500]}") from exc
    if not isinstance(decoded, dict):
        raise SpeakerProfilingSetupError("SpeechBrain embedding service response must be a JSON object")
    return decoded


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


class SpeechBrainECAPAEmbeddingBackend:
    """SpeechBrain ECAPA adapter.

    This adapter is intentionally strict: if SpeechBrain/model loading fails, the
    caller receives a setup error and no profile artifact is written.
    """

    backend_name = "speechbrain-ecapa"

    def __init__(
        self,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str | None = None,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.savedir = savedir
        self.device = device
        resolved = self._configured_device()
        self.backend_version = f"{model_id};device={resolved or 'default'}"
        self._classifier = None

    def _configured_device(self) -> str | None:
        return self.device or os.environ.get("AEGISVOX_SPEECHBRAIN_DEVICE") or os.environ.get("SPEECHBRAIN_DEVICE")

    def _resolve_device(self) -> str | None:
        device = self._configured_device()
        if not device:
            return None
        normalized = device.strip().lower()
        if normalized in {"", "default", "none"}:
            return None
        if normalized == "auto":
            try:
                torch = importlib.import_module("torch")
            except Exception as exc:
                raise SpeakerProfilingSetupError("AEGISVOX_SPEECHBRAIN_DEVICE=auto requires torch to be installed") from exc
            return "cuda" if bool(torch.cuda.is_available()) else "cpu"
        if normalized.startswith("cuda"):
            try:
                torch = importlib.import_module("torch")
            except Exception as exc:
                raise SpeakerProfilingSetupError(f"SpeechBrain CUDA device {device!r} requested but torch is not installed") from exc
            if not bool(torch.cuda.is_available()):
                raise SpeakerProfilingSetupError(f"SpeechBrain CUDA device {device!r} requested but torch.cuda.is_available() is false")
        return device

    def _load(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            try:
                module = importlib.import_module("speechbrain.inference.speaker")
            except ImportError:
                module = importlib.import_module("speechbrain.pretrained")
            classifier_cls = getattr(module, "EncoderClassifier")
        except Exception as exc:
            raise SpeakerProfilingSetupError(
                "SpeechBrain ECAPA backend is not installed. Install speechbrain and model dependencies first."
            ) from exc
        try:
            kwargs = {"source": self.model_id}
            if self.savedir:
                kwargs["savedir"] = self.savedir
            device = self._resolve_device()
            if device:
                kwargs["run_opts"] = {"device": device}
                self.backend_version = f"{self.model_id};device={device}"
            self._classifier = classifier_cls.from_hparams(**kwargs)
        except Exception as exc:  # pragma: no cover - external model path
            raise SpeakerProfilingSetupError(
                f"failed to load SpeechBrain ECAPA model {self.model_id!r}; check network/cache/model setup"
            ) from exc
        return self._classifier

    def embed_wav(self, wav_path: Path) -> list[float]:
        classifier = self._load()
        try:
            if hasattr(classifier, "encode_file"):
                embedding = classifier.encode_file(str(wav_path))
            elif hasattr(classifier, "load_audio") and hasattr(classifier, "encode_batch"):
                wav = classifier.load_audio(str(wav_path))
                if hasattr(wav, "unsqueeze"):
                    wav = wav.unsqueeze(0)
                embedding = classifier.encode_batch(wav)
            else:
                raise SpeakerProfilingSetupError(
                    "SpeechBrain EncoderClassifier has neither encode_file nor load_audio+encode_batch"
                )
        except SpeakerProfilingSetupError:
            raise
        except Exception as exc:  # pragma: no cover - external model path
            raise SpeakerProfilingSetupError(f"failed to compute speaker embedding for {wav_path}: {exc}") from exc
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().flatten().tolist()
        elif hasattr(embedding, "flatten"):
            embedding = embedding.flatten().tolist()
        return [float(x) for x in embedding]


class SpeechBrainHTTPEmbeddingBackend:
    """Persistent SpeechBrain ECAPA embedding service client."""

    backend_name = "speechbrain-ecapa-http"

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.environ.get("AEGISVOX_SPEECHBRAIN_SERVICE_URL") or os.environ.get("SPEECHBRAIN_SERVICE_URL")
        self.backend_version = f"persistent-http-v1;endpoint={self.base_url or 'unset'}"

    def preflight(self) -> dict[str, Any]:
        if not self.base_url:
            raise SpeakerProfilingSetupError("SpeechBrain persistent service is not configured. Set AEGISVOX_SPEECHBRAIN_SERVICE_URL.")
        try:
            with urllib.request.urlopen(_join_url(self.base_url, "/health"), timeout=10) as response:  # noqa: S310
                health_raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise SpeakerProfilingSetupError(f"SpeechBrain persistent service health check failed: {exc}") from exc
        return {"ok": True, "backend": self.backend_name, "backend_version": self.backend_version, "base_url": self.base_url, "health": health_raw[:500]}

    def embed_wav(self, wav_path: Path) -> list[float]:
        self.preflight()
        if not wav_path.exists():
            raise AegisVoxContractError(f"speaker embedding WAV not found: {wav_path}")
        response = _post_json(_join_url(str(self.base_url), "/embed"), {"audio": str(wav_path)}, timeout_sec=180)
        vector = response.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise SpeakerProfilingSetupError("SpeechBrain service response missing non-empty embedding list")
        return [float(x) for x in vector]


@dataclass(frozen=True)
class _WavInfo:
    nchannels: int
    sampwidth: int
    framerate: int
    nframes: int
    rms: float

    @property
    def duration_sec(self) -> float:
        return self.nframes / float(self.framerate)


def _read_wav_info(path: Path) -> _WavInfo:
    with wave.open(str(path), "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
        rms = float(audioop.rms(frames, params.sampwidth)) if frames else 0.0
        return _WavInfo(params.nchannels, params.sampwidth, params.framerate, params.nframes, rms)


def _energy_band(rms_values: list[float]) -> str:
    if not rms_values:
        return "unknown"
    rms = sum(rms_values) / len(rms_values)
    if rms < 250:
        return "quiet"
    if rms < 2500:
        return "calm"
    return "high"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _estimate_median_f0_hz(path: Path, *, min_hz: float = 70.0, max_hz: float = 330.0) -> float | None:
    """Estimate a rough median F0 using frame autocorrelation.

    This is a lightweight operational heuristic, not a demographic classifier.
    It is used only to avoid obviously cross-gender pseudo-speaker assignment
    when the source pitch is clear enough. For long clinical conversations, cap
    the analyzed audio by default; a few minutes of concatenated speech can make
    naive autocorrelation dominate wall time without improving this heuristic.
    """

    try:
        with wave.open(str(path), "rb") as wav:
            params = wav.getparams()
            max_seconds = float(os.environ.get("AEGISVOX_F0_MAX_SECONDS", "30"))
            max_frames = params.nframes
            if max_seconds > 0 and params.framerate > 0:
                max_frames = min(max_frames, max(1, int(round(max_seconds * params.framerate))))
            frames = wav.readframes(max_frames)
    except Exception:
        return None
    if params.sampwidth != 2 or params.nchannels < 1 or params.framerate <= 0 or params.nframes <= 0:
        return None
    raw = [int.from_bytes(frames[i : i + 2], "little", signed=True) for i in range(0, len(frames), params.sampwidth * params.nchannels)]
    if not raw:
        return None
    fr = params.framerate
    frame_len = max(256, int(0.04 * fr))
    hop = max(128, int(0.02 * fr))
    min_lag = max(1, int(fr / max_hz))
    max_lag = min(frame_len - 2, int(fr / min_hz))
    if max_lag <= min_lag:
        return None
    rms_all = math.sqrt(sum(x * x for x in raw) / max(1, len(raw)))
    if rms_all < 80:
        return None
    f0s: list[float] = []
    for start in range(0, max(0, len(raw) - frame_len), hop):
        frame = raw[start : start + frame_len]
        mean = sum(frame) / len(frame)
        centered = [x - mean for x in frame]
        energy = sum(x * x for x in centered)
        if energy <= max(1.0, (rms_all * rms_all * frame_len * 0.08)):
            continue
        best_lag = None
        best_corr = -1.0
        for lag in range(min_lag, max_lag + 1):
            corr = sum(centered[i] * centered[i + lag] for i in range(0, frame_len - lag))
            norm_l = math.sqrt(sum(centered[i] * centered[i] for i in range(0, frame_len - lag)))
            norm_r = math.sqrt(sum(centered[i + lag] * centered[i + lag] for i in range(0, frame_len - lag)))
            denom = norm_l * norm_r
            if denom <= 0:
                continue
            corr /= denom
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        if best_lag and best_corr >= 0.35:
            f0s.append(fr / best_lag)
    return _median(f0s)


def _pitch_band_from_f0(f0_hz: float | None) -> str:
    if f0_hz is None:
        return "unknown"
    if f0_hz < 145:
        return "low"
    if f0_hz > 185:
        return "high"
    return "mid"


def _gender_presentation_from_f0(f0_hz: float | None) -> str:
    if f0_hz is None:
        return "unknown"
    if f0_hz >= 170:
        return "female"
    if f0_hz <= 150:
        return "male"
    return "unknown"


def _load_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("segments")
    if not isinstance(rows, list):
        raise AegisVoxContractError("segments.cleaned.json must contain a segments list")
    return rows


def _majority_known(rows: list[dict[str, Any]], key: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is None and isinstance(row.get("metadata"), dict):
            value = row["metadata"].get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        if not normalized or normalized == "unknown":
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _segment_audio_uri(row: dict[str, Any]) -> str:
    uri = row.get("segment_audio_uri") or row.get("source_audio_uri")
    if not uri:
        raise AegisVoxContractError(f"segment {row.get('segment_id')} is missing segment_audio_uri/source_audio_uri")
    validate_artifact_uri(uri)
    return uri


def concatenate_wav_segments(segment_paths: list[Path], output_path: Path) -> float:
    if not segment_paths:
        raise AegisVoxContractError("cannot concatenate zero WAV segments")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_info = _read_wav_info(segment_paths[0])
    total_frames = 0
    with wave.open(str(output_path), "wb") as dst:
        dst.setnchannels(first_info.nchannels)
        dst.setsampwidth(first_info.sampwidth)
        dst.setframerate(first_info.framerate)
        for path in segment_paths:
            with wave.open(str(path), "rb") as src:
                params = src.getparams()
                if (params.nchannels, params.sampwidth, params.framerate) != (
                    first_info.nchannels,
                    first_info.sampwidth,
                    first_info.framerate,
                ):
                    raise AegisVoxContractError("all speaker WAV segments must share channel/sample-width/rate")
                frames = src.readframes(params.nframes)
                dst.writeframes(frames)
                total_frames += params.nframes
    return total_frames / float(first_info.framerate)


def concatenate_wav_segments_limited(segment_paths: list[Path], output_path: Path, *, max_seconds: float | None) -> float:
    """Concatenate whole segments up to a duration cap.

    Speaker identity assignment does not need minutes of audio per speaker; a
    representative window is enough for ECAPA and avoids very slow long-call
    embeddings. We keep whole segments to avoid cutting words mid-turn.
    """

    if max_seconds is None or max_seconds <= 0:
        return concatenate_wav_segments(segment_paths, output_path)
    if not segment_paths:
        raise AegisVoxContractError("cannot concatenate zero WAV segments")
    selected: list[Path] = []
    total = 0.0
    for path in segment_paths:
        selected.append(path)
        info = _read_wav_info(path)
        total += info.duration_sec
        if total >= max_seconds:
            break
    return concatenate_wav_segments(selected, output_path)


def _speaker_profile_embedding_max_seconds() -> float | None:
    raw = os.environ.get("AEGISVOX_SPEAKER_PROFILE_MAX_EMBED_SECONDS", "30").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    return value if value > 0 else None


@dataclass
class SpeakerProfilingService:
    artifact_store: LocalArtifactStore
    embedding_backend: SpeakerEmbeddingBackend | None = None

    def preflight(self) -> dict[str, Any]:
        if self.embedding_backend is None:
            # Probe the real default backend so setup errors are actionable.
            SpeechBrainECAPAEmbeddingBackend()._load()
            backend = "speechbrain-ecapa"
        else:
            backend = self.embedding_backend.backend_name
        return {"ok": True, "backend": backend}

    def profile_to_artifact(
        self,
        segments_uri: str,
        output_profiles_uri: str,
        *,
        output_prefix_uri: str,
    ) -> dict[str, Any]:
        validate_artifact_uri(segments_uri)
        validate_artifact_uri(output_profiles_uri)
        validate_artifact_uri(output_prefix_uri.rstrip("/") + "/")
        if self.embedding_backend is None:
            raise SpeakerProfilingSetupError(
                "speaker profiling requires a real speaker embedding backend; configure SpeechBrain ECAPA first"
            )
        segments_path = self.artifact_store.local_path(segments_uri)
        rows = _load_segments(segments_path)
        by_speaker: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            speaker_id = row.get("speaker_id")
            if not speaker_id:
                raise AegisVoxContractError("segment row missing speaker_id")
            by_speaker.setdefault(str(speaker_id), []).append(row)
        document_language, document_language_evidence = _infer_document_language_from_asr_artifact(self.artifact_store, output_prefix_uri)
        profiles: dict[str, dict[str, Any]] = {}
        for speaker_id, speaker_rows in sorted(by_speaker.items()):
            segment_paths = [self.artifact_store.local_path(_segment_audio_uri(row)) for row in speaker_rows]
            missing = [str(path) for path in segment_paths if not path.exists()]
            if missing:
                raise AegisVoxContractError(f"missing segment audio for speaker {speaker_id}: {missing[:3]}")
            speaker_audio_uri = f"{output_prefix_uri.rstrip('/')}/speaker_audio/{speaker_id}.wav"
            embedding_audio_uri = f"{output_prefix_uri.rstrip('/')}/speaker_audio/{speaker_id}.embedding_sample.wav"
            embedding_uri = f"{output_prefix_uri.rstrip('/')}/speaker_embeddings/{speaker_id}.json"
            speaker_audio_path = self.artifact_store.local_path(speaker_audio_uri)
            embedding_audio_path = self.artifact_store.local_path(embedding_audio_uri)
            duration = concatenate_wav_segments(segment_paths, speaker_audio_path)
            max_embed_seconds = _speaker_profile_embedding_max_seconds()
            embedding_duration = concatenate_wav_segments_limited(segment_paths, embedding_audio_path, max_seconds=max_embed_seconds)
            embedding = self.embedding_backend.embed_wav(embedding_audio_path)
            self.artifact_store.write_json(
                embedding_uri,
                {
                    "embedding": embedding,
                    "backend": self.embedding_backend.backend_name,
                    "embedding_audio_uri": embedding_audio_uri,
                    "embedding_duration_sec": round(embedding_duration, 3),
                    "embedding_max_seconds": max_embed_seconds,
                },
            )
            rms_values = [_read_wav_info(path).rms for path in segment_paths]
            median_f0_hz = _estimate_median_f0_hz(speaker_audio_path)
            speaker_language = _majority_known(speaker_rows, "language")
            if speaker_language == "unknown" and document_language != "unknown":
                speaker_language = document_language
            profile = SpeakerProfile(
                speaker_id=speaker_id,
                duration_sec=duration,
                source_audio_uri=speaker_audio_uri,
                source_embedding_uri=embedding_uri,
                language=speaker_language,
                accent_group=_majority_known(speaker_rows, "accent_group"),
                energy=_energy_band(rms_values),
                pitch_band=_pitch_band_from_f0(median_f0_hz),
                gender_presentation=(
                    _majority_known(speaker_rows, "gender_presentation")
                    if _majority_known(speaker_rows, "gender_presentation") != "unknown"
                    else _gender_presentation_from_f0(median_f0_hz)
                ),
                embedding_backend=self.embedding_backend.backend_name,
                metadata={
                    "num_segments": len(speaker_rows),
                    "num_short_speaker_turns": sum(1 for row in speaker_rows if "short_speaker_turn" in (row.get("flags") or [])),
                    "profile_reliability": "insufficient_duration" if duration < MIN_RELIABLE_PROFILE_DURATION_SEC else "standard",
                    "min_reliable_profile_duration_sec": MIN_RELIABLE_PROFILE_DURATION_SEC,
                    "median_f0_hz": round(median_f0_hz, 2) if median_f0_hz is not None else None,
                    "embedding_audio_uri": embedding_audio_uri,
                    "embedding_duration_sec": round(embedding_duration, 3),
                    "embedding_max_seconds": max_embed_seconds,
                    "gender_presentation_source": "f0_autocorrelation_heuristic" if median_f0_hz is not None else "unknown",
                    "gender_presentation_claim_boundary": "operational voice-matching heuristic; not demographic or clinical inference",
                    "document_language_inference": document_language_evidence,
                    "language_source": "segment_majority" if _majority_known(speaker_rows, "language") != "unknown" else ("asr_document_text" if document_language != "unknown" else "unknown"),
                    "language_claim_boundary": "operational routing signal inferred from ASR text; not demographic or clinical evidence",
                },
            )
            profiles[speaker_id] = profile.to_dict()
        payload = {"speakers": profiles, "num_speakers": len(profiles)}
        self.artifact_store.write_json(output_profiles_uri, payload)
        return payload
