"""Background-preserving source separation and final remix services.

This path is intended for noisy realistic audio: separate speech-like foreground
from background first, run the normal AegisVox identity/de-id pipeline on the
cleaner foreground, then re-add the preserved background as a final candidate
artifact. It must remain explicitly gated because background stems can still
contain intelligible speech or PHI-like content.
"""

from __future__ import annotations

import math
import shutil
import struct
import wave
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .overlap_handling import CommandSeparationBackend, HTTPAudioSeparatorBackend
from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore


class BackgroundPreservationError(RuntimeError):
    """Raised when background preservation cannot safely complete."""


def _stem_path(manifest: dict[str, Any], stem_name: str) -> Path:
    stems = manifest.get("stems") if isinstance(manifest.get("stems"), list) else []
    for row in stems:
        if isinstance(row, dict) and row.get("stem") == stem_name and row.get("path"):
            return Path(str(row["path"]))
    raise BackgroundPreservationError(f"stems_manifest missing {stem_name!r} stem")


def _read_pcm16_mono(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sampwidth != 2:
        raise BackgroundPreservationError(f"expected PCM16 WAV for {path}, got sample width {sampwidth}")
    values = list(struct.unpack("<" + "h" * (len(frames) // 2), frames)) if frames else []
    if channels == 1:
        return values, rate
    mono = []
    for idx in range(0, len(values), channels):
        mono.append(int(round(sum(values[idx : idx + channels]) / float(channels))))
    return mono, rate


def _write_pcm16_mono(path: Path, samples: list[int], *, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = [max(-32768, min(32767, int(round(value)))) for value in samples]
    frames = struct.pack("<" + "h" * len(clipped), *clipped) if clipped else b""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)


def _resample_linear(samples: list[int], *, in_rate: int, out_rate: int) -> list[int]:
    if in_rate == out_rate or not samples:
        return list(samples)
    out_len = max(1, int(round(len(samples) * out_rate / float(in_rate))))
    if out_len == 1:
        return [samples[0]]
    ratio = (len(samples) - 1) / float(out_len - 1)
    out = []
    for idx in range(out_len):
        pos = idx * ratio
        left = int(math.floor(pos))
        right = min(left + 1, len(samples) - 1)
        frac = pos - left
        out.append(int(round(samples[left] * (1.0 - frac) + samples[right] * frac)))
    return out



def _leakage_int(payload: dict[str, Any] | None, key: str) -> int:
    if not isinstance(payload, dict):
        return 0
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _background_policy_action(
    leakage_report: dict[str, Any] | None,
    *,
    leakage_policy: str,
    requested_background_gain: float,
    speech_duck_gain: float,
    phi_drop_gain: float,
) -> tuple[str, float, dict[str, Any]]:
    """Decide whether preserved background can be remixed.

    The policy intentionally uses only aggregate leakage metadata and never copies
    raw background ASR text into the final remix report/UI path.
    """

    if leakage_policy not in {"drop_phi_duck_speech", "preserve_if_clean"}:
        raise BackgroundPreservationError(f"unsupported background leakage policy {leakage_policy!r}")
    if leakage_report is None:
        return (
            "preserve_background_without_leakage_report",
            requested_background_gain,
            {
                "report_available": False,
                "num_high_confidence_phi": 0,
                "text_chars": 0,
                "num_words": 0,
            },
        )
    high_conf_phi = _leakage_int(leakage_report, "num_high_confidence_phi")
    text_chars = _leakage_int(leakage_report, "text_chars")
    num_words = _leakage_int(leakage_report, "num_words")
    num_detected_phi = _leakage_int(leakage_report, "num_detected_phi")
    summary = {
        "report_available": True,
        "status": leakage_report.get("status"),
        "background_audio_uri": leakage_report.get("background_audio_uri"),
        "asr_artifact_uri": leakage_report.get("asr_artifact_uri"),
        "text_chars": text_chars,
        "num_words": num_words,
        "num_detected_phi": num_detected_phi,
        "num_high_confidence_phi": high_conf_phi,
        "high_confidence_phi_types": leakage_report.get("high_confidence_phi_types") or [],
    }
    if high_conf_phi > 0:
        return "drop_background_due_phi_leakage", phi_drop_gain, summary
    if text_chars > 0 or num_words > 0:
        if leakage_policy == "preserve_if_clean":
            return "drop_background_due_speech_leakage", phi_drop_gain, summary
        return "duck_background_due_speech_leakage", min(requested_background_gain, speech_duck_gain), summary
    return "preserve_background", requested_background_gain, summary

def _fit(samples: list[int], n: int) -> list[int]:
    if len(samples) == n:
        return list(samples)
    if len(samples) > n:
        return list(samples[:n])
    return list(samples) + [0] * (n - len(samples))


@dataclass
class SourceSeparationService:
    artifact_store: LocalArtifactStore
    separation_backend: CommandSeparationBackend | None = None

    def separate_to_artifacts(
        self,
        *,
        input_audio_uri: str,
        output_vocal_uri: str,
        output_background_uri: str,
        output_report_uri: str,
        output_prefix_uri: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for uri in [input_audio_uri, output_vocal_uri, output_background_uri, output_report_uri, output_prefix_uri.rstrip("/") + "/"]:
            validate_artifact_uri(uri)
        backend = self.separation_backend
        if backend is None:
            backend_name = os.environ.get("AEGISVOX_SOURCE_SEPARATION_BACKEND", "command").strip().lower()
            if backend_name in {"http", "audio_separator_http", "persistent", "persistent_http", "service"}:
                backend = HTTPAudioSeparatorBackend()
            elif backend_name in {"command", "audio_separator_command", "bsroformer_command"}:
                backend = CommandSeparationBackend()
            else:
                raise BackgroundPreservationError(
                    "AEGISVOX_SOURCE_SEPARATION_BACKEND must be command or http for source separation"
                )
        metadata = backend.separate(
            overlap_wav=self.artifact_store.local_path(input_audio_uri),
            output_dir=self.artifact_store.local_path(output_prefix_uri.rstrip("/") + "/"),
            region=type("SourceRegion", (), {"overlap_id": "source_full_clip"})(),
            options=options or {},
        )
        manifest_path = Path(str(metadata["stems_manifest_path"]))
        manifest = self.artifact_store.read_json(f"file://{manifest_path}")
        vocal_path = _stem_path(manifest, "vocal")
        background_path = _stem_path(manifest, "background")
        if not vocal_path.exists() or not background_path.exists():
            raise BackgroundPreservationError("source separation backend did not produce vocal/background stems")
        self.artifact_store.local_path(output_vocal_uri).parent.mkdir(parents=True, exist_ok=True)
        self.artifact_store.local_path(output_background_uri).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vocal_path, self.artifact_store.local_path(output_vocal_uri))
        shutil.copy2(background_path, self.artifact_store.local_path(output_background_uri))
        report = {
            "service": "source_separation",
            "status": "completed_vocal_background_separation",
            "input_audio_uri": input_audio_uri,
            "foreground_vocal_uri": output_vocal_uri,
            "preserved_background_uri": output_background_uri,
            "backend": backend.backend_name,
            "backend_version": backend.backend_version,
            "stems_manifest_path": str(manifest_path),
            "claim_boundary": (
                "foreground/background separation is a preprocessing aid. Background stems may contain intelligible speech "
                "or PHI-like content and require separate leakage checks before clinical/privacy-readiness claims."
            ),
        }
        self.artifact_store.write_json(output_report_uri, report)
        return report


@dataclass
class FinalBackgroundRemixService:
    artifact_store: LocalArtifactStore

    def remix_to_artifacts(
        self,
        *,
        foreground_audio_uri: str,
        background_audio_uri: str,
        output_audio_uri: str,
        output_report_uri: str,
        background_leakage_report_uri: str | None = None,
        foreground_gain: float = 0.95,
        background_gain: float = 0.65,
        speech_duck_background_gain: float = 0.10,
        phi_drop_background_gain: float = 0.0,
        leakage_policy: str = "drop_phi_duck_speech",
    ) -> dict[str, Any]:
        uris = [foreground_audio_uri, background_audio_uri, output_audio_uri, output_report_uri]
        if background_leakage_report_uri is not None:
            uris.append(background_leakage_report_uri)
        for uri in uris:
            validate_artifact_uri(uri)
        leakage_report = (
            self.artifact_store.read_json(background_leakage_report_uri) if background_leakage_report_uri is not None else None
        )
        action, effective_background_gain, leakage_summary = _background_policy_action(
            leakage_report,
            leakage_policy=leakage_policy,
            requested_background_gain=background_gain,
            speech_duck_gain=speech_duck_background_gain,
            phi_drop_gain=phi_drop_background_gain,
        )
        foreground, fg_rate = _read_pcm16_mono(self.artifact_store.local_path(foreground_audio_uri))
        background, bg_rate = _read_pcm16_mono(self.artifact_store.local_path(background_audio_uri))
        background = _resample_linear(background, in_rate=bg_rate, out_rate=fg_rate)
        n = max(len(foreground), len(background))
        fg = _fit(foreground, n)
        bg = _fit(background, n)
        mixed = [
            max(-32768, min(32767, int(round(fg[idx] * foreground_gain + bg[idx] * effective_background_gain))))
            for idx in range(n)
        ]
        _write_pcm16_mono(self.artifact_store.local_path(output_audio_uri), mixed, sample_rate=fg_rate)
        report = {
            "service": "final_background_remix",
            "status": "completed_background_preserving_final_mix",
            "foreground_audio_uri": foreground_audio_uri,
            "background_audio_uri": background_audio_uri,
            "background_leakage_report_uri": background_leakage_report_uri,
            "output_audio_uri": output_audio_uri,
            "sample_rate": fg_rate,
            "duration_sec": round(n / float(fg_rate), 4) if fg_rate else 0.0,
            "foreground_gain": foreground_gain,
            "requested_background_gain": background_gain,
            "effective_background_gain": effective_background_gain,
            "background_gain": effective_background_gain,
            "speech_duck_background_gain": speech_duck_background_gain,
            "phi_drop_background_gain": phi_drop_background_gain,
            "background_leakage_policy": leakage_policy,
            "background_policy_action": action,
            "background_leakage_summary": leakage_summary,
            "claim_boundary": (
                "background was re-added after identity/text processing only according to the leakage-aware remix policy. "
                "Ducked/dropped background mitigates obvious background speech/PHI leakage but does not replace a formal "
                "clinical PHI recall or privacy evaluation. Raw background ASR text is intentionally excluded from this report."
            ),
        }
        self.artifact_store.write_json(output_report_uri, report)
        return report
