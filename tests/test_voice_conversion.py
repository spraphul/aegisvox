import shutil
import wave
from pathlib import Path

import pytest

from aegisvox.storage import LocalArtifactStore
from aegisvox.voice_conversion import (
    IdentityVoiceConversionBackend,
    SeedVCCommandBackend,
    VoiceConversionService,
    VoiceConversionSetupError,
    segments_converted_payload,
)


class CopyForTestBackend:
    backend_name = "copy-test-backend"
    backend_version = "test-only"

    def preflight(self):
        return {"ok": True, "backend": self.backend_name}

    def convert(self, *, source_wav: Path, target_wav: Path, output_wav: Path, options):
        assert source_wav.exists()
        assert target_wav.exists()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_wav, output_wav)
        return {"test_only": True}


def write_wav(path: Path, seconds: float = 0.25, frame_rate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(b"\x00\x00" * int(seconds * frame_rate))


def test_seedvc_preflight_requires_command(monkeypatch):
    monkeypatch.delenv("SEEDVC_COMMAND", raising=False)
    with pytest.raises(VoiceConversionSetupError, match="Seed-VC command is not configured"):
        SeedVCCommandBackend().preflight()


def test_seedvc_preflight_requires_placeholders():
    with pytest.raises(VoiceConversionSetupError, match="must include"):
        SeedVCCommandBackend(command_template="python infer.py --bad").preflight()


def test_voice_conversion_with_injected_backend(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/segments/seg_000001.source.wav"
    target_uri = "s3://aegisvox/voice-pools/fixture/target.wav"
    output_uri = "s3://aegisvox/runs/run_001/converted_segments/seg_000001.converted.wav"
    write_wav(store.local_path(source_uri), seconds=0.5)
    write_wav(store.local_path(target_uri), seconds=0.5)
    service = VoiceConversionService(store, backend=CopyForTestBackend())
    payload = service.convert_segment_to_artifact(
        segment_id="seg_000001",
        speaker_id="SPEAKER_00",
        source_audio_uri=source_uri,
        target_reference_audio_uri=target_uri,
        converted_audio_uri=output_uri,
    )
    assert payload["conversion_status"] == "converted"
    assert payload["duration_ratio"] == pytest.approx(1.0)
    assert store.exists(output_uri)
    batch = segments_converted_payload([payload], backend=service.backend)
    assert batch["num_segments"] == 1
    assert batch["backend"] == "copy-test-backend"


def test_voice_conversion_missing_target_fails(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/segments/seg_000001.source.wav"
    write_wav(store.local_path(source_uri))
    service = VoiceConversionService(store, backend=CopyForTestBackend())
    with pytest.raises(AssertionError):
        service.convert_segment_to_artifact(
            segment_id="seg_000001",
            speaker_id="SPEAKER_00",
            source_audio_uri=source_uri,
            target_reference_audio_uri="s3://aegisvox/missing/target.wav",
            converted_audio_uri="s3://aegisvox/runs/run_001/converted_segments/seg_000001.converted.wav",
        )


def test_identity_voice_conversion_requires_explicit_baseline_opt_in(monkeypatch):
    monkeypatch.delenv("AEGISVOX_ALLOW_IDENTITY_VOICE_CONVERSION_BASELINE", raising=False)
    with pytest.raises(VoiceConversionSetupError, match="baseline-only"):
        IdentityVoiceConversionBackend().preflight()


def test_identity_voice_conversion_baseline_copies_source(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGISVOX_ALLOW_IDENTITY_VOICE_CONVERSION_BASELINE", "true")
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/segments/seg_000001.source.wav"
    target_uri = "s3://aegisvox/voice-pools/fixture/target.wav"
    output_uri = "s3://aegisvox/runs/run_001/converted_segments/seg_000001.converted.wav"
    write_wav(store.local_path(source_uri), seconds=0.5)
    # The identity backend ignores the target waveform but keeps the URI in the
    # artifact contract for paired metric compatibility.
    service = VoiceConversionService(store, backend=IdentityVoiceConversionBackend())
    payload = service.convert_segment_to_artifact(
        segment_id="seg_000001",
        speaker_id="SPEAKER_00",
        source_audio_uri=source_uri,
        target_reference_audio_uri=target_uri,
        converted_audio_uri=output_uri,
    )
    assert payload["conversion_status"] == "converted"
    assert payload["backend"] == "identity-no-speaker-conversion"
    assert payload["backend_metadata"]["identity_baseline"] is True
    assert store.local_path(output_uri).read_bytes() == store.local_path(source_uri).read_bytes()
