import os
import sys
from dataclasses import dataclass

import pytest

from aegisvox.diarization import (
    CommandDiarizationConfig,
    CommandDiarizationService,
    DiarizationSetupError,
    PyannoteDiarizationConfig,
    PyannoteDiarizationService,
    diarization_payload,
)
from aegisvox.schema import AegisVoxContractError
from aegisvox.storage import LocalArtifactStore


@dataclass
class FakeTurn:
    start: float
    end: float


class FakeAnnotation:
    def itertracks(self, yield_label=True):
        assert yield_label is True
        yield FakeTurn(0.0, 1.25), "track_0", "SPEAKER_00"
        yield FakeTurn(1.50, 2.75), "track_1", "SPEAKER_01"


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return FakeAnnotation()


def test_config_preflight_requires_runtime_token(monkeypatch, tmp_path):
    for name in ("PYANNOTE_AUTH_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    service = PyannoteDiarizationService(LocalArtifactStore(tmp_path), PyannoteDiarizationConfig(), pipeline=FakePipeline())
    with pytest.raises(DiarizationSetupError, match="requires a Hugging Face token"):
        service.preflight()


def test_preflight_with_injected_pipeline_does_not_import_pyannote(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "token-for-test-not-real")
    service = PyannoteDiarizationService(LocalArtifactStore(tmp_path), PyannoteDiarizationConfig(device="cpu"), pipeline=FakePipeline())
    result = service.preflight()
    assert result["ok"] is True
    assert result["token_present"] is True
    assert result["device"] == "cpu"


def test_diarize_to_artifact_with_injected_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token-for-test-not-real")
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/input.wav"
    output_uri = "s3://aegisvox/runs/run_001/diarization.raw.json"
    store.write_bytes(source_uri, b"not a real wav; fake pipeline ignores it")
    pipeline = FakePipeline()
    service = PyannoteDiarizationService(
        store,
        PyannoteDiarizationConfig(model_id="pyannote/test", num_speakers=2),
        pipeline=pipeline,
    )
    payload = service.diarize_to_artifact(source_uri, output_uri)
    assert payload["backend"] == "pyannote"
    assert payload["model_id"] == "pyannote/test"
    assert payload["num_segments"] == 2
    assert payload["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert pipeline.calls[0][1] == {"num_speakers": 2}
    saved = store.read_json(output_uri)
    assert saved["segments"][1]["speaker_id"] == "SPEAKER_01"


def test_diarize_to_artifact_requires_existing_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token-for-test-not-real")
    store = LocalArtifactStore(tmp_path / "artifacts")
    service = PyannoteDiarizationService(store, pipeline=FakePipeline())
    with pytest.raises(AegisVoxContractError, match="source audio artifact not found"):
        service.diarize_to_artifact("s3://aegisvox/runs/run_001/missing.wav", "s3://aegisvox/runs/run_001/diarization.raw.json")


def test_diarization_payload_requires_artifact_source_uri():
    with pytest.raises(AegisVoxContractError):
        diarization_payload(source_audio_uri="/tmp/raw.wav", model_id="m", segments=[])


def test_command_diarization_service_normalizes_stdout_payload(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_cmd/input.wav"
    output_uri = "s3://aegisvox/runs/run_cmd/diarization.raw.json"
    store.write_bytes(source_uri, b"RIFF fake wav")
    script = tmp_path / "emit_diarization.py"
    script.write_text(
        "import json, os\n"
        "assert os.environ['AEGISVOX_DIARIZATION_SOURCE_URI'].endswith('/input.wav')\n"
        "print(json.dumps({'backend':'nemo_sortformer','model_id':'sortformer-test',"
        "'segments':[{'speaker':'A','start':0.0,'end':1.0},{'speaker':'B','start':1.0,'end':2.0}]}))\n"
    )
    service = CommandDiarizationService(
        store,
        CommandDiarizationConfig(
            command=f"{sys.executable} {script}",
            backend="nemo_sortformer",
            model_id="sortformer-test",
            num_speakers=2,
        ),
    )
    payload = service.diarize_to_artifact(source_uri, output_uri)
    assert payload["backend"] == "nemo_sortformer"
    assert payload["model_id"] == "sortformer-test"
    assert payload["num_segments"] == 2
    assert {row["speaker_id"] for row in payload["segments"]} == {"A", "B"}
    assert store.read_json(output_uri)["segments"][1]["speaker_id"] == "B"


def test_command_diarization_requires_real_command(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    service = CommandDiarizationService(store, CommandDiarizationConfig(command_env="AEGISVOX_TEST_DIARIZATION_COMMAND"))
    with pytest.raises(DiarizationSetupError, match="requires a real command"):
        service.preflight()
