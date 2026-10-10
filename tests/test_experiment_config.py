import json
from pathlib import Path

import pytest

from aegisvox.experiment_config import load_experiment_config, write_resolved_run_manifest
from aegisvox.schema import AegisVoxContractError, RunRequest


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_load_experiment_config_resolves_per_run_requests(tmp_path):
    cfg_path = write_yaml(
        tmp_path / "experiment.yaml",
        """
schema_version: aegisvox_experiment.v1
experiment_id: clinical_long_50_test
claim_boundary: engineering smoke only
defaults:
  privacy_mode: high
  enable_speech_editing: false
  background_preservation_mode: separate_and_remix
runtime:
  max_seconds: null
  concurrency: 1
  smoke_first_n: 2
backend_env:
  AEGISVOX_DIARIZATION_BACKEND: pyannote
audio_inputs:
  - audio_id: first_file
    path: ${AEGISVOX_DATA_ROOT}/data/in/first.wav
    metadata:
      split: eval
  - audio_id: existing_uri
    uri: s3://aegisvox/runs/already/input.wav
""",
    )
    config = load_experiment_config(cfg_path)
    rows = config.resolved_run_requests()

    assert config.experiment_id == "clinical_long_50_test"
    assert len(rows) == 2
    assert rows[0]["run_id"].startswith("clinical_long_50_test_0000_first_file")
    assert rows[0]["input_audio_uri"].endswith("/input.wav")
    assert rows[0]["request"]["privacy_mode"] == "high"
    assert rows[0]["request"]["experiment_config"]["experiment_id"] == "clinical_long_50_test"
    assert rows[1]["input_audio_uri"] == "s3://aegisvox/runs/already/input.wav"


def test_write_resolved_run_manifest_jsonl(tmp_path):
    cfg_path = write_yaml(
        tmp_path / "experiment.yaml",
        """
schema_version: aegisvox_experiment.v1
experiment_id: jsonl_test
audio_inputs:
  - audio_id: a
    path: /tmp/a.wav
""",
    )
    config = load_experiment_config(cfg_path)
    output = write_resolved_run_manifest(config, tmp_path / "runs.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["audio_id"] == "a"
    assert rows[0]["request"]["experiment_config"]["runtime"]["max_seconds"] is None


def test_audio_metadata_language_hint_overrides_default(tmp_path):
    cfg_path = write_yaml(
        tmp_path / "experiment.yaml",
        """
schema_version: aegisvox_experiment.v1
experiment_id: multilingual_test
defaults:
  language_hint: en
audio_inputs:
  - audio_id: french_call
    path: /tmp/fr.wav
    metadata:
      language_hint: fr
""",
    )
    config = load_experiment_config(cfg_path)
    rows = config.resolved_run_requests()
    assert rows[0]["request"]["language_hint"] == "fr"
    assert rows[0]["request"]["experiment_config"]["defaults"]["language_hint"] == "en"


def test_experiment_config_rejects_secret_like_keys(tmp_path):
    cfg_path = write_yaml(
        tmp_path / "bad.yaml",
        """
schema_version: aegisvox_experiment.v1
experiment_id: bad
audio_inputs:
  - audio_id: a
    path: /tmp/a.wav
backend_env:
  OPENAI_API_KEY: not_allowed_here
""",
    )
    with pytest.raises(AegisVoxContractError, match="secret-like"):
        load_experiment_config(cfg_path)


def test_run_request_accepts_experiment_config_roundtrip():
    request = RunRequest(
        input_audio_uri="s3://aegisvox/runs/r/input.wav",
        experiment_config={"experiment_id": "exp", "audio_id": "a"},
    )
    assert request.to_dict()["experiment_config"]["experiment_id"] == "exp"
