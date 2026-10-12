from __future__ import annotations

import shutil
import sys
import wave
from pathlib import Path

import pytest

from aegisvox import speech_editing as speech_editing_module
from aegisvox.speech_editing import SpeechEditingError, SpeechEditingService, SpeechEditingSetupError, VoiceCraftCommandBackend, VoiceCraftHTTPBackend
from aegisvox.storage import LocalArtifactStore


class CopySpanBackend:
    backend_name = "copy-span-test-backend"
    backend_version = "test-only"

    def preflight(self):
        return {"ok": True, "backend": self.backend_name}

    def synthesize_edit(self, *, source_wav: Path, output_wav: Path, job, options):
        assert source_wav.exists()
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        write_wav(output_wav, seconds=float(job["end_sec"]) - float(job["start_sec"]))
        return {"test_only": True}


class RecordingSpanBackend:
    backend_name = "recording-span-test-backend"
    backend_version = "test-only"

    def __init__(self):
        self.calls = []

    def preflight(self):
        return {"ok": True, "backend": self.backend_name}

    def synthesize_edit(self, *, source_wav: Path, output_wav: Path, job, options):
        self.calls.append({"edit_id": job.get("edit_id"), "cuda_visible_devices": options.get("cuda_visible_devices")})
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        write_wav(output_wav, seconds=float(job["end_sec"]) - float(job["start_sec"]))
        return {"test_only": True}


class FloatSpanBackend:
    backend_name = "float-span-test-backend"
    backend_version = "test-only"

    def preflight(self):
        return {"ok": True, "backend": self.backend_name}

    def synthesize_edit(self, *, source_wav: Path, output_wav: Path, job, options):
        sf = pytest.importorskip("soundfile")
        import numpy as np

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 16000
        nframes = int((float(job["end_sec"]) - float(job["start_sec"])) * sample_rate)
        tone = np.zeros((nframes, 1), dtype="float32")
        sf.write(str(output_wav), tone, sample_rate, subtype="FLOAT")
        return {"test_only": True, "emitted_format": "float32_16khz"}


class ConstantSpanBackend:
    backend_name = "constant-span-test-backend"
    backend_version = "test-only"

    def __init__(self, sample: int = 12000):
        self.sample = sample

    def preflight(self):
        return {"ok": True, "backend": self.backend_name}

    def synthesize_edit(self, *, source_wav: Path, output_wav: Path, job, options):
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        write_constant_wav(output_wav, seconds=float(job["end_sec"]) - float(job["start_sec"]), sample=self.sample)
        return {"test_only": True}


def write_wav(path: Path, seconds: float = 0.5, frame_rate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(b"\x00\x00" * int(seconds * frame_rate))


def write_constant_wav(path: Path, seconds: float = 0.5, frame_rate: int = 8000, sample: int = 12000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = int(sample).to_bytes(2, byteorder="little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(frame * int(seconds * frame_rate))


def read_pcm16_samples(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    return [int.from_bytes(raw[idx : idx + 2], byteorder="little", signed=True) for idx in range(0, len(raw), 2)]


def test_voicecraft_preflight_requires_command(monkeypatch):
    monkeypatch.delenv("VOICECRAFT_COMMAND", raising=False)
    with pytest.raises(SpeechEditingSetupError, match="VoiceCraft command is not configured"):
        VoiceCraftCommandBackend().preflight()


def test_voicecraft_preflight_requires_placeholders():
    with pytest.raises(SpeechEditingSetupError, match="missing required placeholders"):
        VoiceCraftCommandBackend(command_template="python infer.py --input {source}").preflight()


def test_speech_editing_with_timestamped_edit(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_wav(store.local_path(source_uri), seconds=2.0)
    store.write_json(
        deid_uri,
        {
            "edits": [
                {
                    "edit_id": "edit_0001",
                    "phi_type": "NAME",
                    "speaker_id": "SPEAKER_00",
                    "start_sec": 0.5,
                    "end_sec": 1.0,
                    "original_text": "Alice",
                    "replacement_text": "[NAME]",
                }
            ],
            "deidentified_text": "hello [NAME]",
        },
    )
    payload = SpeechEditingService(store, backend=CopySpanBackend()).edit_to_artifacts(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
        final_audio_uri=final_uri,
    )
    assert payload["status"] == "completed"
    assert payload["num_jobs"] == 1
    assert payload["jobs"][0]["status"] == "synthesized"
    assert store.exists(final_uri)
    jobs = store.read_json(jobs_uri)
    assert jobs["final_sanitized_audio_uri"] == final_uri


def test_speech_editing_can_stitch_only_mask_intervals_from_full_window(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_wav(store.local_path(source_uri), seconds=2.0, frame_rate=8000)
    store.write_json(
        deid_uri,
        {
            "edits": [
                {
                    "edit_id": "uttwindow_0001",
                    "phi_type": "NAME",
                    "speaker_id": "SPEAKER_00",
                    "start_sec": 0.0,
                    "end_sec": 2.0,
                    "mask_start_sec": 0.8,
                    "mask_end_sec": 1.0,
                    "mask_intervals": [{"start_sec": 0.8, "end_sec": 1.0}],
                    "original_text": "Alice",
                    "replacement_text": "Morgan",
                    "target_text": "hello Morgan goodbye",
                }
            ],
        },
    )
    payload = SpeechEditingService(store, backend=ConstantSpanBackend()).edit_to_artifacts(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
        final_audio_uri=final_uri,
        options={"stitch_policy": "mask_intervals"},
    )
    assert payload["status"] == "completed"
    assert payload["stitch_policy"] == "mask_intervals"
    assert payload["stitch_operations"] == [
        {
            "job_id": "voicecraft_job_0001",
            "edit_id": "uttwindow_0001",
            "start_sec": 0.8,
            "end_sec": 1.0,
            "source_window_start_sec": 0.0,
            "source_window_end_sec": 2.0,
        }
    ]
    samples = read_pcm16_samples(store.local_path(final_uri))
    assert all(value == 0 for value in samples[: int(0.75 * 8000)])
    assert any(value == 12000 for value in samples[int(0.8 * 8000) : int(1.0 * 8000)])
    assert all(value == 0 for value in samples[int(1.05 * 8000) :])


def test_speech_editing_normalizes_float_backend_span_to_source_format(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_wav(store.local_path(source_uri), seconds=2.0, frame_rate=8000)
    store.write_json(
        deid_uri,
        {
            "edits": [
                {
                    "edit_id": "edit_0001",
                    "phi_type": "PUBLIC_LEXICAL_SMOKE",
                    "speaker_id": "SPEAKER_00",
                    "start_sec": 0.5,
                    "end_sec": 1.0,
                    "original_text": "fevered",
                    "replacement_text": "calm",
                }
            ],
            "deidentified_text": "hello calm",
        },
    )
    payload = SpeechEditingService(store, backend=FloatSpanBackend()).edit_to_artifacts(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
        final_audio_uri=final_uri,
    )
    assert payload["status"] == "completed"
    assert payload["jobs"][0]["status"] == "synthesized"
    edited_span = store.local_path("s3://aegisvox/runs/run_001/edited_spans/edit_0001.edited.wav")
    with wave.open(str(edited_span), "rb") as wav:
        assert wav.getframerate() == 8000
        assert wav.getsampwidth() == 2
        assert wav.getnchannels() == 1




def test_speech_editing_parallel_workers_assign_devices_and_resume_existing(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_wav(store.local_path(source_uri), seconds=3.0)
    edits = []
    for idx in range(3):
        edits.append(
            {
                "edit_id": f"edit_{idx+1:04d}",
                "phi_type": "NAME",
                "speaker_id": "SPEAKER_00",
                "start_sec": 0.2 + idx * 0.6,
                "end_sec": 0.5 + idx * 0.6,
                "original_text": f"Name{idx}",
                "replacement_text": "[NAME]",
            }
        )
    store.write_json(deid_uri, {"edits": edits, "deidentified_text": "hello [NAME]"})
    existing_uri = "s3://aegisvox/runs/run_001/edited_spans/edit_0002.edited.wav"
    write_wav(store.local_path(existing_uri), seconds=0.3)
    backend = RecordingSpanBackend()

    payload = SpeechEditingService(store, backend=backend).edit_to_artifacts(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
        final_audio_uri=final_uri,
        options={"num_workers": 2, "cuda_devices": "0,1"},
    )

    assert payload["status"] == "completed"
    assert payload["num_jobs"] == 3
    assert payload["num_resumed_existing"] == 1
    assert payload["edit_parallelism"] == {
        "num_workers": 2,
        "cuda_devices": ["0", "1"],
        "strategy": "job_threadpool_preserve_order",
        "resume_existing": True,
    }
    # Worker invocation order is scheduler-dependent. The returned jobs must
    # retain source order, and the existing edit must not be synthesized again.
    assert sorted(call["edit_id"] for call in backend.calls) == ["edit_0001", "edit_0003"]
    assert [job["edit_id"] for job in payload["jobs"]] == ["edit_0001", "edit_0002", "edit_0003"]
    assert {call["cuda_visible_devices"] for call in backend.calls} == {"0"}
    assert payload["jobs"][1]["resumed_from_existing_output"] is True
    assert store.exists(final_uri)


def test_speech_editing_no_edits_passthrough_without_backend(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_wav(store.local_path(source_uri), seconds=1.0)
    store.write_json(deid_uri, {"edits": [], "deidentified_text": "no PHI here"})
    payload = SpeechEditingService(store).edit_to_artifacts(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri="s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json",
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
        final_audio_uri=final_uri,
    )
    assert payload["status"] == "no_edits_passthrough"
    assert store.exists(final_uri)


def test_speech_editing_unlocalized_edits_fail(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    write_wav(store.local_path(source_uri), seconds=1.0)
    store.write_json(
        deid_uri,
        {"edits": [{"edit_id": "edit_0001", "phi_type": "NAME", "replacement_text": "[NAME]"}]},
    )
    with pytest.raises(SpeechEditingError, match="unlocalized edits"):
        SpeechEditingService(store, backend=CopySpanBackend()).edit_to_artifacts(
            changed_identity_audio_uri=source_uri,
            deid_text_uri=deid_uri,
            output_jobs_uri="s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json",
            output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
            final_audio_uri="s3://aegisvox/runs/run_001/final_sanitized.wav",
        )
    assert store.read_json("s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json")["status"] == "failed_unlocalized_edits"

class FailingSpanBackend:
    backend_name = "failing-span-test-backend"
    backend_version = "test-only"

    def preflight(self):
        return {"ok": True, "backend": self.backend_name}

    def synthesize_edit(self, *, source_wav: Path, output_wav: Path, job, options):
        raise RuntimeError("HF_TOKEN=secret-test-value patient@example.com")


def test_speech_editing_redacts_failed_job_reason(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_wav(store.local_path(source_uri), seconds=2.0)
    store.write_json(
        deid_uri,
        {
            "edits": [
                {
                    "edit_id": "edit_0001",
                    "phi_type": "NAME",
                    "speaker_id": "SPEAKER_00",
                    "start_sec": 0.5,
                    "end_sec": 1.0,
                    "replacement_text": "[NAME]",
                }
            ]
        },
    )

    with pytest.raises(SpeechEditingError, match="speech editing failed"):
        SpeechEditingService(store, backend=FailingSpanBackend()).edit_to_artifacts(
            changed_identity_audio_uri=source_uri,
            deid_text_uri=deid_uri,
            output_jobs_uri=jobs_uri,
            output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
            final_audio_uri=final_uri,
        )

    payload = store.read_json(jobs_uri)
    assert payload["status"] == "failed"
    blob = str(payload["jobs"])
    assert "hf_" not in blob
    assert "patient@example.com" not in blob
    assert "[REDACTED]" in blob



class PartiallyFailingSpanBackend:
    backend_name = "partially-failing-span-test-backend"
    backend_version = "test-only"

    def preflight(self):
        return {"ok": True, "backend": self.backend_name}

    def synthesize_edit(self, *, source_wav: Path, output_wav: Path, job, options):
        if job.get("edit_id") == "edit_0002":
            raise RuntimeError("transient VoiceCraft 500")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        write_constant_wav(output_wav, seconds=float(job["end_sec"]) - float(job["start_sec"]), sample=16000)
        return {"test_only": True}


def test_speech_editing_can_write_partial_final_for_fallback(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_constant_wav(store.local_path(source_uri), seconds=2.0, sample=1000)
    store.write_json(
        deid_uri,
        {
            "edits": [
                {"edit_id": "edit_0001", "phi_type": "NAME", "speaker_id": "SPEAKER_00", "start_sec": 0.2, "end_sec": 0.5, "replacement_text": "Morgan"},
                {"edit_id": "edit_0002", "phi_type": "NAME", "speaker_id": "SPEAKER_00", "start_sec": 0.8, "end_sec": 1.1, "replacement_text": "Brookfield"},
            ]
        },
    )

    payload = SpeechEditingService(store, backend=PartiallyFailingSpanBackend()).edit_to_artifacts(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
        final_audio_uri=final_uri,
        options={"allow_partial_failures": True},
    )

    assert payload["status"] == "completed_with_failed_jobs"
    assert payload["num_failed_jobs"] == 1
    assert payload["failed_job_ids"] == ["voicecraft_job_0002"]
    assert payload["jobs"][0]["status"] == "synthesized"
    assert payload["jobs"][1]["status"] == "failed"
    assert store.exists(final_uri)
    persisted = store.read_json(jobs_uri)
    assert persisted["partial_failure_policy"] == "failed or validation-rejected spans preserved from source pending fallback"


def test_voicecraft_http_backend_retries_transient_failures(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.wav"
    output = tmp_path / "out.wav"
    write_wav(source, seconds=1.0)
    backend = VoiceCraftHTTPBackend(base_url="http://voicecraft.local")
    backend.preflight = lambda: {"ok": True}  # type: ignore[method-assign]
    calls = {"count": 0}

    def fake_post_json(url, payload, *, timeout_sec):
        calls["count"] += 1
        if calls["count"] == 1:
            raise SpeechEditingSetupError("VoiceCraft service request failed: HTTP 500 Internal Server Error: boom")
        write_wav(Path(payload["output"]), seconds=0.2)
        return {"ok": True}

    monkeypatch.setattr(speech_editing_module, "_post_json", fake_post_json)
    meta = backend.synthesize_edit(
        source_wav=source,
        output_wav=output,
        job={"start_sec": 0.0, "end_sec": 0.2, "replacement_text": "hello"},
        options={"max_retries": 1, "retry_backoff_sec": 0, "timeout_sec": 5},
    )

    assert calls["count"] == 2
    assert meta["attempt"] == 2
    assert output.exists()

def test_edit_jobs_include_target_text_from_deid_payload(tmp_path: Path, monkeypatch):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    write_wav(store.local_path(source_uri), seconds=1.0)
    store.write_json(
        deid_uri,
        {
            "deidentified_text": "hello [NAME]",
            "edits": [
                {
                    "edit_id": "edit_0001",
                    "phi_type": "NAME",
                    "start_sec": 0.1,
                    "end_sec": 0.4,
                    "original_text": "Alice",
                    "replacement_text": "[NAME]",
                }
            ],
        },
    )
    monkeypatch.setenv("AEGISVOX_SPEECH_EDITING_TARGET_TEXT_SCOPE", "full")
    payload = SpeechEditingService(store, backend=CopySpanBackend()).plan_jobs_to_artifact(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
    )
    assert payload["jobs"][0]["target_text"] == "hello [NAME]"


def test_voicecraft_command_backend_passes_target_text_placeholder(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "out.wav"
    record = tmp_path / "record.txt"
    script = tmp_path / "fake_voicecraft.py"
    write_wav(source, seconds=1.0)
    script.write_text(
        "import argparse, pathlib, wave\n"
        "p=argparse.ArgumentParser(); p.add_argument('--source'); p.add_argument('--output'); p.add_argument('--start'); p.add_argument('--end'); p.add_argument('--text'); p.add_argument('--target-text'); p.add_argument('--record'); args=p.parse_args()\n"
        "pathlib.Path(args.record).write_text(args.target_text)\n"
        "with wave.open(args.output, 'wb') as w:\n w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b'\\0\\0'*800)\n"
    )
    backend = VoiceCraftCommandBackend(
        command_template=(
            f"{sys.executable} {script} --source {{source}} --output {{output}} --start {{start}} --end {{end}} "
            f"--text {{text}} --target-text {{target_text}} --record {record}"
        )
    )
    backend.synthesize_edit(
        source_wav=source,
        output_wav=output,
        job={"start_sec": 0.1, "end_sec": 0.2, "replacement_text": "[NAME]", "target_text": "hello [NAME]"},
        options={"timeout_sec": 5},
    )
    assert output.exists() and output.stat().st_size > 0
    assert record.read_text() == "hello [NAME]"


def test_voicecraft_command_backend_uses_job_json_path_without_inline_json(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "out.wav"
    capture = tmp_path / "capture.json"
    write_wav(source)
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import json, sys, wave\n"
        "from pathlib import Path\n"
        "job=json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(sys.argv[2]).write_text(json.dumps({'target_text': job.get('target_text'), 'argv': sys.argv}))\n"
        "with wave.open(sys.argv[3], 'wb') as wav:\n"
        "    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000); wav.writeframes(b'\\0\\0'*800)\n"
    )
    backend = VoiceCraftCommandBackend(
        command_template=f"{sys.executable} {helper} {{job_json_path}} {capture} {{output}} --source {{source}} --start {{start}} --end {{end}} --text {{text}}",
    )

    metadata = backend.synthesize_edit(
        source_wav=source,
        output_wav=output,
        job={
            "start_sec": 0.0,
            "end_sec": 0.1,
            "replacement_text": "Synthetic",
            "target_text": "private target transcript should only be in temp file",
        },
        options={"timeout_sec": 10},
    )

    assert output.exists()
    assert "command_stdout_tail" in metadata
    captured = __import__("json").loads(capture.read_text())
    assert captured["target_text"] == "private target transcript should only be in temp file"
    # The helper argv gets only the temp path, not the JSON/transcript payload.
    assert all("private target transcript" not in arg for arg in captured["argv"])


def test_edit_jobs_default_to_span_target_text_for_speed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AEGISVOX_SPEECH_EDITING_TARGET_TEXT_SCOPE", raising=False)
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    write_wav(store.local_path(source_uri), seconds=1.0)
    store.write_json(
        deid_uri,
        {
            "deidentified_text": "a very long transcript [NAME]",
            "edits": [{"edit_id": "edit_0001", "phi_type": "NAME", "start_sec": 0.1, "end_sec": 0.4, "replacement_text": "[NAME]"}],
        },
    )
    payload = SpeechEditingService(store, backend=CopySpanBackend()).plan_jobs_to_artifact(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
    )
    assert payload["jobs"][0]["target_text"] == "uh [NAME]"
    assert payload["jobs"][0]["target_text_scope"] == "span"
    assert payload["jobs"][0]["short_span_render_policy"]["strategy"] == "neutral_filler_prefix"


def test_short_span_edit_job_uses_source_segment_context_when_available(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AEGISVOX_SPEECH_EDITING_TARGET_TEXT_SCOPE", raising=False)
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    write_wav(store.local_path(source_uri), seconds=2.0)
    store.write_json(
        deid_uri,
        {
            "edits": [
                {
                    "edit_id": "edit_0001",
                    "phi_type": "HIPAA_NAME",
                    "start_sec": 0.6,
                    "end_sec": 0.9,
                    "original_text": "Alice",
                    "replacement_text": "Claire",
                    "source_segment_text": "Bonjour Alice",
                    "source_language": "fr",
                    "replacement_language": "fr",
                }
            ],
        },
    )
    payload = SpeechEditingService(store, backend=CopySpanBackend()).plan_jobs_to_artifact(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
    )
    assert payload["jobs"][0]["target_text"] == "Bonjour Claire"
    assert payload["jobs"][0]["short_span_render_policy"]["strategy"] == "context_expanded_phrase"
    assert payload["jobs"][0]["short_span_render_policy"]["language"] == "fr"

class ReplacementAwareValidator:
    validator_name = "replacement-aware-test-validator"
    validator_version = "test-only"

    def __init__(self, pass_edit_ids=None):
        self.pass_edit_ids = set(pass_edit_ids or [])
        self.calls = []

    def preflight(self):
        return {"ok": True, "validator": self.validator_name}

    def validate_edit(self, *, source_wav: Path, edited_wav: Path, job, options):
        self.calls.append(job.get("edit_id"))
        found = str(job.get("edit_id")) in self.pass_edit_ids
        return {
            "status": "pass_replacement_found" if found else "fail_no_replacement_evidence",
            "replacement_found": found,
            "asr_text": str(job.get("replacement_text") if found else "unrelated speech"),
        }


def test_speech_editing_validation_gate_preserves_failed_span_for_fallback(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_constant_wav(store.local_path(source_uri), seconds=2.0, sample=1000)
    store.write_json(
        deid_uri,
        {
            "edits": [
                {"edit_id": "edit_0001", "phi_type": "NAME", "start_sec": 0.2, "end_sec": 0.5, "replacement_text": "Morgan"},
                {"edit_id": "edit_0002", "phi_type": "NAME", "start_sec": 0.8, "end_sec": 1.1, "replacement_text": "Brookfield"},
            ]
        },
    )
    validator = ReplacementAwareValidator(pass_edit_ids={"edit_0001"})

    payload = SpeechEditingService(store, backend=ConstantSpanBackend(sample=16000), edited_span_validator=validator).edit_to_artifacts(
        changed_identity_audio_uri=source_uri,
        deid_text_uri=deid_uri,
        output_jobs_uri=jobs_uri,
        output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
        final_audio_uri=final_uri,
        options={"require_validated_replacements": True, "allow_partial_failures": True},
    )

    assert validator.calls == ["edit_0001", "edit_0002"]
    assert payload["status"] == "completed_with_failed_jobs"
    assert payload["num_failed_jobs"] == 1
    assert payload["edited_span_validation"] == {
        "enabled": True,
        "required_for_stitching": True,
        "validator": "replacement-aware-test-validator",
        "num_validated": 2,
        "num_validation_failed": 1,
    }
    assert payload["jobs"][0]["status"] == "synthesized"
    assert payload["jobs"][1]["status"] == "failed_validation"
    samples = read_pcm16_samples(store.local_path(final_uri))
    assert any(value == 16000 for value in samples[int(0.2 * 8000) : int(0.5 * 8000)])
    # The failed validation interval is deliberately not stitched; source audio is preserved for fallback repair.
    assert all(value == 1000 for value in samples[int(0.8 * 8000) : int(1.1 * 8000)])


def test_speech_editing_validation_gate_fails_loud_without_partial_policy(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    deid_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    jobs_uri = "s3://aegisvox/runs/run_001/voicecraft_edit_jobs.json"
    final_uri = "s3://aegisvox/runs/run_001/final_sanitized.wav"
    write_constant_wav(store.local_path(source_uri), seconds=1.0, sample=1000)
    store.write_json(
        deid_uri,
        {"edits": [{"edit_id": "edit_0001", "phi_type": "NAME", "start_sec": 0.2, "end_sec": 0.5, "replacement_text": "Morgan"}]},
    )

    with pytest.raises(SpeechEditingError, match="speech editing failed"):
        SpeechEditingService(store, backend=ConstantSpanBackend(), edited_span_validator=ReplacementAwareValidator()).edit_to_artifacts(
            changed_identity_audio_uri=source_uri,
            deid_text_uri=deid_uri,
            output_jobs_uri=jobs_uri,
            output_span_prefix_uri="s3://aegisvox/runs/run_001/edited_spans",
            final_audio_uri=final_uri,
            options={"require_validated_replacements": True},
        )
    payload = store.read_json(jobs_uri)
    assert payload["status"] == "failed"
    assert payload["jobs"][0]["status"] == "failed_validation"
    assert not store.exists(final_uri)
