import wave
import sys
import types
from pathlib import Path

import pytest

from aegisvox.asr_alignment import (
    ASRAlignmentService,
    ASRSetupError,
    LocalFasterWhisperASRBackend,
    NVIDIACanaryASRBackend,
    OpenAIASRBackend,
    create_asr_backend,
    normalize_asr_output,
)
from aegisvox.storage import LocalArtifactStore


class FakeASRBackend:
    backend_name = "fake-asr"
    backend_version = "test"

    def preflight(self):
        return {"ok": True, "model": "fake-asr-model", "response_format": "verbose_json"}

    def transcribe(self, audio_path: Path):
        assert audio_path.exists()
        return {
            "text": "hello there goodbye",
            "segments": [
                {"text": "hello there", "start": 0.0, "end": 1.0},
                {"text": "goodbye", "start": 1.2, "end": 2.0},
            ],
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "there", "start": 0.5, "end": 1.0},
                {"word": "goodbye", "start": 1.2, "end": 2.0},
            ],
        }


def write_wav(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 16000)


def write_wav_seconds(path: Path, seconds: float, sample_rate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


class FakeTranscriptions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        audio_file = kwargs["file"]
        with wave.open(audio_file.name, "rb") as wav:
            duration = wav.getnframes() / float(wav.getframerate())
        index = len(self.calls)
        self.calls.append({"kwargs": dict(kwargs), "duration": duration})
        return {
            "text": f"chunk {index}",
            "segments": [{"text": f"segment {index}", "start": 0.1, "end": round(max(0.1, duration - 0.1), 3)}],
            "words": [{"word": f"word{index}", "start": 0.2, "end": round(max(0.2, duration - 0.2), 3)}],
        }


class FakeOpenAIClient:
    def __init__(self):
        self.audio = type("FakeAudio", (), {"transcriptions": FakeTranscriptions()})()


class ChunkingOpenAIASRBackend(OpenAIASRBackend):
    def __init__(self, client: FakeOpenAIClient):
        super().__init__()
        self.fake_client = client

    def preflight(self):
        return {"ok": True}

    def _client(self):
        return self.fake_client


def test_openai_asr_preflight_requires_runtime_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ASRSetupError, match="requires OPENAI_API_KEY or OPENAI_API_KEY"):
        OpenAIASRBackend().preflight()


def test_openai_asr_base_url_is_runtime_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
    assert OpenAIASRBackend()._base_url() == "https://us.api.openai.com/v1"


def test_openai_asr_model_and_format_are_runtime_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_ASR_MODEL", "whisper-1")
    assert OpenAIASRBackend()._model() == "whisper-1"
    assert OpenAIASRBackend()._response_format() == "verbose_json"
    assert OpenAIASRBackend()._timestamp_granularities() == ["word", "segment"]
    monkeypatch.setenv("OPENAI_ASR_RESPONSE_FORMAT", "json")
    assert OpenAIASRBackend()._response_format() == "json"
    assert OpenAIASRBackend()._timestamp_granularities() is None


def test_openai_asr_chunks_large_wav_and_offsets_timestamps(tmp_path, monkeypatch):
    audio_path = tmp_path / "long.wav"
    write_wav_seconds(audio_path, seconds=2.5)
    monkeypatch.setenv("AEGISVOX_ASR_MAX_UPLOAD_BYTES", "30000")
    monkeypatch.setenv("AEGISVOX_ASR_CHUNK_SECONDS", "1.0")
    client = FakeOpenAIClient()

    payload = ChunkingOpenAIASRBackend(client).transcribe(audio_path)

    assert payload["chunked_asr"] is True
    assert payload["text"] == "chunk 0 chunk 1 chunk 2"
    assert [chunk["duration_sec"] for chunk in payload["chunks"]] == [1.0, 1.0, 0.5]
    assert len(client.audio.transcriptions.calls) == 3
    assert [segment["start"] for segment in payload["segments"]] == [0.1, 1.1, 2.1]
    assert [word["start"] for word in payload["words"]] == [0.2, 1.2, 2.2]


def test_openai_asr_caps_chunk_size_by_upload_limit_for_wide_wav(tmp_path, monkeypatch):
    audio_path = tmp_path / "wide.wav"
    write_wav_seconds(audio_path, seconds=3.0, sample_rate=8000)
    monkeypatch.setenv("AEGISVOX_ASR_MAX_UPLOAD_BYTES", "20000")
    monkeypatch.setenv("AEGISVOX_ASR_CHUNK_SECONDS", "10.0")
    client = FakeOpenAIClient()

    payload = ChunkingOpenAIASRBackend(client).transcribe(audio_path)

    assert payload["chunked_asr"] is True
    assert len(payload["chunks"]) > 1
    assert max(chunk["bytes"] for chunk in payload["chunks"]) <= 20000
    assert len(client.audio.transcriptions.calls) == len(payload["chunks"])


def test_openai_asr_uses_single_request_when_under_upload_limit(tmp_path, monkeypatch):
    audio_path = tmp_path / "short.wav"
    write_wav_seconds(audio_path, seconds=0.5)
    monkeypatch.setenv("AEGISVOX_ASR_MAX_UPLOAD_BYTES", str(audio_path.stat().st_size + 1000))
    client = FakeOpenAIClient()

    payload = ChunkingOpenAIASRBackend(client).transcribe(audio_path)

    assert "chunked_asr" not in payload
    assert payload["text"] == "chunk 0"
    assert len(client.audio.transcriptions.calls) == 1


def test_create_asr_backend_selects_local_and_canary(monkeypatch):
    monkeypatch.setenv("AEGISVOX_ASR_BACKEND", "local_faster_whisper")
    assert isinstance(create_asr_backend(), LocalFasterWhisperASRBackend)
    assert isinstance(create_asr_backend("nvidia_canary"), NVIDIACanaryASRBackend)
    with pytest.raises(ASRSetupError, match="unknown AegisVox ASR backend"):
        create_asr_backend("not_real")


def test_local_faster_whisper_backend_transcribes_with_fake_module(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    write_wav_seconds(audio_path, seconds=0.5)

    class FakeWord:
        word = "hello"
        start = 0.0
        end = 0.2
        probability = 0.9

    class FakeSegment:
        id = 0
        text = " hello "
        start = 0.0
        end = 0.5
        words = [FakeWord()]

    class FakeInfo:
        language = "en"
        language_probability = 0.99
        duration = 0.5

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def transcribe(self, path, **kwargs):
            assert Path(path) == audio_path
            assert kwargs["word_timestamps"] is True
            return iter([FakeSegment()]), FakeInfo()

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setenv("AEGISVOX_ASR_DEVICE", "cpu")

    payload = LocalFasterWhisperASRBackend().transcribe(audio_path)

    assert payload["backend"] == "local-faster-whisper"
    assert payload["text"] == "hello"
    assert payload["segments"][0]["words"][0]["word"] == "hello"
    assert payload["language"] == "en"


def test_local_faster_whisper_retries_only_cuda_word_alignment_failure(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    write_wav_seconds(audio_path, seconds=0.5)

    class FakeSegment:
        id = 0
        text = " recovered transcript "
        start = 0.0
        end = 0.5
        words = []

    class FakeInfo:
        language = "fr"
        language_probability = 0.98
        duration = 0.5

    class BrokenAlignmentIterator:
        def __iter__(self):
            raise RuntimeError("parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal")

    class FakeWhisperModel:
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            self.calls.append(kwargs["word_timestamps"])
            if kwargs["word_timestamps"]:
                return BrokenAlignmentIterator(), FakeInfo()
            return iter([FakeSegment()]), FakeInfo()

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setenv("AEGISVOX_ASR_DEVICE", "cpu")

    payload = LocalFasterWhisperASRBackend().transcribe(audio_path)

    assert FakeWhisperModel.calls == [True, False]
    assert payload["text"] == "recovered transcript"
    assert payload["words"] == []
    assert payload["word_timestamps_available"] is False
    assert payload["word_timestamp_fallback"] == "cuda_alignment_invalid_device"


def test_local_faster_whisper_does_not_mask_other_failures(tmp_path, monkeypatch):
    audio_path = tmp_path / "input.wav"
    write_wav_seconds(audio_path, seconds=0.5)

    class BrokenIterator:
        def __iter__(self):
            raise RuntimeError("decoder out of memory")

    class FakeWhisperModel:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            type(self).calls += 1
            return BrokenIterator(), object()

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setenv("AEGISVOX_ASR_DEVICE", "cpu")

    with pytest.raises(ASRSetupError, match="decoder out of memory"):
        LocalFasterWhisperASRBackend().transcribe(audio_path)
    assert FakeWhisperModel.calls == 1


def test_normalize_asr_output_assigns_speakers():
    payload = normalize_asr_output(
        {"segments": [{"text": "hi", "start": 0, "end": 1}], "words": [{"word": "hi", "start": 0, "end": 1}]},
        [{"speaker_id": "SPEAKER_00", "start_sec": 0.0, "end_sec": 1.1}],
    )
    assert payload["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert payload["words"][0]["speaker_id"] == "SPEAKER_00"


def test_asr_alignment_service_writes_artifact(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    audio_uri = "s3://aegisvox/runs/run_001/changed_identity.wav"
    diarization_uri = "s3://aegisvox/runs/run_001/diarization.raw.json"
    output_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    write_wav(store.local_path(audio_uri))
    store.write_json(
        diarization_uri,
        {
            "segments": [
                {"speaker_id": "SPEAKER_00", "start_sec": 0.0, "end_sec": 1.1},
                {"speaker_id": "SPEAKER_01", "start_sec": 1.1, "end_sec": 2.1},
            ]
        },
    )
    service = ASRAlignmentService(store, backend=FakeASRBackend())
    payload = service.transcribe_to_artifact(audio_uri=audio_uri, output_asr_uri=output_uri, diarization_uri=diarization_uri)
    assert payload["backend"] == "fake-asr"
    assert payload["model"] == "fake-asr-model"
    assert payload["response_format"] == "verbose_json"
    assert payload["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert payload["segments"][1]["speaker_id"] == "SPEAKER_01"
    assert store.exists(output_uri)
