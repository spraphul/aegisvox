import json

import pytest

from aegisvox.storage import LocalArtifactStore
from aegisvox.text_deid import (
    DeIDPolicy,
    OpenAILLMTextDeIDBackend,
    TextDeIDService,
    TextDeIDSetupError,
    _segment_language,
    _transcript_language,
)


class FakeHIPAADetectorBackend:
    backend_name = "fake-llm-hipaa-detector"
    backend_version = "test-only"

    def __init__(self, edits=None):
        self.edits = [
            {
                "segment_index": 0,
                "phi_type": "HIPAA_NAME",
                "original_text": "John Smith",
                "confidence": 0.97,
                "rationale": "explicit patient name",
            },
            {
                "segment_index": 0,
                "phi_type": "HIPAA_PHONE",
                "original_text": "415-555-1212",
                "confidence": 0.99,
                "rationale": "phone number",
            },
            {
                "segment_index": 1,
                "phi_type": "HIPAA_EMAIL",
                "original_text": "john@example.com",
                "confidence": 0.99,
                "rationale": "email address",
            },
            {
                "segment_index": 1,
                "phi_type": "HIPAA_MEDICAL_RECORD_NUMBER",
                "original_text": "MRN A12345",
                "confidence": 0.96,
                "rationale": "medical record number",
            },
            {
                "segment_index": 1,
                "phi_type": "HIPAA_DATE",
                "original_text": "May 5, 2025",
                "confidence": 0.93,
                "rationale": "care-event date",
            },
        ] if edits is None else edits

    def detect_edits(self, transcript, policy):
        return list(self.edits)

    def plan_replacements(self, *, transcript, entities, policy):
        return [
            {
                "entity_id": entity["entity_id"],
                "replacement_text": {
                    "HIPAA_NAME": "Emily Carter",
                    "HIPAA_DATE": "March 12, 2026",
                    "HIPAA_GEOGRAPHIC_SUBDIVISION": "Riverside",
                    "HIPAA_PHONE": "[HIPAA_PHONE]",
                    "HIPAA_EMAIL": "[HIPAA_EMAIL]",
                    "HIPAA_MEDICAL_RECORD_NUMBER": "[HIPAA_MEDICAL_RECORD_NUMBER]",
                }.get(entity["phi_type"], f"[{entity['phi_type']}]") ,
                "rationale": "natural synthetic replacement",
            }
            for entity in entities
        ]


class UnsafeReplacementBackend(FakeHIPAADetectorBackend):
    backend_name = "unsafe-llm-hipaa-detector"
    backend_version = "test-only"
    last_entity_types = []

    def __init__(self, replacement_by_type):
        super().__init__()
        self.replacement_by_type = {
            "HIPAA_NAME": "Emily Carter",
            "HIPAA_DATE": "March 12, 2026",
            "HIPAA_GEOGRAPHIC_SUBDIVISION": "Riverside",
            "HIPAA_PHONE": "[HIPAA_PHONE]",
            "HIPAA_EMAIL": "[HIPAA_EMAIL]",
            "HIPAA_MEDICAL_RECORD_NUMBER": "[HIPAA_MEDICAL_RECORD_NUMBER]",
            **replacement_by_type,
        }

    def plan_replacements(self, *, transcript, entities, policy):
        UnsafeReplacementBackend.last_entity_types = [entity["phi_type"] for entity in entities]
        return [
            {
                "entity_id": entity["entity_id"],
                "replacement_text": self.replacement_by_type.get(entity["phi_type"], "Safe Replacement"),
                "rationale": "unsafe test replacement",
            }
            for entity in entities
        ]


def sample_transcript():
    return {
        "text": "My name is John Smith and my phone is 415-555-1212. Email john@example.com. MRN A12345. I visited May 5, 2025 at Cedar Hospital.",
        "segments": [
            {
                "speaker_id": "SPEAKER_00",
                "start_sec": 0.0,
                "end_sec": 3.0,
                "text": "My name is John Smith and my phone is 415-555-1212.",
            },
            {
                "speaker_id": "SPEAKER_01",
                "start_sec": 3.0,
                "end_sec": 6.0,
                "text": "Email john@example.com. MRN A12345. I visited May 5, 2025 at Cedar Hospital.",
            },
        ],
    }


def test_locked_language_hint_overrides_conflicting_asr_language():
    transcript = {
        "text": "Please call me tomorrow.",
        "language_hint": "fr",
        "language": "en",
        "segments": [
            {
                "text": "Please call me tomorrow.",
                "language_hint": "fr",
                "language": "en",
            }
        ],
    }

    assert _transcript_language(transcript) == "fr"
    assert _segment_language(transcript["segments"][0], transcript) == "fr"


def test_text_deid_requires_llm_hipaa_detector(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    store.write_json(asr_uri, sample_transcript())
    with pytest.raises(TextDeIDSetupError, match="requires a configured LLM HIPAA PHI backend"):
        TextDeIDService(store).deidentify_to_artifact(
            asr_uri=asr_uri,
            output_deid_uri="s3://aegisvox/runs/run_001/deid_text.json",
        )


def test_text_deid_service_writes_llm_hipaa_artifact(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    output_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    store.write_json(asr_uri, sample_transcript())
    payload = TextDeIDService(store, llm_backend=FakeHIPAADetectorBackend()).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri=output_uri,
    )
    assert payload["detector"]["policy"] == "hipaa_safe_harbor_llm_only"
    assert payload["detector"]["regex_detector_used"] is False
    assert payload["validator_summary"]["deterministic_edits"] == 0
    assert payload["validator_summary"]["llm_detector_edits"] == 5
    assert "[HIPAA_PHONE]" in payload["deidentified_text"]
    assert "john@example.com" not in payload["deidentified_text"]
    assert "A12345" not in payload["deidentified_text"]
    assert payload["policy_mode"] == "hybrid"
    assert payload["entity_consistency"]["same_original_same_replacement"] is True
    assert store.exists(output_uri)


def test_month_alone_is_not_edited_unless_llm_marks_it(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    output_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    store.write_json(
        asr_uri,
        {
            "text": "We met in January and then talked again in March.",
            "segments": [
                {
                    "speaker_id": "SPEAKER_00",
                    "start_sec": 0.0,
                    "end_sec": 2.0,
                    "text": "We met in January and then talked again in March.",
                }
            ],
        },
    )
    payload = TextDeIDService(store, llm_backend=FakeHIPAADetectorBackend(edits=[])).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri=output_uri,
    )
    assert payload["edits"] == []
    assert "January" in payload["deidentified_text"]
    assert "March" in payload["deidentified_text"]


def test_llm_detector_must_use_hipaa_entity_types(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    store.write_json(asr_uri, sample_transcript())
    with pytest.raises(TextDeIDSetupError, match="HIPAA Safe Harbor entity types only"):
        TextDeIDService(
            store,
            llm_backend=FakeHIPAADetectorBackend(
                edits=[
                    {
                        "segment_index": 0,
                        "phi_type": "LOCATION",
                        "original_text": "John Smith",
                        "confidence": 0.9,
                        "rationale": "bad legacy type",
                    }
                ]
            ),
        ).deidentify_to_artifact(
            asr_uri=asr_uri,
            output_deid_uri="s3://aegisvox/runs/run_001/deid_text.json",
        )


def test_llm_detector_absent_original_text_is_quarantined_not_fatal(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    output_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    store.write_json(asr_uri, sample_transcript())
    payload = TextDeIDService(
        store,
        llm_backend=FakeHIPAADetectorBackend(
            edits=[
                {
                    "segment_index": 0,
                    "phi_type": "HIPAA_NAME",
                    "original_text": "Jane Doe",
                    "confidence": 0.9,
                    "rationale": "hallucinated span",
                }
            ]
        ),
    ).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri=output_uri,
    )

    assert payload["edits"] == []
    assert payload["detector"]["invalid_detector_rows_quarantined"] == 1
    assert payload["validator_summary"]["llm_detector_rows_quarantined"] == 1
    assert payload["quarantined_detector_rows"][0]["reason"] == "original_text_not_present_in_claimed_segment_or_unique_transcript_segment"
    assert store.exists(output_uri)


def test_llm_detector_wrong_segment_index_is_relocated_when_unique(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    store.write_json(asr_uri, sample_transcript())
    payload = TextDeIDService(
        store,
        llm_backend=FakeHIPAADetectorBackend(
            edits=[
                {
                    "segment_index": 0,
                    "phi_type": "HIPAA_EMAIL",
                    "original_text": "john@example.com",
                    "confidence": 0.9,
                    "rationale": "right text wrong segment",
                }
            ]
        ),
    ).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri="s3://aegisvox/runs/run_001/deid_text.json",
    )

    assert len(payload["edits"]) == 1
    assert payload["edits"][0]["start_sec"] == 3.0
    assert "relocated from 0" in payload["edits"][0]["rationale"]
    assert payload["detector"]["invalid_detector_rows_quarantined"] == 0


def test_relexicalization_is_entity_consistent_for_repeated_originals(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    output_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    store.write_json(
        asr_uri,
        {
            "text": "My name is John Smith. My name is John Smith again.",
            "segments": [
                {"speaker_id": "SPEAKER_00", "start_sec": 0.0, "end_sec": 1.0, "text": "My name is John Smith."},
                {"speaker_id": "SPEAKER_00", "start_sec": 1.0, "end_sec": 2.0, "text": "My name is John Smith again."},
            ],
        },
    )
    backend = FakeHIPAADetectorBackend(
        edits=[
            {"segment_index": 0, "phi_type": "HIPAA_NAME", "original_text": "John Smith", "confidence": 0.98, "rationale": "name"},
            {"segment_index": 1, "phi_type": "HIPAA_NAME", "original_text": "John Smith", "confidence": 0.98, "rationale": "name"},
        ]
    )
    payload = TextDeIDService(store, policy=DeIDPolicy(mode="relexicalize"), llm_backend=backend).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri=output_uri,
    )
    name_replacements = {edit["replacement_text"] for edit in payload["edits"] if edit["phi_type"] == "HIPAA_NAME"}
    assert name_replacements == {"Alex Morgan"}
    assert "Alex Morgan" in payload["deidentified_text"]
    assert "John Smith" not in payload["deidentified_segments"][0]["text"]


def test_llm_replacement_plan_overrides_policy_relexicalization(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    output_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    store.write_json(asr_uri, sample_transcript())
    payload = TextDeIDService(
        store,
        policy=DeIDPolicy(mode="hybrid"),
        llm_backend=FakeHIPAADetectorBackend(),
    ).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri=output_uri,
        use_llm_replacements=True,
    )
    assert payload["replacement_plan_summary"]["replacement_source"] == "llm"
    assert "Emily Carter" in payload["deidentified_text"]
    assert "[HIPAA_PHONE]" in payload["deidentified_text"]
    assert "John Smith" not in payload["deidentified_text"]
    assert payload["edits"][0]["source_language"] == "en"


def test_llm_replacement_plan_falls_back_on_original_leakage(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    output_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    store.write_json(asr_uri, sample_transcript())
    payload = TextDeIDService(
        store,
        policy=DeIDPolicy(mode="hybrid"),
        llm_backend=UnsafeReplacementBackend({"HIPAA_NAME": "John Smith"}),
    ).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri=output_uri,
        use_llm_replacements=True,
    )
    assert "John Smith" not in payload["deidentified_text"]
    assert "Alex Morgan" in payload["deidentified_text"]
    assert any(
        edit["rationale"] == "deterministic safe replacement after unsafe LLM replacement"
        for edit in payload["edits"]
    )



def test_language_inferred_from_french_asr_text_without_metadata(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    output_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    store.write_json(
        asr_uri,
        {
            "text": "Bonjour monsieur Richard, je suis dans le Rhône avec le docteur.",
            "segments": [
                {
                    "speaker_id": "SPEAKER_00",
                    "start_sec": 0.0,
                    "end_sec": 3.0,
                    "text": "Bonjour monsieur Richard, je suis dans le Rhône avec le docteur.",
                }
            ],
        },
    )
    backend = UnsafeReplacementBackend({"HIPAA_NAME": "Richard"})
    backend.edits = [
        {"segment_index": 0, "phi_type": "HIPAA_NAME", "original_text": "Richard", "confidence": 0.95, "rationale": "name"}
    ]
    payload = TextDeIDService(
        store,
        policy=DeIDPolicy(mode="hybrid"),
        llm_backend=backend,
    ).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri=output_uri,
        use_llm_replacements=True,
    )
    assert payload["transcript_language"] == "fr"
    assert payload["edits"][0]["source_language"] == "fr"
    assert payload["edits"][0]["replacement_language"] == "fr"
    assert payload["edits"][0]["replacement_text"] != "Richard"

def test_language_aware_relexicalization_uses_french_deterministic_fallback(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    output_uri = "s3://aegisvox/runs/run_001/deid_text.json"
    store.write_json(
        asr_uri,
        {
            "language": "fr",
            "text": "Bonjour je suis Jean Dupont et je viens le 5 mai 2025.",
            "segments": [
                {
                    "speaker_id": "SPEAKER_00",
                    "start_sec": 0.0,
                    "end_sec": 3.0,
                    "language": "fr",
                    "text": "Bonjour je suis Jean Dupont et je viens le 5 mai 2025.",
                }
            ],
        },
    )
    backend = UnsafeReplacementBackend({"HIPAA_DATE": "March 12, 2026"})
    backend.edits = [
        {"segment_index": 0, "phi_type": "HIPAA_DATE", "original_text": "5 mai 2025", "confidence": 0.95, "rationale": "date"}
    ]
    payload = TextDeIDService(
        store,
        policy=DeIDPolicy(mode="hybrid"),
        llm_backend=backend,
    ).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri=output_uri,
        use_llm_replacements=True,
    )
    assert "le douze mars deux mille vingt-six" in payload["deidentified_text"]
    assert payload["edits"][0]["source_language"] == "fr"
    assert payload["edits"][0]["replacement_language"] == "fr"
    assert payload["edits"][0]["source_segment_text"].startswith("Bonjour")


def test_llm_replacement_plan_preserves_policy_masks_for_hybrid_identifiers(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asr_uri = "s3://aegisvox/runs/run_001/asr.changed_identity.json"
    store.write_json(asr_uri, sample_transcript())
    payload = TextDeIDService(
        store,
        policy=DeIDPolicy(mode="hybrid"),
        llm_backend=UnsafeReplacementBackend({"HIPAA_PHONE": "555-010-2222"}),
    ).deidentify_to_artifact(
        asr_uri=asr_uri,
        output_deid_uri="s3://aegisvox/runs/run_001/deid_text.json",
        use_llm_replacements=True,
    )
    sent_types = set(UnsafeReplacementBackend.last_entity_types)
    assert "HIPAA_PHONE" not in sent_types
    assert "HIPAA_EMAIL" not in sent_types
    assert "HIPAA_MEDICAL_RECORD_NUMBER" not in sent_types
    assert "[HIPAA_PHONE]" in payload["deidentified_text"]
    assert "415-555-1212" not in payload["deidentified_text"]


def test_openai_llm_backend_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(TextDeIDSetupError, match="requires OPENAI_API_KEY"):
        OpenAILLMTextDeIDBackend().detect_edits(sample_transcript(), DeIDPolicy())


def test_openai_llm_detector_parses_structured_output(monkeypatch):
    calls = {}

    class FakeResponses:
        def create(self, **kwargs):
            calls.update(kwargs)

            class Resp:
                output_text = (
                    '{"edits": ['
                    '{"segment_index": 0, "phi_type": "HIPAA_NAME", "original_text": "John Smith", '
                    '"confidence": 0.97, "rationale": "explicit name"}'
                    "]}"
                )

            return Resp()

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.responses = FakeResponses()

    fake_openai = type("FakeOpenAI", (), {"OpenAI": FakeClient})
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    edits = OpenAILLMTextDeIDBackend(model="gpt-5.4-test").detect_edits(sample_transcript(), DeIDPolicy())

    assert edits[0]["phi_type"] == "HIPAA_NAME"
    assert calls["model"] == "gpt-5.4-test"
    assert calls["text"]["format"]["type"] == "json_schema"
    assert calls["text"]["format"]["strict"] is True
    assert calls["text"]["format"]["schema"]["properties"]["edits"]["items"]["properties"]["phi_type"]["enum"]
    assert calls["client_kwargs"]["api_key"] == "sk-test"


def test_openai_llm_replacement_planner_parses_structured_output(monkeypatch):
    calls = {}

    class FakeResponses:
        def create(self, **kwargs):
            calls.update(kwargs)

            class Resp:
                output_text = (
                    '{"replacements": ['
                    '{"entity_id": "entity_0001", "replacement_text": "Emily Carter", "rationale": "synthetic name"}'
                    "]}"
                )

            return Resp()

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.responses = FakeResponses()

    fake_openai = type("FakeOpenAI", (), {"OpenAI": FakeClient})
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    backend = OpenAILLMTextDeIDBackend(model="gpt-5.4-test")
    replacements = backend.plan_replacements(
        transcript=sample_transcript(),
        entities=[{"entity_id": "entity_0001", "phi_type": "HIPAA_NAME", "original_text": "John Smith"}],
        policy=DeIDPolicy(mode="hybrid"),
    )

    assert replacements[0]["replacement_text"] == "Emily Carter"
    assert calls["model"] == "gpt-5.4-test"
    assert calls["text"]["format"]["type"] == "json_schema"
    assert calls["text"]["format"]["strict"] is True
    assert calls["client_kwargs"]["api_key"] == "sk-test"


def test_openai_llm_replacement_planner_uses_aegisvox_base_url(monkeypatch):
    calls = {}

    class FakeResponses:
        def create(self, **kwargs):
            calls.update(kwargs)

            class Resp:
                output_text = (
                    '{"replacements": ['
                    '{"entity_id": "entity_0001", "replacement_text": "Emily Carter", "rationale": "synthetic name"}'
                    "]}"
                )

            return Resp()

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.responses = FakeResponses()

    fake_openai = type("FakeOpenAI", (), {"OpenAI": FakeClient})
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AEGISVOX_OPENAI_BASE_URL", "https://us.api.openai.com/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    OpenAILLMTextDeIDBackend(model="gpt-5.4-test").plan_replacements(
        transcript=sample_transcript(),
        entities=[{"entity_id": "entity_0001", "phi_type": "HIPAA_NAME", "original_text": "John Smith"}],
        policy=DeIDPolicy(mode="hybrid"),
    )

    assert calls["client_kwargs"]["base_url"] == "https://us.api.openai.com/v1"


def test_openai_llm_replacement_planner_masks_hybrid_high_risk_without_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    backend = OpenAILLMTextDeIDBackend(model="gpt-5.4-test")
    replacements = backend.plan_replacements(
        transcript=sample_transcript(),
        entities=[
            {
                "entity_id": "entity_0001",
                "phi_type": "HIPAA_AGE_OVER_89",
                "original_text": "ninety two",
            }
        ],
        policy=DeIDPolicy(mode="hybrid"),
    )

    assert replacements == [
        {
            "entity_id": "entity_0001",
            "replacement_text": "[HIPAA_AGE_OVER_89]",
            "rationale": "hybrid policy mask for high-risk HIPAA identifier",
        }
    ]


def test_openai_llm_replacement_planner_relexicalizes_age_over_89_without_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    backend = OpenAILLMTextDeIDBackend(model="gpt-5.4-test")
    replacements = backend.plan_replacements(
        transcript=sample_transcript(),
        entities=[
            {
                "entity_id": "entity_0001",
                "phi_type": "HIPAA_AGE_OVER_89",
                "original_text": "ninety two",
            }
        ],
        policy=DeIDPolicy(mode="relexicalize"),
    )

    assert replacements == [
        {
            "entity_id": "entity_0001",
            "replacement_text": "seventy eight",
            "rationale": "deterministic safe age below ninety for HIPAA age-over-89",
        }
    ]


def test_openai_compatible_client_kwargs_supports_openai_compat_bearer_and_headers(monkeypatch):
    from aegisvox.text_deid import _openai_compatible_client_kwargs

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "local-bearer-token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai_compat.example/openai/v1")
    monkeypatch.setenv("OPENAI_COMPAT_SERVICE_NAME", "ai-factory-azure")
    monkeypatch.setenv("AEGISVOX_OPENAI_COMPAT_SET_X_MODEL", "true")
    kwargs = _openai_compatible_client_kwargs("gpt-5.4")
    assert kwargs["api_key"] == "local-bearer-token"
    assert kwargs["base_url"] == "https://openai_compat.example/openai/v1"
    assert kwargs["default_headers"]["service-name"] == "ai-factory-azure"
    assert kwargs["default_headers"]["X-Model"] == "gpt-5.4"


def test_raw_openai_compat_chat_fallback_detects_json(monkeypatch):
    import urllib.request
    from aegisvox.text_deid import OpenAILLMTextDeIDBackend, DeIDPolicy

    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai_compat.example")
    monkeypatch.setenv("AEGISVOX_OPENAI_COMPAT_RAW", "1")
    monkeypatch.setenv("OPENAI_COMPAT_API_MODE", "chat")
    monkeypatch.setenv("OPENAI_COMPAT_SERVICE_NAME", "ai-factory-azure")

    captured = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "{\\"edits\\": []}"}}]}'

    def fake_urlopen(request, timeout=None):
        captured.update(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "json": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    edits = OpenAILLMTextDeIDBackend(model="gpt-5.4").detect_edits(sample_transcript(), DeIDPolicy())
    assert edits == []
    assert captured["url"] == "https://openai_compat.example/openai/deployments/gpt-5.4/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["headers"]["Service-name"] == "ai-factory-azure"


def test_raw_openai_compat_chat_retries_unsupported_phi_type_without_mapping(monkeypatch):
    import urllib.request
    from aegisvox.text_deid import OpenAILLMTextDeIDBackend, DeIDPolicy

    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai_compat.example")
    monkeypatch.setenv("AEGISVOX_OPENAI_COMPAT_RAW", "1")
    monkeypatch.setenv("OPENAI_COMPAT_API_MODE", "chat")

    calls = []

    class Resp:
        def __init__(self, content):
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": self._content}}]}).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        calls.append(json.loads(request.data.decode("utf-8")))
        if len(calls) == 1:
            return Resp(
                '{"edits": ['
                '{"segment_index": 1, "phi_type": "HIPAA_ORGANIZATION", "original_text": "Cedar Hospital", '
                '"confidence": 0.92, "rationale": "invalid invented type"}'
                "]}"
            )
        return Resp(
            '{"edits": ['
            '{"segment_index": 1, "phi_type": "HIPAA_OTHER_UNIQUE_IDENTIFIER", "original_text": "Cedar Hospital", '
            '"confidence": 0.92, "rationale": "facility name treated as unique identifier"}'
            "]}"
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    edits = OpenAILLMTextDeIDBackend(model="gpt-5.4").detect_edits(sample_transcript(), DeIDPolicy())

    assert len(calls) == 2
    assert "CORRECTION REQUIRED" in calls[1]["messages"][0]["content"]
    assert edits[0]["phi_type"] == "HIPAA_OTHER_UNIQUE_IDENTIFIER"


def test_raw_openai_compat_responses_fallback_uses_x_model(monkeypatch):
    import urllib.request
    from aegisvox.text_deid import OpenAILLMTextDeIDBackend, DeIDPolicy

    monkeypatch.setenv("OPENAI_API_KEY", "token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai_compat.example")
    monkeypatch.setenv("AEGISVOX_OPENAI_COMPAT_RAW", "1")
    monkeypatch.setenv("OPENAI_COMPAT_API_MODE", "responses")

    captured = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "output_text": (
                    '{"replacements": ['
                    '{"entity_id": "entity_0001", "replacement_text": "Emily Carter", "rationale": "synthetic name"}'
                    "]}"
                )
            }).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        captured.update(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "json": json.loads(request.data.decode("utf-8")),
            }
        )
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    replacements = OpenAILLMTextDeIDBackend(model="gpt-5.4").plan_replacements(
        transcript=sample_transcript(),
        entities=[{"entity_id": "entity_0001", "phi_type": "HIPAA_NAME", "original_text": "John Smith"}],
        policy=DeIDPolicy(),
    )
    assert replacements[0]["replacement_text"] == "Emily Carter"
    assert captured["url"] == "https://openai_compat.example/openai/responses"
    assert captured["headers"]["X-model"] == "gpt-5.4"
