from pathlib import Path

from aegisvox.evaluation import EvaluationInputs, EvaluationPolicy, EvaluationService
from aegisvox.provenance import SmokeProvenanceInputs, build_smoke_provenance_manifest
from aegisvox.storage import LocalArtifactStore


def _write_fixture(store: LocalArtifactStore, prefix: str = "s3://aegisvox/run1") -> EvaluationInputs:
    uris = EvaluationInputs(
        speaker_profiles_uri=f"{prefix}/speaker_profiles.json",
        assignments_uri=f"{prefix}/speaker_target_map.json",
        converted_segments_uri=f"{prefix}/segments.converted.json",
        stitch_report_uri=f"{prefix}/stitch_report.json",
        asr_uri=f"{prefix}/asr.changed_identity.json",
        deid_text_uri=f"{prefix}/deid_text.json",
        voicecraft_edit_jobs_uri=f"{prefix}/voicecraft_edit_jobs.json",
        final_sanitized_audio_uri=f"{prefix}/final_sanitized.wav",
        speaker_privacy_metrics_uri=f"{prefix}/speaker_privacy_metrics.json",
    )
    store.write_json(uris.speaker_profiles_uri, {"speakers": {"S1": {"total_duration_sec": 1.0}}})
    store.write_json(
        uris.assignments_uri,
        {
            "policy": {"min_source_target_distance": 0.55, "min_target_target_distance": 0.30},
            "assignments": {
                "S1": {
                    "target_voice_id": "T1",
                    "target_voice_sample_id": "TS1",
                    "source_target_distance": 0.7,
                    "status": "assigned",
                }
            },
        },
    )
    store.write_json(
        uris.converted_segments_uri,
        {
            "backend": "seed-vc",
            "segments": [
                {
                    "segment_id": "seg1",
                    "speaker_id": "S1",
                    "conversion_status": "converted",
                    "converted_audio_uri": f"{prefix}/seg1.wav",
                }
            ],
        },
    )
    store.write_json(
        uris.stitch_report_uri,
        {
            "segments": [{"segment_id": "seg1", "stitch_status": "stitched", "duration_drift_sec": 0.01}],
            "excessive_duration_drift": [],
        },
    )
    store.write_json(
        uris.asr_uri,
        {"text": "hello world", "segments": [{"text": "hello world"}], "words": [{"word": "hello", "start_sec": 0.0, "end_sec": 0.2}]},
    )
    store.write_json(
        uris.deid_text_uri,
        {"deidentified_text": "hello world", "edits": [], "validator_summary": {"unresolved_high_confidence_phi": 0}},
    )
    store.write_json(
        uris.voicecraft_edit_jobs_uri,
        {
            "status": "no_edits_passthrough",
            "backend": None,
            "backend_version": None,
            "num_jobs": 0,
            "num_unlocalized_edits": 0,
            "jobs": [],
            "unlocalized_edits": [],
            "final_sanitized_audio_uri": uris.final_sanitized_audio_uri,
        },
    )
    store.write_bytes(uris.final_sanitized_audio_uri, b"final audio placeholder")
    store.write_json(
        uris.speaker_privacy_metrics_uri,
        {
            "backend_version": "speechbrain/spkrec-ecapa-voxceleb",
            "speaker_metrics": [
                {
                    "speaker_id": "S1",
                    "status": "computed",
                    "source_converted_similarity": 0.2,
                    "target_converted_similarity": 0.8,
                }
            ],
            "summary": {
                "num_speakers": 1,
                "num_computed": 1,
                "max_source_converted_similarity": 0.2,
                "min_target_converted_similarity": 0.8,
            },
        },
    )
    return uris


def test_smoke_provenance_manifest_contains_required_service_versions(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    inputs = _write_fixture(store)
    run_manifest_uri = "s3://aegisvox/run1/manifest.json"

    manifest = build_smoke_provenance_manifest(
        store,
        SmokeProvenanceInputs(
            run_id="run1",
            speaker_profiles_uri=inputs.speaker_profiles_uri,
            assignments_uri=inputs.assignments_uri,
            converted_segments_uri=inputs.converted_segments_uri,
            stitch_report_uri=inputs.stitch_report_uri,
            asr_uri=inputs.asr_uri,
            deid_text_uri=inputs.deid_text_uri,
            voicecraft_edit_jobs_uri=inputs.voicecraft_edit_jobs_uri,
            speaker_privacy_metrics_uri=inputs.speaker_privacy_metrics_uri,
            voice_pool_id="openswara_smoke",
            voice_pool_version="v1",
            openai_asr_model="whisper-1",
        ),
    )
    store.write_json(run_manifest_uri, manifest)

    required = {
        "orchestrator",
        "diarization_service",
        "pyannote",
        "segmentation",
        "speaker_profiling",
        "speechbrain-ecapa",
        "assignment",
        "voice_pool",
        "voice_conversion",
        "seed-vc",
        "stitching",
        "asr_alignment",
        "openai-asr",
        "text_deid",
        "speech_editing",
        "speech_editing_backend",
    }
    assert required.issubset(set(manifest["service_versions"]))
    assert manifest["service_versions"]["voice_pool"] == "openswara_smoke:v1"
    assert manifest["service_versions"]["openai-asr"] == "whisper-1"
    assert manifest["artifact_status"]["speaker_profiles.json"]["exists"] is True

    eval_inputs = EvaluationInputs(**{**inputs.__dict__, "run_manifest_uri": run_manifest_uri})
    report = EvaluationService(store, EvaluationPolicy(require_service_versions=True)).evaluate_to_artifacts(
        inputs=eval_inputs,
        output_report_uri="s3://aegisvox/run1/evaluation_report.json",
        output_markdown_uri="s3://aegisvox/run1/evaluation_report.md",
        output_pass_fail_uri="s3://aegisvox/run1/pass_fail.json",
    )
    assert report["overall_status"] == "pass"
    assert next(g for g in report["gates"] if g["name"] == "service_version_audit")["status"] == "pass"
