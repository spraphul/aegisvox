import pytest

from aegisvox.assignment import AssignmentError, AssignmentPolicy, AssignmentService
from aegisvox.schema import PrivacyMode
from aegisvox.storage import LocalArtifactStore
from aegisvox.voice_pool import VoicePoolIndex, VoicePoolManifest, VoicePoolSample, VoiceSearchResult, write_embedding_index, write_voice_pool_manifest


def make_pool(tmp_path, embeddings=None):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="assign-fixture-v1",
        samples=[
            VoicePoolSample(voice_id="target_a", sample_id="sample_a", audio_uri="s3://aegisvox/pool/a.wav", language="en", speaking_rate="moderate"),
            VoicePoolSample(voice_id="target_b", sample_id="sample_b", audio_uri="s3://aegisvox/pool/b.wav", language="en", speaking_rate="moderate"),
            VoicePoolSample(voice_id="target_c", sample_id="sample_c", audio_uri="s3://aegisvox/pool/c.wav", language="en", speaking_rate="moderate"),
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    index_path = tmp_path / "index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(
        index_path,
        embeddings
        or {
            "sample_a": [-1.0, 0.0, 0.0],
            "sample_b": [0.0, -1.0, 0.0],
            "sample_c": [0.0, 0.0, -1.0],
        },
    )
    return VoicePoolIndex.from_paths(manifest_path, index_path)


def make_gender_pool(tmp_path):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="gender-fixture-v1",
        samples=[
            VoicePoolSample(
                voice_id="english_male_open_swara_001",
                sample_id="english_male_open_swara_001",
                audio_uri="s3://aegisvox/pool/male.wav",
                language="en",
                metadata={"repo_file": "voices/english/male/english_male_open_swara_001.wav"},
            ),
            VoicePoolSample(
                voice_id="english_female_open_swara_001",
                sample_id="english_female_open_swara_001",
                audio_uri="s3://aegisvox/pool/female.wav",
                language="en",
                metadata={"repo_file": "voices/english/female/english_female_open_swara_001.wav"},
            ),
        ],
    )
    manifest_path = tmp_path / "gender_manifest.json"
    index_path = tmp_path / "gender_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(
        index_path,
        {
            # Both satisfy privacy distance, but male is farther. Gender
            # preservation should still pick the female target.
            "english_male_open_swara_001": [-1.0, 0.0, 0.0],
            "english_female_open_swara_001": [-0.8, 0.6, 0.0],
        },
    )
    return VoicePoolIndex.from_paths(manifest_path, index_path)


def make_accent_pool(tmp_path):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="accent-fixture-v1",
        samples=[
            VoicePoolSample(
                voice_id="english_female_open_swara_001",
                sample_id="english_female_open_swara_001",
                audio_uri="s3://aegisvox/pool/english_female.wav",
                language="en",
                metadata={"repo_file": "voices/english/female/english_female_open_swara_001.wav"},
            ),
            VoicePoolSample(
                voice_id="arabic_female_english_open_swara_001",
                sample_id="arabic_female_english_open_swara_001",
                audio_uri="s3://aegisvox/pool/arabic_female_english.wav",
                language="en",
                metadata={"repo_file": "voices/arabic/female/arabic_female_english_open_swara_001.wav"},
            ),
        ],
    )
    manifest_path = tmp_path / "accent_manifest.json"
    index_path = tmp_path / "accent_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(
        index_path,
        {
            # English accent is farther, but Arabic accent should be selected
            # when source accent_group is known.
            "english_female_open_swara_001": [-1.0, 0.0, 0.0],
            "arabic_female_english_open_swara_001": [-0.7, 0.7, 0.0],
        },
    )
    return VoicePoolIndex.from_paths(manifest_path, index_path)


def write_profiles(store, profiles_uri="s3://aegisvox/runs/run_001/speaker_profiles.json"):
    emb0 = "s3://aegisvox/runs/run_001/speaker_embeddings/SPEAKER_00.json"
    emb1 = "s3://aegisvox/runs/run_001/speaker_embeddings/SPEAKER_01.json"
    store.write_json(emb0, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(emb1, {"embedding": [0.0, 1.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "duration_sec": 1.0,
                    "source_audio_uri": "s3://aegisvox/runs/run_001/speaker_audio/SPEAKER_00.wav",
                    "source_embedding_uri": emb0,
                    "language": "en",
                    "speaking_rate": "moderate",
                },
                "SPEAKER_01": {
                    "duration_sec": 1.0,
                    "source_audio_uri": "s3://aegisvox/runs/run_001/speaker_audio/SPEAKER_01.wav",
                    "source_embedding_uri": emb1,
                    "language": "en",
                    "speaking_rate": "moderate",
                },
            }
        },
    )
    return profiles_uri


def test_assignment_policy_modes():
    assert AssignmentPolicy.for_privacy_mode(PrivacyMode.LOW).min_source_target_distance < AssignmentPolicy.for_privacy_mode(PrivacyMode.HIGH).min_source_target_distance
    assert AssignmentPolicy.for_privacy_mode("medium").preserve_speaking_rate is True


def test_assignment_unique_targets_and_writes_artifact(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    profiles_uri = write_profiles(store)
    service = AssignmentService(store, make_pool(tmp_path), AssignmentPolicy.for_privacy_mode("medium"))
    output_uri = "s3://aegisvox/runs/run_001/speaker_target_map.json"
    report_uri = "s3://aegisvox/runs/run_001/assignment_report.md"
    payload = service.assign_to_artifact(profiles_uri, output_uri, output_report_uri=report_uri)
    assert set(payload["assignments"]) == {"SPEAKER_00", "SPEAKER_01"}
    targets = [row["target_voice_id"] for row in payload["assignments"].values()]
    assert len(targets) == len(set(targets))
    assert store.exists(output_uri)
    assert store.exists(report_uri)
    assert payload["assignments"]["SPEAKER_00"]["source_target_distance"] >= 0.55
    assert payload["target_target_summary"]["num_pairs"] == 1
    assert payload["target_target_summary"]["min_target_target_distance"] >= 0.30
    assert payload["target_target_distances"][0]["left_target_sample_id"]
    report = store.read_bytes(report_uri).decode("utf-8")
    assert "# Pseudo-Speaker Assignment Report" in report
    assert "SPEAKER_00" in report
    assert "source-target distance" in report.lower()
    assert "raw embeddings" in report.lower()


def test_assignment_enforces_target_target_distinctiveness(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    profiles_uri = write_profiles(store)
    index = make_pool(
        tmp_path,
        embeddings={
            "sample_a": [-1.0, 0.0, 0.0],
            "sample_b": [-0.99, 0.01, 0.0],  # too close to sample_a target
            "sample_c": [0.0, 0.0, -1.0],
        },
    )
    policy = AssignmentPolicy.for_privacy_mode("medium")
    policy = AssignmentPolicy(
        privacy_mode=policy.privacy_mode,
        min_source_target_distance=policy.min_source_target_distance,
        min_target_target_distance=0.5,
        candidate_top_k=policy.candidate_top_k,
        preserve_language=policy.preserve_language,
        preserve_speaking_rate=policy.preserve_speaking_rate,
    )
    payload = AssignmentService(store, index, policy).assign_to_artifact(profiles_uri, "s3://aegisvox/runs/run_001/speaker_target_map.json")
    chosen_samples = [row["target_voice_sample_id"] for row in payload["assignments"].values()]
    assert "sample_b" not in chosen_samples or "sample_a" not in chosen_samples


def test_assignment_backtracks_without_relaxing_constraints_when_greedy_dead_ends(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    profiles_uri = write_profiles(store)
    index = make_pool(
        tmp_path,
        embeddings={
            "sample_a": [1.0, 0.0, 0.0],
            "sample_b": [0.0, 1.0, 0.0],
            "sample_c": [0.0, 0.0, 1.0],
        },
    )

    def result(sample_id, score):
        sample = index.samples_by_id[sample_id]
        return VoiceSearchResult(
            voice_id=sample.voice_id,
            sample_id=sample.sample_id,
            audio_uri=sample.audio_uri,
            score=score,
            cosine_similarity=0.0,
            source_target_distance=1.0,
            assignment_reason="test candidate",
            metadata={},
        )

    def search(request):
        # Greedy takes A for speaker 0. Speaker 1 can only use A, so the greedy
        # path dead-ends. B-for-speaker-0 plus A-for-speaker-1 is feasible and
        # satisfies the unchanged one-to-one and target-distance constraints.
        candidates = [result("sample_a", 2.0), result("sample_b", 1.0)] if request.query_embedding[0] else [result("sample_a", 2.0)]
        return [row for row in candidates if row.voice_id not in request.exclude_voice_ids]

    index.search = search
    policy = AssignmentPolicy(
        privacy_mode=PrivacyMode.HIGH,
        candidate_top_k=75,
        min_source_target_distance=0.0,
        min_target_target_distance=0.5,
        preserve_language=False,
        preserve_gender_presentation=False,
        preserve_accent_group=False,
        preserve_geography_region=False,
        preserve_tone=False,
        preserve_vocal_weight=False,
        preserve_resonance=False,
        preserve_voice_maturity=False,
        use_hybrid_audio_traits=False,
        reject_child_like_targets=False,
        accent_style_constrained=False,
        require_native_english_targets=False,
        require_native_english_for_north_america=False,
    )
    payload = AssignmentService(store, index, policy).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_001/speaker_target_map.json",
    )
    assert payload["assignment_solver"] == "global_backtracking_exact_constraints_v1"
    assert payload["assignments"]["SPEAKER_00"]["target_voice_sample_id"] == "sample_b"
    assert payload["assignments"]["SPEAKER_01"]["target_voice_sample_id"] == "sample_a"
    assert payload["target_target_summary"]["min_target_target_distance"] >= 0.5


def test_assignment_fails_when_no_candidate_meets_threshold(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    profiles_uri = write_profiles(store)
    index = make_pool(tmp_path, embeddings={"sample_a": [1, 0, 0], "sample_b": [1, 0, 0], "sample_c": [1, 0, 0]})
    policy = AssignmentPolicy.for_privacy_mode("high")
    with pytest.raises(AssignmentError, match="no valid pseudo-speaker candidate"):
        AssignmentService(store, index, policy).assign_to_artifact(profiles_uri, "s3://aegisvox/runs/run_001/speaker_target_map.json")


def test_assignment_preserves_known_gender_presentation(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_gender/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_gender/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "duration_sec": 1.0,
                    "source_audio_uri": "s3://aegisvox/runs/run_gender/speaker_audio/SPEAKER_00.wav",
                    "source_embedding_uri": emb,
                    "language": "en",
                    "gender_presentation": "female",
                }
            }
        },
    )
    payload = AssignmentService(store, make_gender_pool(tmp_path), AssignmentPolicy.for_privacy_mode("medium")).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_gender/speaker_target_map.json",
    )
    row = payload["assignments"]["SPEAKER_00"]
    assert row["target_voice_id"] == "english_female_open_swara_001"
    assert row["source_gender_presentation"] == "female"
    assert row["target_gender_presentation"] == "female"
    assert row["gender_match_status"] == "match"
    assert payload["policy"]["preserve_gender_presentation"] is True


def test_assignment_uses_source_accent_as_soft_signal_not_hard_filter(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_accent/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_accent/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "duration_sec": 1.0,
                    "source_audio_uri": "s3://aegisvox/runs/run_accent/speaker_audio/SPEAKER_00.wav",
                    "source_embedding_uri": emb,
                    "language": "en",
                    "gender_presentation": "female",
                    "accent_group": "arabic",
                }
            }
        },
    )
    payload = AssignmentService(store, make_accent_pool(tmp_path), AssignmentPolicy.for_privacy_mode("medium")).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_accent/speaker_target_map.json",
    )
    row = payload["assignments"]["SPEAKER_00"]
    assert row["target_voice_id"] == "english_female_open_swara_001"
    assert row["source_accent_group"] == "arabic"
    assert row["target_accent_group"] == "english"
    assert row["accent_match_status"] == "mismatch"
    assert payload["policy"]["preserve_accent_group"] is True
    assert payload["policy"]["hard_filter_source_audio_traits"] is False
    assert payload["retrieval_contract"]["source_gpt_audio_traits"] == "soft_scoring_only"



def test_assignment_profile_query_traits_respect_policy_and_top_level_override():
    from aegisvox.assignment import _query_audio_traits_from_profile

    profile = {
        "language": "english",
        "gender_presentation": "female",
        "accent_group": "top_level_indian_english",
        "metadata": {
            "audio_traits": {
                "accent_group": "metadata_american_english",
                "geography_region": "south_asia",
                "tone": "conversational",
                "energy": "high",
                "speaking_rate": "fast",
                "pitch_band": "mid",
            }
        },
    }
    policy = AssignmentPolicy(
        preserve_language=True,
        preserve_gender_presentation=True,
        preserve_accent_group=True,
        preserve_geography_region=True,
        preserve_tone=True,
        preserve_energy=False,
        preserve_speaking_rate=False,
        preserve_pitch_band=False,
        use_hybrid_audio_traits=True,
    )
    traits = _query_audio_traits_from_profile(profile, policy)
    assert traits == {
        "language": "english",
        "gender_presentation": "female",
        "accent_group": "top_level_indian_english",
        "geography_region": "south_asia",
        "tone": "conversational",
    }

    disabled = _query_audio_traits_from_profile(profile, AssignmentPolicy(use_hybrid_audio_traits=False))
    assert disabled == {}


def test_assignment_report_includes_hybrid_trait_weights_for_audit(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    profiles_uri = write_profiles(store)
    policy = AssignmentPolicy(
        candidate_top_k=3,
        trait_weight=2.5,
        vector_weight=0.75,
        trait_weights={"accent_group": 2.0, "tone": 1.0, "geography_region": 0.5},
    )
    report_uri = "s3://aegisvox/runs/run_001/assignment_report.md"
    payload = AssignmentService(store, make_pool(tmp_path), policy).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_001/speaker_target_map.json",
        output_report_uri=report_uri,
    )
    assert payload["policy"]["trait_weights"]["accent_group"] == 2.0
    report = store.read_bytes(report_uri).decode("utf-8")
    assert "Use hybrid audio traits: True" in report
    assert "Vector weight: 0.75" in report
    assert "Trait weight: 2.5" in report
    assert "Trait weights" in report
    assert "accent_group" in report

def test_assignment_uses_enriched_audio_traits_for_hybrid_matching(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="hybrid-fixture-v1",
        samples=[
            VoicePoolSample(
                voice_id="far_nonmatching_voice",
                sample_id="far_nonmatching_sample",
                audio_uri="s3://aegisvox/pool/far.wav",
                language="en",
                metadata={
                    "repo_file": "voices/english/female/far.wav",
                    "audio_traits": {"accent_group": "general_american", "tone": "calm", "gender_presentation": "female", "language": "en"},
                },
            ),
            VoicePoolSample(
                voice_id="closer_matching_voice",
                sample_id="closer_matching_sample",
                audio_uri="s3://aegisvox/pool/closer.wav",
                language="en",
                metadata={
                    "repo_file": "voices/english/female/closer.wav",
                    "audio_traits": {"accent_group": "neutral", "tone": "calm", "gender_presentation": "female", "language": "en"},
                    "audio_trait_extraction": {"model": "mock", "confidence": None},
                },
            ),
        ],
    )
    manifest_path = tmp_path / "hybrid_manifest.json"
    index_path = tmp_path / "hybrid_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(
        index_path,
        {
            # far target has higher source-target distance but wrong enriched accent;
            # hybrid trait search should preserve the neutral accent profile.
            "far_nonmatching_sample": [-1.0, 0.0, 0.0],
            "closer_matching_sample": [-0.7, 0.7, 0.0],
        },
    )
    index = VoicePoolIndex.from_paths(manifest_path, index_path)
    emb = "s3://aegisvox/runs/run_hybrid/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_hybrid/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "duration_sec": 1.0,
                    "source_audio_uri": "s3://aegisvox/runs/run_hybrid/speaker_audio/SPEAKER_00.wav",
                    "source_embedding_uri": emb,
                    "language": "en",
                    "gender_presentation": "female",
                    "metadata": {"audio_traits": {"accent_group": "neutral", "tone": "calm"}},
                }
            }
        },
    )
    payload = AssignmentService(store, index, AssignmentPolicy.for_privacy_mode("medium")).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_hybrid/speaker_target_map.json",
    )
    row = payload["assignments"]["SPEAKER_00"]
    assert row["target_voice_id"] == "closer_matching_voice"
    assert row["source_audio_traits"]["accent_group"] == "neutral"
    assert row["target_audio_traits"]["accent_group"] == "neutral"
    assert row["query_trait_match"]["matched"]["tone"] == "calm"
    explanation = row["hybrid_search_explanation"]
    assert explanation["hard_guardrails"]["gender_presentation"]["status"] == "matched"
    assert explanation["hard_guardrails"]["accent_group"]["status"] == "matched"
    assert explanation["soft_rerank"]["tone"]["status"] == "matched"
    assert explanation["warnings"] == []
    assert payload["policy"]["use_hybrid_audio_traits"] is True
    assert payload["policy"]["trait_weight"] == 2.0


def test_assignment_rejects_missing_known_hard_trait_candidate(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_failclosed/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_failclosed/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "duration_sec": 1.0,
                    "source_audio_uri": "s3://aegisvox/runs/run_failclosed/speaker_audio/SPEAKER_00.wav",
                    "source_embedding_uri": emb,
                    "language": "english",
                    "gender_presentation": "female",
                    "accent_group": "american_english",
                }
            }
        },
    )
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="assign-failclosed-v1",
        samples=[
            VoicePoolSample(
                voice_id="unknown_target_farther",
                sample_id="unknown_target_farther",
                audio_uri="s3://aegisvox/pool/unknown.wav",
                language="en",
                metadata={"audio_traits": {"language": "english", "accent_group": "american_english"}},
            ),
            VoicePoolSample(
                voice_id="known_target_match",
                sample_id="known_target_match",
                audio_uri="s3://aegisvox/pool/known.wav",
                language="en",
                metadata={
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "female",
                        "accent_group": "american_english",
                    }
                },
            ),
        ],
    )
    manifest_path = tmp_path / "assign_failclosed_manifest.json"
    index_path = tmp_path / "assign_failclosed_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"unknown_target_farther": [-1.0, 0.0], "known_target_match": [-0.8, 0.6]})
    index = VoicePoolIndex.from_paths(manifest_path, index_path)

    payload = AssignmentService(store, index, AssignmentPolicy.for_privacy_mode("medium")).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_failclosed/speaker_target_map.json",
    )
    row = payload["assignments"]["SPEAKER_00"]
    assert row["target_voice_sample_id"] == "known_target_match"
    assert row["gender_match_status"] == "match"
    assert row["accent_match_status"] == "match"


def test_assignment_v2_rejects_non_native_english_for_north_american_source(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_v2/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_v2/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "source_embedding_uri": emb,
                    "language": "english",
                    "gender_presentation": "male",
                    "geography_region": "north_america",
                }
            }
        },
    )
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="v2-fixture",
        samples=[
            VoicePoolSample(
                voice_id="german_male_english_open_swara_001",
                sample_id="german_male_english_open_swara_001",
                audio_uri="s3://aegisvox/pool/german.wav",
                language="en",
                gender_presentation="male",
                metadata={
                    "repo_file": "voices/german/male/german_male_english_open_swara_001.wav",
                    "duration_seconds": 8.0,
                    "sample_rate": 48000,
                    "audio_traits": {"language": "english", "gender_presentation": "male", "geography_region": "north_america"},
                },
            ),
            VoicePoolSample(
                voice_id="english_male_open_swara_001",
                sample_id="english_male_open_swara_001",
                audio_uri="s3://aegisvox/pool/english.wav",
                language="en",
                gender_presentation="male",
                metadata={
                    "repo_file": "voices/english/male/english_male_open_swara_001.wav",
                    "duration_seconds": 8.0,
                    "sample_rate": 48000,
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "male",
                        "geography_region": "north_america",
                        "accent_group": "american_english",
                    },
                },
            ),
        ],
    )
    manifest_path = tmp_path / "v2_manifest.json"
    index_path = tmp_path / "v2_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"german_male_english_open_swara_001": [-1.0, 0.0, 0.0], "english_male_open_swara_001": [-0.8, 0.6, 0.0]})
    payload = AssignmentService(store, VoicePoolIndex.from_paths(manifest_path, index_path), AssignmentPolicy.for_privacy_mode("medium")).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_v2/speaker_target_map.json",
    )
    row = payload["assignments"]["SPEAKER_00"]
    assert row["target_voice_id"] == "english_male_open_swara_001"
    assert row["assignment_strategy"] == "accent_style_constrained_v2"
    assert row["target_catalog_traits"]["catalog_type"] == "native"
    assert payload["rejected_candidates"]["SPEAKER_00"][0]["reason"] in {
        "catalog_language_not_native_english:german",
        "catalog_type_not_native:english_accent",
        "catalog_is_english_accent",
    }


def test_assignment_v2_rejects_thick_target_for_thin_source_when_thin_exists(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_style/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_style/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "source_embedding_uri": emb,
                    "language": "english",
                    "gender_presentation": "male",
                    "geography_region": "north_america",
                    "metadata": {"audio_traits": {"vocal_weight": "thin", "resonance": "light"}},
                }
            }
        },
    )
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="style-fixture",
        samples=[
            VoicePoolSample(
                voice_id="english_male_open_swara_900",
                sample_id="english_male_open_swara_900",
                audio_uri="s3://aegisvox/pool/thick.wav",
                language="en",
                gender_presentation="male",
                metadata={
                    "repo_file": "voices/english/male/english_male_open_swara_900.wav",
                    "duration_seconds": 7.0,
                    "sample_rate": 48000,
                    "audio_traits": {"language": "english", "gender_presentation": "male", "geography_region": "north_america", "accent_group": "american_english", "vocal_weight": "thick", "resonance": "deep"},
                },
            ),
            VoicePoolSample(
                voice_id="english_male_open_swara_901",
                sample_id="english_male_open_swara_901",
                audio_uri="s3://aegisvox/pool/thin.wav",
                language="en",
                gender_presentation="male",
                metadata={
                    "repo_file": "voices/english/male/english_male_open_swara_901.wav",
                    "duration_seconds": 7.0,
                    "sample_rate": 48000,
                    "audio_traits": {"language": "english", "gender_presentation": "male", "geography_region": "north_america", "accent_group": "american_english", "vocal_weight": "thin", "resonance": "light"},
                },
            ),
        ],
    )
    manifest_path = tmp_path / "style_manifest.json"
    index_path = tmp_path / "style_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"english_male_open_swara_900": [-1.0, 0.0, 0.0], "english_male_open_swara_901": [-0.8, 0.6, 0.0]})
    policy = AssignmentPolicy(
        **{**AssignmentPolicy.for_privacy_mode("medium").__dict__, "trait_weight": 0.0}
    )
    payload = AssignmentService(store, VoicePoolIndex.from_paths(manifest_path, index_path), policy).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_style/speaker_target_map.json",
    )
    assert payload["assignments"]["SPEAKER_00"]["target_voice_id"] == "english_male_open_swara_900"
    reasons = [row["reason"] for row in payload["rejected_candidates"]["SPEAKER_00"]]
    assert not any("mismatched_style_trait:vocal_weight" in reason for reason in reasons)
    assert payload["policy"]["hard_filter_source_audio_traits"] is False


def test_assignment_v2_rejects_short_or_low_sample_rate_references(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_quality/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_quality/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {"speakers": {"SPEAKER_00": {"source_embedding_uri": emb, "language": "english", "gender_presentation": "female", "geography_region": "north_america"}}},
    )
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="quality-fixture",
        samples=[
            VoicePoolSample(
                voice_id="english_female_open_swara_100",
                sample_id="english_female_open_swara_100",
                audio_uri="s3://aegisvox/pool/short.wav",
                language="en",
                gender_presentation="female",
                metadata={"repo_file": "voices/english/female/english_female_open_swara_100.wav", "duration_seconds": 2.0, "sample_rate": 48000, "audio_traits": {"language": "english", "gender_presentation": "female", "geography_region": "north_america", "accent_group": "american_english"}},
            ),
            VoicePoolSample(
                voice_id="english_female_open_swara_101",
                sample_id="english_female_open_swara_101",
                audio_uri="s3://aegisvox/pool/low_sr.wav",
                language="en",
                gender_presentation="female",
                metadata={"repo_file": "voices/english/female/english_female_open_swara_101.wav", "duration_seconds": 8.0, "sample_rate": 8000, "audio_traits": {"language": "english", "gender_presentation": "female", "geography_region": "north_america", "accent_group": "american_english"}},
            ),
            VoicePoolSample(
                voice_id="english_female_open_swara_102",
                sample_id="english_female_open_swara_102",
                audio_uri="s3://aegisvox/pool/good.wav",
                language="en",
                gender_presentation="female",
                metadata={"repo_file": "voices/english/female/english_female_open_swara_102.wav", "duration_seconds": 8.0, "sample_rate": 48000, "audio_traits": {"language": "english", "gender_presentation": "female", "geography_region": "north_america", "accent_group": "american_english"}},
            ),
        ],
    )
    manifest_path = tmp_path / "quality_manifest.json"
    index_path = tmp_path / "quality_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"english_female_open_swara_100": [-1.0, 0.0, 0.0], "english_female_open_swara_101": [-0.95, 0.05, 0.0], "english_female_open_swara_102": [-0.8, 0.6, 0.0]})
    payload = AssignmentService(store, VoicePoolIndex.from_paths(manifest_path, index_path), AssignmentPolicy.for_privacy_mode("medium")).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_quality/speaker_target_map.json",
    )
    assert payload["assignments"]["SPEAKER_00"]["target_voice_id"] == "english_female_open_swara_102"
    reasons = [row["reason"] for row in payload["rejected_candidates"]["SPEAKER_00"]]
    assert any(reason.startswith("reference_too_short") for reason in reasons)
    assert any(reason.startswith("reference_sample_rate_too_low") for reason in reasons)


def test_assignment_v2_rejects_child_like_target_for_adult_source(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_maturity/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_maturity/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "source_embedding_uri": emb,
                    "language": "english",
                    "gender_presentation": "female",
                    "geography_region": "north_america",
                    "metadata": {
                        "audio_traits": {
                            "language": "english",
                            "gender_presentation": "female",
                            "geography_region": "north_america",
                            "voice_maturity": "adult",
                        },
                        "voice_quality_profile": {"profile": {"voice_maturity": "adult", "child_like_risk": 0.01}},
                    },
                }
            }
        },
    )
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="maturity-fixture",
        samples=[
            VoicePoolSample(
                voice_id="english_female_open_swara_childlike",
                sample_id="english_female_open_swara_childlike",
                audio_uri="s3://aegisvox/pool/childlike.wav",
                language="en",
                gender_presentation="female",
                metadata={
                    "repo_file": "voices/english/female/english_female_open_swara_childlike.wav",
                    "duration_seconds": 8.0,
                    "sample_rate": 48000,
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "female",
                        "geography_region": "north_america",
                        "accent_group": "american_english",
                    },
                    "voice_quality_profile": {"profile": {"voice_maturity": "youthful", "child_like_risk": 0.72}},
                },
            ),
            VoicePoolSample(
                voice_id="english_female_open_swara_adult",
                sample_id="english_female_open_swara_adult",
                audio_uri="s3://aegisvox/pool/adult.wav",
                language="en",
                gender_presentation="female",
                metadata={
                    "repo_file": "voices/english/female/english_female_open_swara_adult.wav",
                    "duration_seconds": 8.0,
                    "sample_rate": 48000,
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "female",
                        "geography_region": "north_america",
                        "accent_group": "american_english",
                    },
                    "voice_quality_profile": {"profile": {"voice_maturity": "adult", "child_like_risk": 0.02}},
                },
            ),
        ],
    )
    manifest_path = tmp_path / "maturity_manifest.json"
    index_path = tmp_path / "maturity_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"english_female_open_swara_childlike": [-1.0, 0.0, 0.0], "english_female_open_swara_adult": [-0.8, 0.6, 0.0]})
    base_policy = AssignmentPolicy.for_privacy_mode("medium")
    policy = AssignmentPolicy(**{**base_policy.__dict__, "trait_weight": 0.0})
    payload = AssignmentService(store, VoicePoolIndex.from_paths(manifest_path, index_path), policy).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_maturity/speaker_target_map.json",
    )
    assert payload["assignments"]["SPEAKER_00"]["target_voice_id"] == "english_female_open_swara_adult"
    reasons = [row["reason"] for row in payload["rejected_candidates"]["SPEAKER_00"]]
    assert any("target_child_like" in reason for reason in reasons)


def test_assignment_prefers_enriched_gender_over_f0_top_level(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_audio_gender/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_audio_gender/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "duration_sec": 1.0,
                    "source_audio_uri": "s3://aegisvox/runs/run_audio_gender/speaker_audio/SPEAKER_00.wav",
                    "source_embedding_uri": emb,
                    "language": "en",
                    "gender_presentation": "female",
                    "metadata": {
                        "gender_presentation_source": "f0_autocorrelation_heuristic",
                        "audio_traits": {
                            "language": "english",
                            "gender_presentation": "male",
                            "geography_region": "north_america",
                            "vocal_weight": "thick",
                            "resonance": "deep",
                            "voice_maturity": "mature",
                        },
                    },
                }
            }
        },
    )
    payload = AssignmentService(store, make_gender_pool(tmp_path), AssignmentPolicy.for_privacy_mode("medium")).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_audio_gender/speaker_target_map.json",
    )
    row = payload["assignments"]["SPEAKER_00"]
    assert row["source_gender_presentation"] == "male"
    assert row["target_gender_presentation"] == "male"
    assert row["target_voice_id"] == "english_male_open_swara_001"
    assert row["source_audio_traits"]["voice_maturity"] == "mature"


def test_assignment_preserves_known_french_language_and_does_not_force_english_targets(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    emb = "s3://aegisvox/runs/run_fr/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_fr/speaker_profiles.json"
    store.write_json(emb, {"embedding": [1.0, 0.0, 0.0]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "source_embedding_uri": emb,
                    "language": "french",
                    "gender_presentation": "female",
                }
            }
        },
    )
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="fr-fixture",
        samples=[
            VoicePoolSample(
                voice_id="english_female_open_swara_001",
                sample_id="english_female_open_swara_001",
                audio_uri="s3://aegisvox/pool/en.wav",
                language="en",
                gender_presentation="female",
                metadata={"repo_file": "voices/english/female/english_female_open_swara_001.wav", "duration_seconds": 8.0, "sample_rate": 48000},
            ),
            VoicePoolSample(
                voice_id="french_female_open_swara_001",
                sample_id="french_female_open_swara_001",
                audio_uri="s3://aegisvox/pool/fr.wav",
                language="fr",
                gender_presentation="female",
                metadata={"repo_file": "voices/french/female/french_female_open_swara_001.wav", "duration_seconds": 8.0, "sample_rate": 48000},
            ),
        ],
    )
    manifest_path = tmp_path / "fr_manifest.json"
    index_path = tmp_path / "fr_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(
        index_path,
        {
            # English is farther, but must be rejected by language guardrail.
            "english_female_open_swara_001": [-1.0, 0.0, 0.0],
            "french_female_open_swara_001": [-0.8, 0.6, 0.0],
        },
    )

    payload = AssignmentService(store, VoicePoolIndex.from_paths(manifest_path, index_path), AssignmentPolicy.for_privacy_mode("medium")).assign_to_artifact(
        profiles_uri,
        "s3://aegisvox/runs/run_fr/speaker_target_map.json",
    )

    row = payload["assignments"]["SPEAKER_00"]
    assert row["target_voice_id"] == "french_female_open_swara_001"
    assert row["source_language"] == "french"
    assert row["target_language"] == "french"
    assert row["language_match_status"] == "match"
    assert row["target_audio_traits"]["language"] == "french"
