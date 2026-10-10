from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval/compute_retrieval_assignment_metrics.py"


def load_module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("compute_retrieval_assignment_metrics", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assignment_distributions_and_constraints(tmp_path: Path):
    mod = load_module()
    run = tmp_path / "aegisvox" / "runs" / "run_1"
    run.mkdir(parents=True)
    payload = {
        "voice_pool_id": "pool",
        "assignment_strategy": "constrained",
        "policy": {"min_source_target_distance": 0.7, "min_target_target_distance": 0.65, "candidate_top_k": 75},
        "assignments": {
            "A": {"status": "assigned", "target_voice_sample_id": "v1", "source_target_distance": 1.1, "query_trait_match": {"score": 0.8}},
            "B": {"status": "assigned", "target_voice_sample_id": "v2", "source_target_distance": 1.2, "query_trait_match": {"score": 1.0}},
        },
        "target_target_distances": [{"target_target_distance": 0.9}],
        "retrieval_contract": {"source_gpt_audio_traits": "soft_scoring_only"},
    }
    (run / "speaker_target_map.json").write_text(json.dumps(payload))
    row = mod.run_metrics({"run_id": "run_1", "audio_id": "a", "completed": True}, tmp_path)
    assert row["all_assignment_constraints_pass"] is True
    assert row["num_duplicate_target_assignments"] == 0
    agg = mod.aggregate([row])
    assert agg["num_rows_all_assignment_constraints_pass"] == 1
    assert agg["min_source_target_distance"] == 1.1
    assert agg["min_target_target_distance"] == 0.9


def test_duplicate_or_threshold_failure_is_not_a_pass(tmp_path: Path):
    mod = load_module()
    run = tmp_path / "aegisvox" / "runs" / "run_1"
    run.mkdir(parents=True)
    payload = {
        "policy": {"min_source_target_distance": 0.7, "min_target_target_distance": 0.65},
        "assignments": {
            "A": {"status": "assigned", "target_voice_sample_id": "v1", "source_target_distance": 0.6},
            "B": {"status": "assigned", "target_voice_sample_id": "v1", "source_target_distance": 1.2},
        },
        "target_target_distances": [{"target_target_distance": 0.5}],
    }
    (run / "speaker_target_map.json").write_text(json.dumps(payload))
    row = mod.run_metrics({"run_id": "run_1", "audio_id": "a", "completed": True}, tmp_path)
    assert row["all_assignment_constraints_pass"] is False
    assert row["num_duplicate_target_assignments"] == 1
    assert row["num_source_target_threshold_failures"] == 1
    assert row["num_target_target_threshold_failures"] == 1
