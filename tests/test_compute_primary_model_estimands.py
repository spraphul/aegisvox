import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from compute_primary_model_estimands import cap_estimand, orientation_robust_asv_estimand, privacy_utility_estimand, relex_estimand  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_paired_cap_and_clustered_relex_estimands(tmp_path: Path) -> None:
    f5_cap = tmp_path / "f5_cap.jsonl"; stock_cap = tmp_path / "stock_cap.jsonl"
    write_jsonl(f5_cap, [
        {"audio_id":"a0","result":{"conversation_assessment":"no_major_error"}},
        {"audio_id":"a1","result":{"conversation_assessment":"uncertain"}},
    ])
    write_jsonl(stock_cap, [
        {"audio_id":"a0","result":{"conversation_assessment":"major_omission"}},
        {"audio_id":"a1","result":{"conversation_assessment":"uncertain"}},
    ])
    cap = cap_estimand(f5_cap, stock_cap, expected_rows=2, iterations=100, seed=1)
    assert cap["primary"]["f5_rate"] == 0.5
    assert cap["primary"]["stock_rate"] == 0.0
    assert cap["uncertainty_sensitivity"]["optimistic_uncertain_success"]["stock_rate"] == 0.5

    f5_relex = tmp_path / "f5_relex.jsonl"; stock_relex = tmp_path / "stock_relex.jsonl"
    ids = ["run_a::edit_1", "run_a::edit_2", "run_b::edit_1"]
    write_jsonl(f5_relex, [
        {"adjudication_id":ids[0],"replacement_status":"realized"},
        {"adjudication_id":ids[1],"replacement_status":"uncertain"},
        {"adjudication_id":ids[2],"replacement_status":"acceptable_variant"},
    ])
    write_jsonl(stock_relex, [{"adjudication_id":value,"replacement_status":"not_realized"} for value in ids])
    relex = relex_estimand(f5_relex, stock_relex, expected_edits=3, iterations=100, seed=2)
    assert relex["primary"]["n_clusters"] == 2
    assert relex["primary"]["f5_rate"] == pytest.approx(2/3)
    assert relex["uncertainty_sensitivity"]["optimistic_uncertain_success"]["f5_rate"] == 1.0


def test_estimands_fail_on_unpaired_population(tmp_path: Path) -> None:
    a=tmp_path/'a'; b=tmp_path/'b'
    write_jsonl(a,[{"audio_id":"a0","result":{"conversation_assessment":"no_major_error"}}])
    write_jsonl(b,[{"audio_id":"a1","result":{"conversation_assessment":"no_major_error"}}])
    with pytest.raises(ValueError, match="paired"):
        cap_estimand(a,b,expected_rows=1,iterations=10,seed=1)


def test_privacy_utility_vector_is_paired_without_scalarization(tmp_path: Path) -> None:
    f5_cap=tmp_path/'f5_cap'; stock_cap=tmp_path/'stock_cap'
    cap_rows_f5=[]; cap_rows_stock=[]
    for i in range(2):
        cap_rows_f5.append({'audio_id':f'a{i}','metrics':{'strict_preservation_rate':0.9+i*0.01}})
        cap_rows_stock.append({'audio_id':f'a{i}','metrics':{'strict_preservation_rate':0.8+i*0.01}})
    write_jsonl(f5_cap,cap_rows_f5); write_jsonl(stock_cap,cap_rows_stock)
    paths=[]
    for name,offset in [('f5e',0.0),('ste',0.1),('f5x',0.0),('stx',0.2)]:
        p=tmp_path/name
        p.write_text(json.dumps({'rows':[{'audio_id':f'a{i}','max_source_converted_similarity':0.4+offset+i*0.01} for i in range(2)]}))
        paths.append(p)
    report=privacy_utility_estimand(
        f5_cap_path=f5_cap,stock_cap_path=stock_cap,
        f5_ecapa_path=paths[0],stock_ecapa_path=paths[1],
        f5_xvector_path=paths[2],stock_xvector_path=paths[3],
        expected_rows=2,iterations=50,seed=3,
    )
    assert report['privacy_endpoints']['ecapa_max_source_converted_similarity']['risk_difference_f5_minus_stock'] == pytest.approx(-0.1)
    assert report['clinical_utility_endpoint']['strict_cap_preservation_rate']['risk_difference_f5_minus_stock'] == pytest.approx(0.1)
    assert 'scalar' in report['joint_interpretation_policy']


def test_orientation_robust_asv_requires_exact_four_cells(tmp_path: Path) -> None:
    systems = {}
    for system in ("aegisvox_f5", "stock_voicecraft"):
        for backend in ("ecapa", "xvector"):
            systems[f"{system}/{backend}"] = {
                "num_source_recordings": 390,
                "num_source_speakers": 809,
                "attacker_optimal_eer": 0.3,
                "attacker_optimal_eer_ci95": [0.28, 0.32],
            }
    path = tmp_path / "asv.json"
    path.write_text(json.dumps({
        "schema_version": "aegisvox.orientation_robust_asv_summary.v1",
        "status": "complete",
        "systems": systems,
        "headline_metric": "attacker_optimal_eer",
    }))
    report = orientation_robust_asv_estimand(path)
    assert report["systems"]["aegisvox_f5/ecapa"]["attacker_optimal_eer"] == 0.3
    systems.pop("stock_voicecraft/xvector")
    path.write_text(json.dumps({
        "schema_version": "aegisvox.orientation_robust_asv_summary.v1",
        "status": "complete",
        "systems": systems,
    }))
    with pytest.raises(ValueError, match="exact four"):
        orientation_robust_asv_estimand(path)
