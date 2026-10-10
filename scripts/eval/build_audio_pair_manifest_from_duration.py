#!/usr/bin/env python3
"""Build a fail-closed source/target audio manifest from duration metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    args = parser.parse_args()
    payload = json.loads(args.duration_json.read_text())
    details = payload.get("rows") or payload.get("details") or []
    rows = []
    seen = set()
    for item in details:
        audio_id = str(item.get("audio_id") or "")
        source = Path(str(item.get("input_audio_path") or ""))
        target = Path(str(item.get("final_audio_path") or ""))
        if not audio_id or audio_id in seen:
            raise ValueError(f"missing or duplicate audio_id: {audio_id!r}")
        if not source.is_file() or not target.is_file() or source.stat().st_size <= 44 or target.stat().st_size <= 44:
            raise ValueError(f"missing/empty source-target audio for {audio_id}")
        rows.append({"audio_id": audio_id, "source_audio": str(source), "target_audio": str(target)})
        seen.add(audio_id)
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} unique audio pairs, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(json.dumps({"status": "complete", "num_rows": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
