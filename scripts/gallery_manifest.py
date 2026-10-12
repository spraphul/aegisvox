"""Validate the deliberately narrow public audio release allowlist."""
from pathlib import Path
import hashlib
import json
import math


def approved_audio(root: Path):
    manifest = root / "docs/assets/audio/examples.json"
    issues = []
    allowed = set()
    try:
        data = json.loads(manifest.read_text())
        rows = data["examples"]
        assert len(rows) == 10
        assert {r["id"] for r in rows} == {f"example-{n:02d}" for n in range(1, 11)}
        assert data["publication_scope"] == "ten-curated-full-conversation-pairs"
        assert data["output_publication_clearance"] == "author-confirmed-2026-10-12"
        for row in rows:
            assert set(row["media"]) == {"original", "final"}
            for role, media in row["media"].items():
                filename = f'{row["id"]}-{role}.mp3'
                assert media["file"] == filename
                rel = Path("docs/assets/audio") / filename
                path = root / rel
                assert path.is_file() and not path.is_symlink(), str(rel)
                assert 0 < media["bytes"] < 40 * 1024**2
                assert path.stat().st_size == media["bytes"], str(rel)
                assert hashlib.sha256(path.read_bytes()).hexdigest() == media["sha256"], str(rel)
                assert math.isfinite(media["duration_sec"]) and media["duration_sec"] > 60
                assert abs(media["encoded_duration_sec"] - media["duration_sec"]) < .2
                allowed.add(rel)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        issues.append(f"Audio publication manifest failed: {exc}")
        allowed.clear()
    return allowed, issues
