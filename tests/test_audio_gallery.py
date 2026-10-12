"""Guard the public demo scope without accepting arbitrary runtime audio."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gallery_manifest', ROOT / 'scripts/gallery_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fake_release(tmp_path):
    data = json.loads((ROOT / 'docs/assets/audio/examples.json').read_text())
    folder = tmp_path / 'docs/assets/audio'
    folder.mkdir(parents=True)
    for row in data['examples']:
        for media in row['media'].values():
            payload = media['file'].encode()
            (folder / media['file']).write_bytes(payload)
            media['bytes'] = len(payload)
            media['sha256'] = hashlib.sha256(payload).hexdigest()
    (folder / 'examples.json').write_text(json.dumps(data))
    return data, folder


def test_exact_public_files_pass():
    allowed, errors = module.approved_audio(ROOT)
    assert not errors
    assert len(allowed) == 20
    assert {ROOT / p for p in allowed} == set((ROOT / 'docs/assets/audio').glob('*.mp3'))


@pytest.mark.parametrize('fault', ['tampered', 'missing', 'traversal', 'extra-row', 'no-clearance', 'zero-duration'])
def test_manifest_fails_closed(tmp_path, fault):
    data, folder = fake_release(tmp_path)
    first = data['examples'][0]['media']['original']
    if fault == 'tampered':
        (folder / first['file']).write_bytes(b'changed')
    elif fault == 'missing':
        (folder / first['file']).unlink()
    elif fault == 'traversal':
        first['file'] = '../unexpected.mp3'
    elif fault == 'extra-row':
        data['examples'].append(copy.deepcopy(data['examples'][0]))
    elif fault == 'no-clearance':
        data['output_publication_clearance'] = ''
    elif fault == 'zero-duration':
        first['duration_sec'] = 0
    (folder / 'examples.json').write_text(json.dumps(data))
    allowed, errors = module.approved_audio(tmp_path)
    assert not allowed
    assert errors


def test_static_gallery_matches_manifest():
    text = (ROOT / 'docs/index.html').read_text()
    assert text.count('<audio controls') == 20
    data = json.loads((ROOT / 'docs/assets/audio/examples.json').read_text())
    for row in data['examples']:
        assert f'id="{row["id"]}"' in text
        for media in row['media'].values():
            assert f'src="assets/audio/{media["file"]}"' in text
    assert 'Curated demonstrations, not a performance estimate.' in text
    assert 'preload="metadata"' in text
