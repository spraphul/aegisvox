"""Public film must be the exact reviewed, photo-free export."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import runpy

ROOT=Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT/'docs/assets/film.json').read_text())


def test_public_film_checksum_and_scope():
    m=manifest();video=ROOT/'docs/assets'/m['file']
    assert hashlib.sha256(video.read_bytes()).hexdigest()==m['sha256']
    assert video.stat().st_size==m['bytes']<40*1024**2
    assert m['duration_seconds']==150.9 and m['audio_streams']==1
    assert m['stock_photo_included'] is False and m['subtitles_burned'] is False
    assert 'public' in m['file']


def test_film_embed_and_readme_are_current():
    m=manifest();page=(ROOT/'docs/index.html').read_text();readme=(ROOT/'README.md').read_text()
    tags=[]
    class Tags(HTMLParser):
        def handle_starttag(self,tag,attrs):tags.append((tag,dict(attrs)))
    Tags().feed(page)
    video=[a for t,a in tags if t=='video']
    assert len(video)==1 and video[0]['id']=='project-film'
    assert all(key not in video[0] for key in ['autoplay','muted'])
    assert not any(t=='track' for t,_ in tags)
    assert f'assets/{m["file"]}?v={m["sha256"][:12]}' in page
    assert '#film' in readme and 'Watch the narrated film' in readme
    assert 'Watch the silent film' not in readme
    assert '78 seconds inside' not in page


def test_film_chapters_and_render_are_deterministic():
    m=manifest()
    times=[row['start'] for row in m['chapters']]
    assert times==sorted(times) and times[0]==0 and times[-1]<m['duration_seconds']
    render=runpy.run_path(str(ROOT/'scripts/build_project_film.py'))['render']
    assert render(m) in (ROOT/'docs/index.html').read_text()
    transcript=(ROOT/'docs/guides/film-transcript.md').read_text()
    assert 'Clinical conversations reveal identity through words and voices.' in transcript
    assert 'See you at ML4H twenty twenty-six, in Sydney.' in transcript
