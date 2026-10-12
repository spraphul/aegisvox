"""Render static, no-JavaScript-required players from the frozen public manifest."""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def duration(seconds):
    value = int(seconds)
    return f'{value // 60}:{value % 60:02d}'


def render(data):
    cards = []
    for row in data['examples']:
        dataset = data['datasets'][row['dataset']]
        language = {'en': 'English', 'fr': 'French'}[row['language']]
        title = f'{dataset["name"]} · {row["source_id"]}'
        players = []
        for role, label in [('original', 'Original'), ('final', 'Final transformed')]:
            media = row['media'][role]
            url = 'assets/audio/' + media['file']
            audio_label = html.escape(f'{title}: {label}, full conversation')
            players.append(f'''<div class="audio-side {role}">
<div class="player-heading"><span>{label}</span><span class="duration">{duration(media['duration_sec'])} <span>· full</span></span></div>
<audio controls preload="metadata" src="{url}" aria-label="{audio_label}">Your browser cannot play this audio. <a href="{url}">Open MP3</a>.</audio>
<div class="audio-links"><a href="{url}" target="_blank" rel="noopener" aria-label="Open MP3: {audio_label}">Open MP3 ↗</a><a href="{url}" download aria-label="Download: {audio_label}">Download · {media['bytes']/1_000_000:.1f} MB</a><button class="retry-audio" hidden type="button">Retry</button></div>
<p class="audio-status" role="status" hidden></p></div>''')
        cards.append(f'''<article class="audio-example" id="{row['id']}" data-corpus="{row['dataset']}" data-language="{row['language']}">
<div class="example-heading"><span class="example-number">{row['id'][-2:]}</span><div><h3>{html.escape(dataset['name'])}</h3><p>{html.escape(dataset['setting'])}</p></div><span class="language-pill">{language}</span></div>
<div class="audio-pair">{''.join(players)}</div>
<details class="example-details"><summary>Recording details <span>{row['speaker_count']} detected speakers · {row['edited_turns']} edited turn{'s' if row['edited_turns'] != 1 else ''}</span></summary>
<p>Source recording: <code>{html.escape(row['source_id'])}</code>. Speaker counts are model-detected, not demographic or reference labels. Final audio uses consistent SeedVC voice conversion, F5-TTS complete-turn lexical editing, and conversation reconstruction.</p>
<p>{html.escape(dataset['source_note'])} <a href="{dataset['url']}">Dataset source ↗</a> · <a href="guides/audio-examples.html">Attribution and selection</a></p></details></article>''')
    return '''<!-- AUDIO-GALLERY:START -->
<section class="block" id="listen" aria-labelledby="listen-heading">
<div class="section-head"><div><span class="eyebrow">Listen, side by side</span><h2 id="listen-heading">Hear the whole conversation.</h2></div><p>The original input and the final reconstructed output. Ten complete conversations, from short French dispatch calls to longer English consultations.</p></div>
<div class="listening-guide"><span class="listening-icon" aria-hidden="true">◉</span><p>Use headphones. Compare the voices, pacing, and delivery. These are public simulated or non-patient conversations, not real-patient recordings. The final output includes both voice conversion and lexical editing.</p></div>
<div class="gallery-toolbar"><div class="gallery-filters"><label>Dataset<select id="dataset-filter"><option value="all">All three datasets</option><option value="primock57">PriMock57</option><option value="fareez">Fareez</option><option value="simsamu">SIMSAMU</option></select></label><label>Language<select id="language-filter"><option value="all">English + French</option><option value="en">English</option><option value="fr">French</option></select></label><button type="button" id="reset-filters">Reset</button></div><p id="example-count" aria-live="polite">10 of 10 conversations</p></div>
<p class="gallery-caption">Full-length playback · 2–15 minutes per recording · Only one player plays at a time</p>
<noscript><p>Filters require JavaScript. All ten pairs are available below; pause one player before starting another.</p></noscript>
<p id="no-examples" hidden>No examples match these filters. Choose another dataset or reset.</p>
<div class="audio-examples">''' + '\n'.join(cards) + '''</div>
<div class="gallery-footnote"><p><strong>Curated demonstrations, not a performance estimate.</strong> Selection spans datasets, languages, lengths, model-detected speaker counts, and edit counts. These examples are not an independent listening study or a guarantee of complete identifier removal.</p><p>MP3 files preserve the full source and final timelines. No passages were trimmed for this gallery. Measurements use the underlying WAVs, not these compressed playback files. <a href="guides/audio-examples.html">Selection, provenance, and permissions →</a></p></div>
</section>
<!-- AUDIO-GALLERY:END -->'''


if __name__ == '__main__':
    data = json.loads((DOCS / 'assets/audio/examples.json').read_text())
    page = DOCS / 'index.html'
    content = page.read_text()
    start = content.index('<!-- AUDIO-GALLERY:START -->')
    end = content.index('<!-- AUDIO-GALLERY:END -->') + len('<!-- AUDIO-GALLERY:END -->')
    page.write_text(content[:start] + render(data) + content[end:])
    print(f'Rendered {len(data["examples"])} full-length audio pairs.')
