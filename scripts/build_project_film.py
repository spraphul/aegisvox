"""Render the homepage film section from the public media manifest."""
from pathlib import Path
import html
import json

ROOT=Path(__file__).resolve().parents[1]


def render(data):
    file=html.escape(data['file'])
    poster=html.escape(data['poster'])
    version=data['sha256'][:12]
    buttons=''.join(f'<button type="button" data-film-time="{row["start"]:.5f}" aria-controls="project-film">{html.escape(row["label"])}</button>' for row in data['chapters'])
    return f'''<!-- PROJECT-FILM:START -->
<section class="block" id="film" aria-labelledby="film-heading">
<div class="section-head"><div><span class="eyebrow">The narrated overview · 2 min 31 sec</span><h2 id="film-heading">Privacy beyond the words.</h2></div><p>Why clinical speech needs two privacy interventions, three separate evaluation questions, and governed use.</p></div>
<video id="project-film" controls playsinline preload="metadata" poster="assets/{poster}" aria-label="AegisVox narrated clinical speech overview">
<source src="assets/{file}?v={version}" type="video/mp4">
Your browser cannot play this video. <a href="assets/{file}">Download the narrated MP4.</a>
</video>
<nav class="film-chapters" aria-label="Film chapters">{buttons}</nav>
<div class="film-actions"><button class="button primary" type="button" id="film-play" aria-controls="project-film">Play with sound</button><a class="button" href="assets/{file}" download>Download video · {data['bytes']/1_000_000:.1f} MB</a><a class="button" href="guides/film-transcript.html">Read the narration</a><span id="film-status" class="subtle" role="status">Female narration · No subtitles</span></div>
<p class="subtle">An illustrated explanation with synthetic narration, not patient footage, model recordings, or a measured before/after demonstration. No autoplay or burned subtitles. <a href="assets/README.md">Media provenance and rights.</a></p>
</section>
<!-- PROJECT-FILM:END -->'''


if __name__=='__main__':
    page=ROOT/'docs/index.html'
    text=page.read_text()
    start=text.index('<!-- PROJECT-FILM:START -->')
    end=text.index('<!-- PROJECT-FILM:END -->')+len('<!-- PROJECT-FILM:END -->')
    data=json.loads((ROOT/'docs/assets/film.json').read_text())
    page.write_text(text[:start]+render(data)+text[end:])
    print('Built the photo-free narrated film section.')
