"""Build self-contained guide pages from Markdown. No remote fonts or scripts."""
from pathlib import Path
import html
import re
import markdown
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
files=sorted((DOCS/'guides').glob('*.md'))
nav=[(p.stem,p.read_text().splitlines()[0].lstrip('# ')) for p in files]
for source in files:
    body=markdown.markdown(source.read_text(),extensions=['tables','fenced_code'])
    def link(m):
        href=m.group(1)
        if href.startswith('../../'):
            return 'href="https://github.com/spraphul/aegisvox/blob/main/'+href[6:]+'"'
        if href.endswith('.md'): return 'href="'+href[:-3]+'.html"'
        return m.group(0)
    body=re.sub(r'href="([^"]+)"',link,body)
    title=source.read_text().splitlines()[0].lstrip('# ')
    links=''.join(f'<a class="{"active" if slug==source.stem else ""}" href="{slug}.html">{html.escape(name)}</a>' for slug,name in nav)
    out=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)} | AegisVox</title><link rel="stylesheet" href="../assets/site.css"></head><body><a class="skip" href="#main">Skip to content</a><header><nav class="nav" aria-label="Primary"><a class="brand" href="../index.html">AegisVox</a><span class="wordmark">Oracle Health AI</span><div class="navlinks"><a href="../index.html">Overview</a><a href="https://github.com/spraphul/aegisvox">GitHub ↗</a></div></nav></header><div class="docs-layout"><nav class="sidebar" aria-label="Guides">{links}</nav><main id="main" class="article"><div class="title-rule"></div>{body}</main></div></body></html>"""
    source.with_suffix('.html').write_text(out+'\n')
print(f'Built {len(files)} guide pages.')
