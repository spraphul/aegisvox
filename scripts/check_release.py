"""Public-release hygiene and local documentation-link checks, no network calls."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import json
ROOT=Path(__file__).resolve().parents[1]
IGNORE={'.git','.venv','__pycache__','.pytest_cache','build','dist','runs','artifacts'}
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in IGNORE or x.endswith('.egg-info') for x in p.relative_to(ROOT).parts)]
issues=[]
for p in files:
    rel=p.relative_to(ROOT)
    if p.is_symlink():issues.append(f'{rel}: symlinks are not allowed')
    if p.name in {'.env','.aifactory.env'} or p.suffix in {'.wav','.mp3','.ckpt','.safetensors','.sqlite3'}:
        issues.append(f'{rel}: restricted runtime artifact')
    if p.stat().st_size>40*1024**2:issues.append(f'{rel}: unexpectedly large file')
    if p.suffix not in {'.py','.md','.html','.json','.yaml','.yml','.toml','.txt','.cff','.css'}:continue
    s=p.read_text()
    if p!=Path(__file__).resolve():
        for pattern in [r'\bhf_[A-Za-z0-9]{20,}',r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{25,}',r'\bolp_[A-Za-z0-9]{15,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',r'/Users/[^/\s]+/',r'/mount/sds/',r'\bprapsing\b']:
            if re.search(pattern,s):issues.append(f'{rel}: credential/private-path pattern')
    if p.suffix in {'.md','.html'}:
        links=re.findall(r'(?:href|src|poster)="([^"]+)"',s)
        if p.suffix=='.md':links+=re.findall(r'\]\(([^)\s]+)\)',s)
        for raw in links:
            url=urlsplit(raw)
            if url.scheme or url.netloc or not url.path or '{' in raw:continue
            target=(p.parent/unquote(url.path)).resolve()
            if not target.exists():issues.append(f'{rel}: broken local link {raw}')
report={'checked_files':len(files),'issues':issues}
print(json.dumps(report,indent=2))
raise SystemExit(bool(issues))
