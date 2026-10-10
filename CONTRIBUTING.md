# Contributing

Start with a small, documented change and a synthetic test. Do not include recordings,
transcripts, voice references, secrets, or private infrastructure in issues or commits.

```bash
pip install -e '.[dev,evaluation,api,docs]'
pytest -q
python scripts/build_docs.py
python scripts/check_release.py
python -m build
```

Preserve public schemas or version them. Keep GPU model imports lazy. State what each
backend consumes and produces. Keep credentials in the runtime environment. Add tests
for unavailable models, invalid outputs, missing evidence, and reconstruction lineage.
Never add a silent source-copy fallback to a successful transformation path.

Changes to evaluation must define the population, units, waveform stage, uncertainty,
and missingness. Do not relabel diagnostic proxies as paper metrics. Contributions are
under the repository's Apache-2.0 license; only submit work you have permission to share.
