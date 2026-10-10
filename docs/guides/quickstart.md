# Start with mechanics, then connect models

## Install

Use Python 3.10 or newer. The lightweight package does not install CUDA models.
Run commands from the repository root unless a guide says otherwise.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,evaluation]'
aegisvox --help
```

The base dependency is PyYAML, plus the Python audioop compatibility package on
Python 3.13+. The evaluation extra adds NumPy, pandas, SciPy, and statsmodels.
The API extra adds FastAPI, Uvicorn, and HTTPX. Install each model's dependencies
in a separate environment to avoid incompatible Torch, audio, and codec packages.

## Run the offline example

```bash
aegisvox demo --output runs/demo-001
cat runs/demo-001/result/reconstruction.json
pytest -q
```

Use a new directory for each run. The command refuses to overwrite existing output.
Listen only to verify a signal exists: these are tones, not generated speech.
`sample_equivalent: true` means the inserted PCM samples equal the rendered file's
samples. It says nothing about the words, identity, or clinical meaning.

## Supply actual speech

Prepare a PCM WAV conversation after voice conversion. Create a manifest matching
`examples/complete_turns.json`. Times refer to that converted waveform, not to a
pre-conversion waveform whose length may differ. Each edited entry contains the
**whole turn's** source and revised transcript. The complete converted turn is
cropped as the renderer reference. No user voice sample is bundled.

```bash
aegisvox render-turns /secure/work/turns.json \
  --renderer examples/f5_renderer.json --output /secure/work/render-001
```

Configure the renderer's executable to its dedicated model environment. Real-model
execution is not covered by the offline test; see [backend provisioning](backends.md).
Rendering and evaluation are separate operations. Do not mistake `assembled` for a
privacy, clinical, or production-readiness verdict.
