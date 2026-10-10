# Troubleshooting

| Symptom | Check |
| --- | --- |
| Demo refuses the output directory | Choose a new directory. Existing runs are never overwritten. |
| Empty or unplayable WAV | Check frame count, sample format, channels, and that generation completed. A filename is not evidence of audio. |
| F5 command fails | Verify its separate Python environment, GPU/Torch compatibility, checkpoint access, FFmpeg/ffprobe, and upstream dependencies. |
| Format mismatch | The complete-turn API requires matching PCM. Resample explicitly in the adapter and record it. |
| Model returns silence | Keep the failure. Inspect reference conditioning and model output; do not label an unedited source turn as a successful replacement. |
| Wrong words after editing | Give the entire turn and its full revised transcript. Check alignment, language, and ASR separately. |
| Changed speaker/accent | Check consistent assignments, reference quality, embeddings/metadata, and post-render voice drift. Retrieval compatibility is not a guarantee. |
| API has no GPU activity | The API is an orchestrator. Provision and connect workers/model services separately. |
| CLI returns only an error type | Console output avoids raw text/secrets. Inspect protected request/artifact files and reproduce the model call in a protected environment. |
| Repeated model loads | Use resident HTTP or in-process workers; the example command adapter is intentionally one-shot. |
| New metrics differ from the paper | Check waveform stage, corpus revisions, settings, prompts, failures, denominator, and micro/macro weighting. |

Do not post clinical recordings, transcripts, API keys, or reviewer identifiers in
GitHub issues. Make a minimal synthetic reproduction and report package/model versions,
input format, and the exception type. Review [security guidance](../../SECURITY.md).
