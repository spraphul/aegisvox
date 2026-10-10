# Complete-turn rendering contract

## Manifest

See `examples/complete_turns.json`. Required top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Exactly `aegisvox.complete_turns.v1`. |
| `converted_audio` | PCM WAV path, relative to the manifest or absolute. Identity conversion is already complete. |
| `turns` | Ordered, non-overlapping turns with unique `turn_id`, `speaker_id`, `start_sec`, and `end_sec`. |

For an affected turn, add both `source_transcript` and `target_transcript`. They must
contain the entire turn's words. Omit `target_transcript` for turns that do not need
lexical rendering. Times are in the **converted** audio. Provide correctly assigned
exclusive turns; this interface does not separate concurrent speakers or infer turns.

Names in the example are fictional schema fixtures, not patient records. Raw manifests,
transcripts, and reference clips can contain sensitive data. Store them privately.

## Renderer interface

```python
from pathlib import Path
from aegisvox.complete_turns import render_conversation

# renderer.name: str
# renderer.render(request: dict, reference: Path, output: Path) -> dict
report = render_conversation(Path('turns.json'), Path('runs/run-001'), renderer)
```

`reference` contains all samples of that converted turn. `request` includes its full
source/target transcripts, speaker/turn IDs, and PCM format. The renderer must write
`output` with matching rate, channels, and sample width. Its return value is non-secret
metadata such as model revision, settings, and seed. It must not contain credentials.

A `CommandTurnRenderer` accepts an argv list with `{request}` and `{output}` placeholders.
The request JSON file is the text boundary. The command is not passed through a shell;
transcript words are never interpolated into command arguments. Commands and model
artifacts still need to be trusted. External network access is not sandboxed by this library.

## Reconstruction guarantees and boundaries

- Keeps converted samples before, between, and after edits, including existing gaps.
- Inserts exact generated PCM. It does not stretch, crossfade, normalize, or revalidate
  lexical content after generation. This is intentional for a traceable assembly path.
- Records source and output sample intervals, hashes, and exact insertion checks.
- Allows duration changes. Later turns move when an edited turn grows or shrinks.
- Does not guarantee equal total duration, smooth joins, acoustic preservation, or privacy.
- Rejects invalid intervals, overlapping turns, format mismatch, empty/truncated WAVs,
  and all-zero output. No semantic fallback or original-turn substitution is used.
- Does not overwrite a previous run. Failed runs retain a failure report, not a final WAV.

The implementation holds the conversation in memory. Use a worker with sufficient RAM;
this release is not a streaming audio engine. Boundary smoothing is available in the
lower-level stitching research components, but would break exact PCM insertion if
applied inside the edited region. Record that operation as a different experiment.

## Output

`final.wav`, `reconstruction.json`, and one directory per edited turn containing
`reference.wav` and `edited.wav`. Command adapters also store `request.json` locally.
`status: assembled` certifies assembly only. Run lexical, ASV, and clinical evaluation
on the final waveform before making any scientific claim about it.
