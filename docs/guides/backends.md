# Backends and resident workers

## Component roles

| Component | Operation | Public integration |
| --- | --- | --- |
| pyannote / alternate diarization backend | Speaker turn boundaries and labels | `diarization.py`: local, command, HTTP interfaces. |
| faster-whisper / Canary | Audio to timestamped text | `asr_alignment.py`: explicit backend selection. |
| SpeechBrain ECAPA | Audio to speaker embedding | `speaker_profile.py`: local or HTTP service. |
| Voice-pool retrieval | Embeddings and metadata to assigned references | `voice_pool.py`, `voice_pool_faiss.py`, `assignment.py`. |
| SeedVC | Source turn + assigned reference to converted speech | `SeedVCCommandBackend`, `SeedVCHTTPBackend`. |
| Text LLM | Transcript to identifier/replacement plan | `OpenAILLMTextDeIDBackend`; configure the provider/model explicitly. |
| VoiceCraft | Audio/text-conditioned lexical edit | `VoiceCraftCommandBackend`, `VoiceCraftHTTPBackend`. |
| F5-TTS | Converted voice reference + revised words to a complete turn | `examples/f5_turn_worker.py`, complete-turn renderer API. |

These are adapter boundaries, not bundled model servers. Obtain model packages and
weights from their upstream maintainers. The service queue's bundled lexical job uses
the VoiceCraft adapter; it does not automatically route to the separate F5 example.
Register your chosen complete-turn worker explicitly when integrating the two APIs.

## Real F5-TTS example

Follow the [official F5-TTS setup](https://github.com/SWivid/F5-TTS) for device-matched
Torch and FFmpeg. Install AegisVox into that model environment too. The example uses
`F5TTS_v1_Base` and 16 inference steps, from the research runner. This is a model label,
not a pinned checkpoint digest: record the exact installed revision for a new study.
Review the code **and checkpoint** license; they are not necessarily the same.

```python
from pathlib import Path
from examples.f5_turn_worker import F5TurnRenderer
from aegisvox.complete_turns import render_conversation

renderer = F5TurnRenderer(device='cuda', nfe_step=16)  # load once
for manifest, destination in jobs:
    render_conversation(Path(manifest), Path(destination), renderer)
```

Run from the repository root. The object stays resident for all turns and conversations
handled by that worker. Use one such process per allocated GPU and a bounded queue.
The small command example loads per process; do not mistake it for the resident mode.
Upstream preprocessing can crop or normalize a supplied reference internally; AegisVox
supplies the entire turn but does not claim every upstream model attends to every sample.

The example adapter explicitly resamples native output if needed, writes mono PCM16,
and returns the native/output rates. The reconstruction path then inserts that file
exactly. Upstream F5 inference is not executed by CPU unit tests.

## SeedVC and VoiceCraft

Use the interfaces in `voice_conversion.py` and `speech_editing.py` with the
[official SeedVC](https://github.com/Plachtaa/seed-vc) and
[official VoiceCraft](https://github.com/jasonppy/VoiceCraft) repositories.
Commands are operator-supplied adapters, not guaranteed upstream CLI spellings.
VoiceCraft also requires correctly aligned edit positions; providing a complete turn
does not remove that requirement. Do not silently use a tiny lexical crop as its only
reference. Verify target-speaker consistency after synthesis.

## Text planning

Supply provider credentials through runtime environment variables, not YAML. Configure
`AEGISVOX_TEXT_DEID_MODEL` explicitly; the paper used GPT-5.4 for the relevant text
judgments/planning. `OPENAI_BASE_URL` supports an authorized compatible provider. The
repository does not give access to any paid or internal API. Review provider retention
and data-transfer policies before sending any clinical text.

## Model isolation

Keep ASR, speaker embedding, SeedVC, F5, and VoiceCraft environments separate. Pin source
revisions, checkpoint hashes, CUDA, Torch, codec, prompt, and inference settings. Caches,
weights, audio, logs, and artifacts should live on an operator-selected data volume.
No local workstation or private GPU-server path is assumed by these instructions.
