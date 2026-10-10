# One conversation, two interventions

## What enters and leaves

The research pipeline accepts long-form speech. It produces a transformed conversation,
transcript/replacement artifacts, speaker assignments, and stage-specific measurements.
The framework seeks to change vocal identity and protected lexical content while retaining
useful surrounding evidence. Preservation must be measured, not assumed.

## Identity transformation

1. **Diarization and transcription** estimate who spoke when and what was said. Errors
   here propagate into speaker profiles, target assignment, lexical planning, and editing.
2. **Speaker profiling and retrieval** build embeddings and use available metadata to
   choose voice references. Source-to-reference distance and reference-to-reference
   distinctiveness are separate constraints. A consistent assignment is reused for each
   source speaker throughout a conversation. A retrieval distance is not an ASV EER.
3. **SeedVC** converts every speaker turn to its assigned voice. It does not detect
   identifiers or decide which words should change.

## Lexical transformation

4. **ASR and LLM planning** produce a transcript and document-consistent synthetic
   replacements in the transcript's language. Detection and planning are distinct from
   whether the planned words are actually spoken later.
5. **Complete-turn rendering** gives the editor the converted speaker turn, its complete
   source transcript, and its complete revised transcript. F5-TTS synthesizes a turn;
   VoiceCraft uses audio/text conditioning to edit speech. Only affected turns are
   lexically rerendered. The framework does not define F5-TTS as the entire system.
6. **Reconstruction** inserts rendered turns into converted context. The non-edited
   context must come from the voice-converted audio, not the original identifiable voice.

## Public interfaces

The package exposes two levels:

- **Service components:** profiles, retrieval, conversion, ASR, planning, editing,
  artifact storage, events, queue, and control-plane API.
- **Complete-turn assembly:** a narrow renderer-independent interface for a converted
  waveform plus revised turn transcripts. It preserves intervening samples and records
  exact output intervals and hashes. This interface does not run the upstream stages.

The service engine retains configurable historical research modes. Its default workflow
is not the exact fixed paper experiment. Use explicit backend, model, language, and
policy settings, and label baseline/ablation configurations. Do not silently substitute
legacy runtime quality proxies for the clinical study endpoints.

## Evaluation is a separate path

Score original audio, voice-converted audio, generated edits, and final reconstructed
speech separately. Keep failures and uncertainty in the registered population. The
complete-turn interface never uses an ASR score to reject an otherwise valid WAV or
silently replace it with source speech. Structural failures stop assembly visibly.
