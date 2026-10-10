# Stage-resolved evaluation

## Four waveform stages, different questions

| Stage | Question | Do not infer |
| --- | --- | --- |
| Original | What does the same ASR/judge measure before transformation? | That text self-consistency is a clinical ground truth. |
| Voice-converted | What changes after the shared speaker transformation? | That a shared-stage EER describes either final lexical renderer. |
| Edited complete turn | Was a planned substitute spoken? Was its original retained? | Detector recall, complete-conversation removal, or absence from an uncertain judgment. |
| Reconstructed conversation | What is present in the exact final waveform? | That generated-turn scores automatically transfer to this stage. |

## Endpoints

**Lexical:** retain separate `realized`, `not_realized`, and `uncertain` outcomes and
original-identifier retention. Keep tolerable ASR spelling/phonetic variants explicit
in the adjudication protocol. Measure detection separately against independent source
annotations if available. Planned-edit realization cannot establish detector recall.

**Speaker:** state enrollment and test waveform stages, same/different-speaker trial
construction, session conditions, backend, score direction, and attacker knowledge.
Retrieval cosine distance is not verification performance. A score-direction sensitivity
analysis is not a trained attacker adapted to converted speech.

**Clinical:** freeze a source inventory of clinical atomic propositions (CAPs) before
judging the transformed transcript. A CAP expresses one fact with its speaker role,
negation, timing, dose/value, and uncertainty. Report preserved, partial, missing,
contradicted, and uncertain labels, and separate unsupported insertions. Model judgments
are not verified clinical harms. Use a fresh original-audio ASR control and an independent
judge to expose measurement dependence.

**Interaction/audio:** report completeness, duration, intelligibility, perceived
naturalness, attribution, and acoustic cues separately. Signal-quality heuristics are
not listener ratings. Missing measurements are not zero.

## Released measurement code

The `scripts/eval` directory includes modular research implementations:

- `compute_matched_asr_wer_cer.py`: paired original/output transcription comparisons.
- `compute_asv_privacy.py`, `compute_asv_privacy_independent.py`: speaker verification.
- `run_item_level_relex_adjudication.py`: item-level lexical judgments.
- `run_clinical_cap_extraction.py`, `run_clinical_cap_target_adjudication.py`: source
  proposition inventory and transformed-transcript assessment.
- `run_clinical_note_generation.py`, `run_clinical_note_pair_adjudication.py`: note studies.
- `compute_primary_model_estimands.py`: paired estimates with explicit population checks.
- `compute_retrieval_assignment_metrics.py`: assignment distributions and constraint checks.

Run each with `--help` for its concrete input schema. Model-backed scripts require their
provider/model configuration; optional GPU modules are not installed by the base package.
Several older diagnostic scripts explicitly compute proxies or approximate metrics.
Their inclusion is not an endorsement as clinical truth or as the final paper protocol.

## Statistical reporting

Register populations and match by stable IDs, not row order. Preserve failures and
report missingness. Distinguish item-weighted rates from conversation-weighted rates.
When edits share a conversation, use conversation-clustered paired inference for that
estimand. Clearly identify historical edit-IID intervals rather than relabeling them.
A non-significant difference is not evidence of equivalence. Report model ordering,
judge agreement, and clinical correctness as different quantities.

## Reproducibility boundary

This public code release is not a frozen results database or a one-command reexecution
of the publication. It omits licensed audio, weights, item-level transcripts, and private
human-study exports. The camera-ready evidence package remains separate. Reproducing a
new run requires authorized data, exact models/prompts, trial definitions, seeds, and
artifact hashes. Do not compare a newly selected sample to the published full population
as though the conditions were matched.
