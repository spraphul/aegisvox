# Release scope and provenance

## Included

- Existing AegisVox component library, service contracts, and focused unit tests.
- Modular evaluation implementations and their prompts in source code.
- A new documented complete-turn assembly API and CLI with CPU-only mechanics tests.
- A real-model F5-TTS adapter example adapted from the research runner, not GPU-tested
  as part of this packaging task.
- The approved 78-second silent AegisVox technical demonstration, its existing cover image,
  and a visual documentation site. No stock-photo conference outro is bundled.

## Excluded

Source/reference/clinical audio, transformed research recordings, model weights,
credentials, private host configuration, raw reviewer ratings/access codes, confidential
reviews, unpublished manuscript source, and research run directories are not bundled.
These omissions do not mean a result is zero or that a full inference run is local-only.

## Tested scope

The CPU test suite checks contracts, assignment, segmentation, conversion/editing
interfaces, complete-turn assembly, and selected evaluation functions with fixtures.
It does not run the upstream speech models or certify privacy or clinical validity.
`docs/release-check.json` records the local release checks. CI exercises Python package
installation, unit tests, the mechanics demo, documentation, and the release scan.

The generic service workflow and the direct complete-turn interface serve different
integration needs. They are not silently interchangeable. The paper's frozen evidence
is maintained separately; no claims depend on rerunning a different public default.
