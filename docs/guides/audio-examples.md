# Full-conversation listening examples

[Open the ten-pair gallery](../index.html#listen).

Each pair contains the **full original input** and the **full final reconstructed
conversation**. The final recording follows SeedVC voice conversion, source-language
identifier replacement planning, and F5-TTS complete-turn lexical editing. It is not
an intermediate voice-conversion output. The unchanged technical film remains silent.

## What these examples show

The gallery includes three PriMock57 consultations, four Fareez respiratory interviews,
and three SIMSAMU dispatch calls: seven English and three French conversations.
Recordings range from roughly two to fifteen minutes. Model-detected speaker counts
range from two to four and edited-turn counts from one to six.

Use headphones. Listen to the original and final versions separately, then compare
voice identity, delivery, pacing, and conversational continuity. Start each player
independently: lexical rendering can change duration, so identical clock times are
not guaranteed to refer to identical words. The page pauses other players when a new
one starts. Every pair also has direct MP3 and download links.

These are **curated demonstrations, not a random evaluation sample, a perceptual
quality ranking, or a privacy guarantee**. All source recordings are public simulated
or non-patient clinical conversations, not recordings from real-patient encounters.
Dataset setting and language are confounded and should not be interpreted as a
controlled language comparison. No unverified gender, age, or accent labels are shown.

## Selection and technical checks

Selection starts with the existing 390-conversation evaluated F5-TTS condition.
Eligible recordings have a complete final reconstruction, at least one rendered
lexical edit, no failed edited turns, at least two model-detected speakers, a
final/pre-edit-converted duration ratio within 0.95 to 1.08, and previously verified
sample lineage.
There are 204 eligible recordings: 55 PriMock57, 101 Fareez, and 48 SIMSAMU.

We selected examples near the 10th, 50th, and 90th duration percentiles within each
eligible dataset, with four Fareez positions near the 10th, 35th, 65th, and 90th
percentiles. The middle SIMSAMU position was replaced by a four-detected-speaker
example to broaden conversational structure. The exact IDs are in the manifest.
No clinical-preservation or replacement-realization score was optimized, and no
new human listening score is claimed. This eligibility filter excludes failed
reconstructions, so the gallery cannot estimate failure rates.

For every final WAV, we checked its hash against the frozen reconstruction record.
All 20 exported files were decoded and checked for positive duration and non-silence.
Playback uses 128 kb/s MP3 compression. Gallery export does not trim passages, remove
silences, normalize gain, denoise, or alter speed. The input WAV can itself be a
processed dataset input, as with the constructed PriMock57 mixtures. Reported study
measurements use WAV artifacts, not compressed browser playback.

[The public manifest](../assets/audio/examples.json) records source IDs, source
revisions, original/final durations, file sizes, and MP3 and WAV checksums. These
identify the exact examples without disclosing private storage paths or voice references.

## Source attribution and permissions

### PriMock57

Alex Papadopoulos Korfiatis, Francesco Moramarco, Radmila Sarac, and Aleksandar Savkov;
Babylon Health. *PriMock57: A Dataset Of Primary Care Mock Consultations* (2022).
[Dataset and citation](https://github.com/babylonhealth/primock57).
Frozen revision: `cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5`.
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Original gallery inputs are constructed mixtures of the corresponding doctor and
patient role tracks. Final versions additionally transform voice identity and edit
planned lexical content. Both are compressed to MP3 for playback. These are adaptations;
the original dataset creators do not endorse AegisVox. The upstream license is retained
in [PRIMOCK57-LICENSE.txt](../assets/audio/PRIMOCK57-LICENSE.txt).

### Fareez respiratory interviews

Faiha Fareez and colleagues, *Collection of simulated medical exams*.
[Figshare dataset](https://doi.org/10.6084/m9.figshare.16550013.v1).
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
Source IDs identify recordings in the released respiratory interview collection.
Final versions transform voice identity and planned lexical content; browser copies
are MP3 exports of the evaluated input and reconstructed WAVs.

### SIMSAMU

HeKA Research Team, *Simsamu dataset* (2023).
[Dataset](https://huggingface.co/datasets/medkit/simsamu).
Frozen revision: `6b639431f658d636c30dbf342074d723bf303e0f`.
MIT License, copyright (c) 2023 HeKA Research Team. The complete copyright and
permission notice is retained in [SIMSAMU-LICENSE.txt](../assets/audio/SIMSAMU-LICENSE.txt).
Final versions transform voice identity and planned lexical content; all gallery
files are MP3 playback adaptations.

### Transformed-output release scope

The project author confirmed clearance to publish these ten selected transformed
outputs on October 12, 2026. This permission is scoped to this gallery; it does not
release the reference-voice pool or establish unrestricted rights for other outputs.
Upstream dataset rights and notices remain applicable. The repository's Apache-2.0
software license does not relicense these audio recordings. No OpenSwara reference
clips, private clinical recordings, reviewer exports, or access codes are included.
