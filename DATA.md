# Data access and redistribution

The evaluation uses public simulated or non-patient clinical conversations.
This repository distributes manifests, dataset revisions, aggregate results,
and checksums, not the source recordings.

| Dataset | Frozen source | Scale used | License / access |
|---|---|---:|---|
| PriMock57 | Git revision `cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5` | 57 constructed conversations | CC BY 4.0 |
| Fareez respiratory interviews | Figshare `10.6084/m9.figshare.16550013.v1` | 272 conversations | CC0 1.0 |
| SIMSAMU | Hugging Face revision `6b639431f658d636c30dbf342074d723bf303e0f` | 61 simulated calls | MIT |
| ASQ-PHI | DOI `10.17632/csz5dzp7nx.1` | 1,051 fully synthetic text queries | MIT |

The pseudo-speaker pool was built from OpenSwara revision
`3fc1ab35af18faf42ad1d51f04a5d83dc644c7d2`. Although the publisher describes
the collection as CC BY-SA 4.0, upstream provenance is not uniformly clear.
Therefore no OpenSwara source audio, reference audio, transformed audio, or
review audio is redistributed here. Recreate the pool from lawfully obtained
upstream material using the released manifest/index builders.

The limited listener-study exports and reviewer codes are also excluded. Only
aggregate, non-identifying statistics included in the paper artifact are
released.
