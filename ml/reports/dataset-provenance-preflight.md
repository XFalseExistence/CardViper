# Playing-card seed provenance preflight

Status on 2026-09-24: **unresolved and unusable for training**.

The Phase B plan identifies the seed only as a “CC BY 4.0 52-class Playing
Cards dataset.” It does not record an owner, Roboflow workspace/project slug,
dataset version, export identifier, image count, retrieval date, attribution,
or source archive checksum. Those fields are necessary to distinguish the
intended source from many similarly named 52-class projects.

## Direct source pages checked

The following canonical Roboflow Universe project pages were opened directly,
not inferred from search snippets:

| Owner | Canonical project | Page facts observed | Why it cannot be selected |
|---|---|---|---|
| Joshuas Workspace | <https://universe.roboflow.com/joshuas-workspace/playing-cards-9gfac> | 52 classes, 3,596 images, one dataset version, CC BY 4.0 | The plan does not name this owner/slug or its version/export. |
| Simply Connected Systems | <https://universe.roboflow.com/simply-connected-systems/playing-cards-wjh4e> | 52 classes, 23 dataset versions, CC BY 4.0; current hosted model shown against project version 23 | The plan gives no version, and this project has many materially distinct versions. |
| plsmeow | <https://universe.roboflow.com/plsmeow/playing-cards-n7nhw-yawwz> | 52 classes, 9,900 project images, one dataset version, CC BY 4.0; hosted model reports a 27,153-image generated dataset | The counts and hosted-model derivative differ; the plan does not identify either artifact. |
| Playing Card Recognition | <https://universe.roboflow.com/playing-card-recognition/playing-cards-pzvb1> | 52 classes, 1,412 images, one dataset version, CC BY 4.0 | The plan does not name this owner/slug or its version/export. |
| PlaycardsDetection | <https://universe.roboflow.com/playcardsdetection/playing-cards-detection> | 52 classes, 6,680 images, one dataset version, CC BY 4.0 | Labels use mixed case and the plan does not name this source. |

These are distinct publishers and datasets even though their titles, class
counts, and displayed licenses overlap. Selecting one would be guesswork.

Each page displays CC BY 4.0 in its **dataset citation** and names its own
author/publisher. That evidence applies only after the exact project and export
are selected and recorded. A page also offering hosted inference or a trained
model does not make hosted service access, generated datasets, or trained
weights interchangeable with the raw dataset license. CardViper will train only
from a locally exported dataset whose exact project/version and attribution are
locked; it will not import a hosted model or rely on hosted inference.

## Current tracked decision

`playing-cards-seed` remains unchanged:

- `url`, `version`, `license`, and `attribution`: null
- `claimed_license`: `CC-BY-4.0`
- `verified`: false
- detector/classifier permissions: false

Consequently, the classifier preflight rejects any split using this source.

## Evidence required to unlock it

Provide or recover all of the following from the dataset-selection record or
the exact local export:

1. canonical project URL and owner/workspace;
2. immutable project/version or export identifier and retrieval date;
3. source-page license and required attribution text;
4. original export class-index list and export format;
5. archive or normalized-image checksums sufficient to connect local files to
   that version;
6. confirmation that dataset reuse permits the intended local classifier and,
   separately, detector training.

Only then should `sources.json` be updated and the two training permissions be
decided independently. No B2 training should start before that lock.
