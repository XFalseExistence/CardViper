# CardViper-owned data capture plan

This plan closes the seed dataset’s two known gaps without committing private
photos. Actual captures live outside git or under ignored `ml/local/`. A source
record is added only after a capture batch has an owner, date/session IDs,
consent/provenance, usage rights, checksums, and a stable local-root alias.

An unverified template would begin with `verified: false` and both
`permitted_use` values false. It must not be promoted merely because CardViper
created the folder; the capture provenance and rights record must exist first.

## BACK classifier captures

Capture multiple physical card-back designs rather than teaching one deck’s
artwork. Keep each continuous capture as one session group. Within and across
sessions include:

- rotations and scale variation;
- chest-level perspective and overhead-ish reference views;
- glare, exposure shifts, and hard shadow;
- partial occlusion and hand overlap;
- edge/corner truncation;
- felt and ordinary table backgrounds;
- motion blur and realistic compression;
- varied detector-box/crop padding.

Every visible card-back box is labeled exactly `BACK`. No rank or suit is
invented. Capture train/validation/test sessions independently; do not split
near-duplicate frames. Do not use the Pixel 7 acceptance images for training or
calibration.

## Detector negatives and full-table scenes

Capture realistic empty or card-free table regions containing:

- chips and chip stacks;
- hands and arms;
- felt;
- drinks, coasters, and ordinary table clutter;
- chip trays and discard trays;
- card-like rectangular distractors;
- phones/screens where reasonable;
- mixed lighting, glare, shadows, blur, and chest-level geometry.

No-card negative images are valid normalized records with `objects: []`.
Missing annotation files are not negatives. Full-table positive scenes remain
one semantic detector class, `CARD`, while retaining exact classifier labels in
the normalized manifest.

## Capture registration checklist

1. Assign a source ID and local-root alias for the capture batch.
2. Record owner, capture date, device/deck designs, consent and usage rights.
3. Hash source images and preserve scene/session grouping before augmentation.
4. Register the source in `sources.json` with separate classifier/detector
   permissions supported by that evidence.
5. Normalize annotations, create grouped splits, and run
   `cardviper-classifier-preflight` before training.
6. Keep the Pixel 7 hard-case acceptance set in `holdout.jsonl`; never copy it
   into train or validation.
