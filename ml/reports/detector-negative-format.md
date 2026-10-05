# Explicit no-card detector scenes

Future CardViper-owned table/clutter photos can provide detector negatives, but
an image without annotations is never assumed to contain no cards. Keep original
photos and `scenes.json` under ignored `ml/local/detector-negatives/`. A human
must inspect every scene and set `no_card_confirmed: true`; the importer
rejects missing or false confirmation. No real negative photos are bundled.

```json
{
  "schema_version": 1,
  "source_id": "cardviper-detector-negatives",
  "owner": "Actual owner",
  "rights": {
    "detector": true,
    "evidence": "Actual ownership or consent reference"
  },
  "scenes": [
    {
      "scene_id": "unique-scene-001",
      "session_id": "actual-capture-session-001",
      "capture_date": "2026-10-04",
      "device": "Actual camera/device",
      "image_path": "session-001/table.png",
      "image_sha256": "REPLACE_WITH_SHA256_OF_ORIGINAL_BYTES",
      "annotation_provenance": "Who inspected the scene and how",
      "no_card_confirmed": true,
      "held_out": false
    }
  ]
}
```

Image paths are relative to the source root. SHA-256, source pixels, dates,
session IDs, rights, and the human no-card confirmation are checked before
normalization. Run
`cardviper-import-detector-negatives --root ml/local/detector-negatives --output ml/local/negative-scenes.jsonl`.
Confirmed negatives serialize as canonical B1 `ImageRecord` rows with
`objects: []`. The output is still private; it grants no training permission
in `datasets/sources.json`. Split generation must remain grouped by source,
scene, and session, with held-out scenes isolated.

For face/BACK positive records, run
`cardviper-detector-records --manifest path/to/approved.jsonl --output path/to/detector.jsonl`.
All source identities collapse to detector class ID 0, `CARD`, while source
provenance, original pixel boxes, grouping, and held-out state remain recorded.
This conversion does not bypass any B1/B2 source or family-review gate.
