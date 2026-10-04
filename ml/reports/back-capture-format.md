# CardViper-owned BACK capture format

Keep original photos and `captures.json` under an ignored local directory such as
`ml/local/cardviper-back/`. Nothing in this format fetches images or changes
source pixels. Register the batch separately in `datasets/sources.json` with
`source_id: "cardviper-back"` and a `local_root` environment-variable name
such as `CARDVIPER_BACK_ROOT`; point that variable at the directory containing
`captures.json`. Register training permissions only when the audit is `READY`
and the provenance and rights evidence has been reviewed. No real capture files
are shipped with this repository.

To draft `captures.json` without typing hashes, put original photos directly
under distinct session folders, then run:

```sh
cardviper-prepare-back-captures --root /Volumes/Toshi4TB/CardViperData/cardviper-back \
  --device "Pixel 7" --capture-mode tight-back-crop
```

The helper refuses to overwrite an existing manifest. It records image paths,
SHA-256, raster dimensions, device, and session-folder IDs. It leaves owner,
rights evidence, deck design, annotation provenance, review approval, and any
missing capture dates blank. For tight crops it sets `full_image_back: false`
until a person confirms each stored image is one card back; for annotated scenes
it leaves `boxes: []` until real pixel boxes are supplied. The draft is **NOT
READY** and cannot pass the importer without those explicit decisions.

Aim for at least four independently captured sessions, two physical back
designs, and roughly 20–30 usable images per session. Vary rotation, distance,
scale, background placement, lighting, shadows, mild glare, and partial
occlusion. Avoid exact duplicates. A tight-back-crop file contains one card
back; a wider scene needs hand-reviewed boxes. These are capture targets, not
an automatic diversity or readiness claim.

The following JSON is a **format illustration with placeholder hashes**, not
production data. Replace every placeholder and compute SHA-256 from the
original file bytes. Each entry needs a distinct normalized path, a capture
date, device, continuous session ID, scene ID, physical deck-back design,
annotation provenance, and explicit capture mode. The common owner and rights
record applies to every entry in this batch.
Set `rights.detector` to `false` for classifier-only permission. A `true`
detector permission also requires a separate, nonempty `detector_evidence`
field describing that grant; classifier evidence alone does not grant it.

```json
{
  "schema_version": 1,
  "source_id": "cardviper-back",
  "owner": "OWNER NAME",
  "rights": {
    "classifier": true,
    "detector": false,
    "evidence": "Owner consent and classifier-use authorization reference; detector use not granted"
  },
  "captures": [
    {
      "capture_date": "2026-10-03",
      "device": "DEVICE AND CAMERA DETAILS",
      "session_id": "continuous-session-001",
      "scene_id": "continuous-session-001-frame-001",
      "deck_design": "PHYSICAL BACK DESIGN ID",
      "image_path": "images/frame-001.png",
      "image_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_DIGITS",
      "annotation_provenance": "Annotator, method, and review reference",
      "capture_mode": "tight-back-crop",
      "full_image_back": true
    },
    {
      "capture_date": "2026-10-03",
      "device": "DEVICE AND CAMERA DETAILS",
      "session_id": "continuous-session-002",
      "scene_id": "continuous-session-002-frame-001",
      "deck_design": "SECOND PHYSICAL BACK DESIGN ID",
      "image_path": "images/frame-002.png",
      "image_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_DIGITS",
      "annotation_provenance": "Annotator, method, and review reference",
      "capture_mode": "annotated-scene",
      "boxes": [
        {"annotation_id": "back-1", "label": "BACK", "bbox": [120, 75, 280, 310]}
      ]
    }
  ],
  "review": {
    "reviewer": "HUMAN REVIEWER",
    "notes": "Actual session independence, designs, and similarity candidates inspected",
    "approved": true,
    "session_independence_confirmed": true,
    "near_duplicate_clues_reviewed": true
  }
}
```

Use `tight-back-crop` only when the stored image itself is a tight crop of a
single card back. `full_image_back: true` explicitly authorizes its full-image
box `(0, 0, width, height)`. This mode accepts no separate boxes. Use
`annotated-scene` for a wider view; it requires one or more real, hand supplied
`BACK` boxes in source-pixel `(left, top, right, bottom)` coordinates. A missing
box list is never inferred as a full-image box or a negative. Stored raster
geometry is used without EXIF rotation. Preserve the original image files and
their SHA-256 digests; if the bytes change, recapture provenance and reannotate.

Keep every frame from a continuous capture in the same `session_id` and put
related images in the same `scene_id`. Separate physical recording sessions
need distinct IDs. Mark reserved acceptance images with `held_out: true` and
keep them outside training. The importer preserves both IDs and held-out state
in `ImageRecord`; its `annotation_path` points to `captures.json`, which holds
the original annotation provenance.

Run the importer from `ml/` with the project's Python environment:

```sh
python -m cardviper_ml.import_back_captures \
  --root local/cardviper-back \
  --output local/cardviper-back.normalized.jsonl \
  --report local/cardviper-back.audit.json
```

The Python API is `import_back_source(root, *, capture_mode=None)` for normalized
`ImageRecord` rows and `audit_back_source(root, *, capture_mode=None)` for an
audit dictionary. Optional `capture_mode` restricts the entire batch to one of
the two modes and rejects mismatches. Both require the same local
`captures.json`; neither changes image files. The CLI writes only the two
explicitly named output files.

The audit reports samples, annotations, sessions, physical designs, image
dimensions, exact SHA-256 duplicate groups, and difference-hash similarity
clues. It also reports session/design distributions and counts independent
non-held-out scene/session connected groups. `split_eligibility: READY` means
there are at least three such groups, classifier rights are granted, and no exact
duplicate remains. Detector rights are reported independently and remain false
until separately justified. Overall `status: READY` additionally requires more than
one physical design and an explicit human review of session independence and
similarity clues. `reasons` explains each `NOT READY` result. Counts, hashes,
and perceptual similarity are evidence for review; they do not establish
visual diversity or independent capture by themselves. A READY audit remains
a dataset prerequisite, not a model quality or device validation claim.
