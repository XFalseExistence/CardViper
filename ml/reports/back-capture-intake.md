# Local BACK capture intake

Photograph real card backs you own and have permission to use. Keep the original
image bytes. Capture separate physical sessions (different outings or setups, not
adjacent frames renamed as sessions) and at least two physical back designs.
Store files in the ignored private directory, for example
`ml/local/cardviper-back/inbox/session-001/photo-001.jpg`. Do not commit photos
or `captures.json`; `/ml/local/` is already ignored.

From the repository root, with the ML package installed (`python -m pip install
-e 'ml[test]'`), run:

```sh
cardviper-back-capture intake --root ml/local/cardviper-back
cardviper-back-capture validate --root ml/local/cardviper-back
```

Intake measures SHA-256 and raster dimensions. It does not infer capture date,
device, session ID, design, ownership, rights, or annotation. A draft is NOT
READY. To supply explicit metadata, create a private JSON file outside Git,
then pass `--metadata /path/to/metadata.json` to `intake`. Its `common`
object applies to the batch and `images` maps paths relative to the BACK root
to per-image overrides:

```json
{
  "common": {
    "owner": "Photographer name",
    "rights": {
      "classifier": true,
      "detector": false,
      "evidence": "Reference to actual consent or ownership record"
    },
    "capture_date": "2026-10-04",
    "device": "Actual camera/device",
    "deck_design": "Physical design identifier",
    "annotation_provenance": "Name and method of human annotation",
    "capture_mode": "tight-back-crop"
  },
  "images": {
    "inbox/session-001/photo-001.jpg": {
      "session_id": "Actual independent session ID",
      "scene_id": "Unique scene ID",
      "full_image_back": true
    }
  }
}
```

Only set `full_image_back: true` when the stored image itself is one tight
card-back crop. For `annotated-scene`, supply hand-reviewed `boxes` using
source-pixel `[left, top, right, bottom]` coordinates, each with an
`annotation_id` and `label: "BACK"`. See
[the authoritative format](back-capture-format.md) for the complete schema and
human review requirements. If dates, designs, or sessions differ by photo, put
them in `images`; never apply a batch value that is not true for every photo.

The intake command preserves existing entries and refuses conflicting metadata
or changed image bytes. It writes `captures.json` atomically. Validation
reports missing fields, files, hash mismatches, exact duplicates, session and
design counts, and the existing BACK audit's READY/NOT READY result. Validation
exits nonzero until READY. To revise a known conflict or add human review, edit
the private manifest deliberately and rerun validation. Intake never approves
review or promotes source permissions.

After genuine BACK data and independent human review are complete, rerun the
Joshua onboarding dry run with the real face ZIP and
`--back-source ml/local/cardviper-back`; the separate face-family review gate
will remain until its 137 candidates have been reviewed by a human.
