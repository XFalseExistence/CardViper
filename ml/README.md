# CardViper ML/data foundation

B1 prepares local data for a future **one-class CARD detector** and **53-way
classifier**. It does not train a model, download a dataset, require a GPU/CUDA,
or integrate an Android inference runtime. Android production still uses
`UnavailableCardDetector`. Only Pillow is required at runtime; pytest is for tests.

## Python 3.11 setup and tests

Run from the repository root:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e './ml[test]'
python -m pytest ml/tests -q
```

The installed `cardviper-*` commands below also work as
`python -m cardviper_ml.labels`, `python -m cardviper_ml.import_roboflow`,
`python -m cardviper_ml.splits`, and `python -m cardviper_ml.build_classifier_crops`.
Tests create tiny synthetic images in temporary directories, with no downloads.

## Canonical labels and output order

Android `CardLabelCodec`, `CardRank`, and `CardSuit` remain authoritative.
`labels.py` contains a literal ordered tuple: ranks **A, 2–10, J, Q, K**, each
with suits **C, D, H, S**, followed by **BACK**. Thus indices 0–3 are
AC/AD/AH/AS, 36–39 are 10C/10D/10H/10S, 48–51 are KC/KD/KH/KS, and 52 is BACK.
The order never depends on Enum/dict iteration. Canonical decode is case-sensitive;
`ac`, `TC`, and whitespace aliases are rejected. BACK is a separate type with no
rank or suit. Python tests compare the 53-label vocabulary to the Android codec.

```sh
mkdir -p ml/local
cardviper-labels ml/local/labels.txt
```

Generated labels are ignored. This establishes the **intended training index
order**, not a claim about an existing model's tensors. Later exports must test
both the full codec vocabulary and the real output-index mapping. No model shape,
normalization, quantization or tensor layout is invented in B1.

## Sources, licenses and local roots

`datasets/sources.json` stores metadata only: stable source ID, human-readable
name, URL/export version, license, attribution, local root alias, explicit
detector/classifier training permissions, verification status and notes.

The selected face seed is now locked to Joshuas Workspace's `playing-cards-9gfac`
project, dataset version 2 (`Initial`, generated 2024-09-12). Its public page
verifies the 52 exact face labels, CC BY 4.0 citation, version preprocessing and
augmentation recipe. The requested YOLOv8 archive still requires Roboflow login,
so its bytes, `data.yaml` class-index order, checksums and augmentation-family
grouping have not been audited. Overall `verified` and both training permissions
therefore remain **false**. A source-page fact is not a verified training artifact.
`read_sources` rejects training permission on unverified sources. Future training
entry points must call
`require_training_permission(sources, source_id, "detector" | "classifier")`
for every contributing source. Import/crop preparation itself grants no rights.

That seed supplies 52 face identities, **not BACK**. Register separately licensed
or CardViper-owned BACK photos as additional sources. Include diverse card backs.
The evaluated Goethal and Kim PEX4 candidate is not registered: its public browse
surface exposes only four `Back` images, all in the upstream train split, with
mixed source class names and no dependable scene/session grouping. It cannot
satisfy the grouped 53-class train/validation/test gate.
Detector data also needs explicitly annotated **no-card negative scenes** with
chips, hands, felt and other distractors for false-positive control. An empty
annotation file means a checked negative; a missing file is an error.
Do not use Ultralytics code, pretrained weights or models under this dataset
permission; those require separate approval.

Keep exports outside git or under ignored `ml/local/`. A local `roots.json` maps
stable source IDs to actual filesystem roots, e.g.:

```json
{"playing-cards-seed": "/absolute/path/to/local/export"}
```

Machine-specific paths, dataset images and private Pixel photos stay local.
The tracked source file contains only a portable root alias. Source IDs identify
original provenance, not export splits: use the same source ID across related
exports. Reconcile duplicates rather than invent new source IDs for them.

## Normalized manifest (schema 1, JSONL)

Each line is one `ImageRecord`: `source_id`, required `scene_id`, optional
`session_id`, root-relative `image_path`, `width`, `height`, `image_sha256`,
root-relative `annotation_path`, `held_out`, `schema_version`, and `objects`.
Each object has a source `annotation_id`, exact canonical `label`,
`detector_class: "CARD"`, and `bbox: [left, top, right, bottom]`.

Coordinates are finite **source-raster pixels**, half-open XYXY, fully in bounds;
right/bottom may equal width/height. Fractional coordinates are retained until
crop extraction. The importer/cropper use stored raster dimensions without EXIF
transposition; annotations must refer to that same raster. Negatives retain image
and grouping metadata with `objects: []`. Invalid labels, missing provenance,
bad dimensions, reversed/empty/out-of-bounds boxes, duplicate annotation IDs/boxes,
absolute/traversing paths and nonfinite values fail explicitly. Raw annotation
file path/line ID and image SHA-256 preserve traceability. Input errors are not
silently repaired.

## Local Roboflow/YOLO normalization

No Roboflow SDK, network access, YAML parser or training framework is required.
Use a local **YOLO detection** export: images plus same-stem `.txt` annotations,
one `class_id center_x center_y width height` row per card, normalized to 0..1.
Segmentation polygons and prediction-confidence columns are rejected.

1. Copy the export's `data.yaml` **names in its original numeric index order**
   into a local JSON array, e.g. `ml/local/classes.json` with
   `["7C", "KH", "2D", "10S", "BACK"]`. These are examples, not the seed's order.
   Never substitute classifier output order for export class order.
2. Prepare `ml/local/groups.json`, keyed by each image path relative to the root:

```json
{
  "train/images/frame001.jpg": {"scene_id": "table-001", "session_id": "capture-001"},
  "train/images/frame002.jpg": {"scene_id": "table-001", "session_id": "capture-001"},
  "train/images/pixel-heldout.jpg": {"scene_id": "pixel-001", "session_id": "pixel-session", "held_out": true}
}
```

Group all near-duplicates, crops, augmentations and frames from the same capture
sequence together. Never infer independence from image filenames or upstream
train/valid/test folders. If upstream scene provenance is unavailable, assign
that source conservatively to one scene group until grouping is verified; fewer
than three independent eligible groups cannot form train/val/test.

3. Normalize each export folder, then combine manifests **before** splitting:

```sh
export CARDVIPER_DATA="$HOME/cardviper-data/playing-cards-export"
cardviper-import --root "$CARDVIPER_DATA" --source-id playing-cards-seed \
  --sources ml/datasets/sources.json --classes ml/local/classes.json \
  --groups ml/local/groups.json --images train/images --annotations train/labels \
  --output ml/local/train-import.jsonl
```

Repeat for valid/test folders if present, preserving the same source/group IDs;
combine their JSONL files into `ml/local/all.jsonl`. For a single images/labels
folder, omit `--images/--annotations` and write `ml/local/all.jsonl` directly.
If export labels use aliases, supply an explicit local JSON mapping through
`--class-map`, e.g. `{"Ac":"AC"}`. No lowercase aliases are silently accepted.
All canonical identities including BACK stay exact for the classifier; **every
object's detector semantic class is CARD**, never one of 53 detector classes.

## Grouped splits and independent leakage validation

```sh
cardviper-splits generate --manifest ml/local/all.jsonl \
  --directory ml/local/splits --seed 42
cardviper-splits validate --manifest ml/local/all.jsonl \
  --directory ml/local/splits
```

Outputs: `train.jsonl`, `val.jsonl`, `test.jsonl`, `holdout.jsonl`. Generate into a
new directory; existing files are never overwritten. Source+scene and
source+session edges form connected components, including transitive bridges.
A whole component belongs to one split. Any `held_out: true` member reserves the
**entire component** to holdout, separate from train/val/test. The Pixel 7
hard-case set must remain held out; it is not training or calibration data.

Unique images are assigned once. Duplicate source/path or identical file SHA-256
is rejected even across sources: reconcile metadata/duplicates first. Byte hashes
cannot identify recompressed near-duplicates; honest scene/session metadata is a
hard requirement, not a perceptual-deduplication claim. Group order is seeded
SHA-256 over stable source/image keys, independent of input record order.
Default ratios are 80/10/10 by **group count**, reserving at least one group per
split, then allocating remaining groups by largest remainder. Image/class ratios
may differ; there is no stratification promise. At least three independent,
non-held-out groups are required. The validator checks groups, duplicates,
holdout placement, nonempty train/val/test, and exact original-record coverage
(including annotations), so edited or missing records cannot pass unnoticed.

## Classifier crops with provenance

After split validation, create each split's crops separately:

```sh
cardviper-crops --manifest ml/local/splits/train.jsonl \
  --roots ml/local/roots.json --output ml/local/crops/train --padding 0.1
cardviper-crops --manifest ml/local/splits/val.jsonl \
  --roots ml/local/roots.json --output ml/local/crops/val --padding 0.1
cardviper-crops --manifest ml/local/splits/test.jsonl \
  --roots ml/local/roots.json --output ml/local/crops/test --padding 0.1
```

Each exact source box produces an RGB PNG without resizing, label inference or
augmentation. Padding defaults to zero and is bounded to 0..0.5 of box width/
height per side. Padded bounds clamp to the raster; left/top round down and
right/bottom round up. Degenerate/malformed boxes fail with a reason instead of
producing empty crops. Source dimensions/hash are rechecked; root-escaping
symlinks and stale images are rejected. No-card records produce no crops.

`crops.jsonl` records original source/scene/session, held-out flag, image path and
hash, image dimensions, source annotation file/ID, original box, exact label,
padding, integer crop box and generated relative crop path. Crop filenames are
content/provenance-derived; deterministic reruns with the same toolchain produce
identical files. Existing output directories are rejected. A failed run may leave
partial PNGs but **no completed crop manifest**; investigate and use a fresh
output directory. Downstream training must consume the manifest, not glob PNGs.

## Git safety and next boundary

Ignored: `ml/local/`, raw/download/private dataset folders, generated crops/splits,
training runs, checkpoints, model exports, Python environments/caches, and private
Pixel acceptance images. Source/license metadata, tools, tests and this README
stay tracked. Tests generate fixtures temporarily; deliberately approved tiny
fixtures may live under `ml/tests/fixtures/`. Future model binaries require
explicit approval and deliberate ignore-rule changes, not an accidental add.

Before a checkpoint, inspect `git status --short`, `git diff --check`, and the
staged file list. **B1 does not train the classifier/detector or start Android
runtime integration.** B2 requires explicit approval and verified data provenance.

## Classifier readiness gate

Before B2 training, validate the locked split directory and every contributing
source license/permission:

```sh
cardviper-classifier-preflight \
  --sources ml/datasets/sources.json \
  --splits ml/local/splits \
  --json ml/local/classifier-preflight.json \
  --markdown ml/local/classifier-preflight.md
```

The command exits zero only for `READY`; `NOT READY` exits 2. It reuses the B1
split validator, rejects unverified classifier sources, duplicates, leakage and
held-out data in train/validation/test, then requires all 53 classes in each of
train, validation and test. That per-split requirement supports B2’s planned
per-class recall; it is a coverage gate, not a balance requirement. The report
quantifies images, annotations, independent groups, class distribution, BACK,
held-out images and permitted source IDs in JSON and Markdown.

See [dataset provenance preflight](reports/dataset-provenance-preflight.md) for
why the current 52-class seed remains blocked, and the
[CardViper-owned data plan](reports/cardviper-owned-data-plan.md) for BACK and
realistic detector-negative capture requirements.
