# Classifier data provenance lock

Status on 2026-09-25: **selected sources researched; classifier data NOT READY**.

The face source is no longer ambiguous. It is locked to Joshuas Workspace's
`playing-cards-9gfac` project, version 2. The source remains unusable for training
because the exact export is login-gated and has not been acquired or audited.
The evaluated BACK candidate is rejected as insufficient.

## Selected 52-face source

Canonical project:
<https://universe.roboflow.com/joshuas-workspace/playing-cards-9gfac>

Canonical version:
<https://universe.roboflow.com/joshuas-workspace/playing-cards-9gfac/dataset/2>

Facts verified directly from those pages:

- owner/workspace: **Joshuas Workspace** / `joshuas-workspace`;
- project slug: `playing-cards-9gfac`;
- task: object detection;
- project source images: 3,596;
- one published dataset version, identified as **v2**, name `Initial`, generated
  2024-09-12;
- v2 materialized images: 10,233 (10,014 train, 153 validation, 66 test);
- exactly 52 project classes: `10C`, `10D`, `10H`, `10S`, `2C`, `2D`, `2H`,
  `2S`, `3C`, `3D`, `3H`, `3S`, `4C`, `4D`, `4H`, `4S`, `5C`, `5D`, `5H`,
  `5S`, `6C`, `6D`, `6H`, `6S`, `7C`, `7D`, `7H`, `7S`, `8C`, `8D`, `8H`,
  `8S`, `9C`, `9D`, `9H`, `9S`, `AC`, `AD`, `AH`, `AS`, `JC`, `JD`, `JH`,
  `JS`, `KC`, `KD`, `KH`, `KS`, `QC`, `QD`, `QH`, `QS`;
- license shown by the project citation: **CC BY 4.0**;
- citation author/publisher: `Joshuas Workspace`, published through Roboflow
  Universe, year 2024;
- preprocessing: stretch-resize to 640x640;
- augmentation: three outputs per training example and brightness from -15% to
  +15%;
- listed local export formats include YOLOv8, other YOLO variants, COCO JSON,
  Pascal VOC XML, TFRecord, PaliGemma and CreateML JSON.

The intended B1-compatible artifact is the version-2 YOLOv8 export:
<https://universe.roboflow.com/joshuas-workspace/playing-cards-9gfac/dataset/2/download/yolov8>.
Opening that exact endpoint on 2026-09-25 displayed `Login or create a free
account`; no anonymous archive was returned. No account was created and no API
key or secret was requested.

Consequently, these material artifact facts remain unverified:

- archive bytes and SHA-256;
- the export's original numeric class-index order from `data.yaml`;
- annotation/image integrity;
- original-to-generated image family identifiers;
- trustworthy scene/session grouping for leakage-safe resplitting.

The v2 totals and augmentation recipe prove generated variants exist. CardViper
must not trust the upstream 98/1/1 image split as independent evidence. Generated
variants and any near-duplicate originals must share one CardViper group. Until a
local archive supports that audit, `playing-cards-seed` remains `verified: false`
with detector and classifier permissions false.

## Evaluated BACK candidate

Project:
<https://universe.roboflow.com/goethal-and-kim-pex4/playing-cards-8ycnm-th85w>

Version:
<https://universe.roboflow.com/goethal-and-kim-pex4/playing-cards-8ycnm-th85w/dataset/1>

Verified public facts:

- owner/workspace: **Goethal and Kim PEX4** / `goethal-and-kim-pex4`;
- project slug: `playing-cards-8ycnm-th85w`;
- task: object detection;
- project/version: 149 images, v1 generated 2025-11-18;
- split: 105 train, 29 validation, 15 test;
- license displayed by the project citation: **Public Domain**;
- no preprocessing and no augmentation reported;
- the class filter exposes a literal `Back` annotation identity in addition to
  rank labels and other mixed classes;
- filtering the public image browser for exact class `Back` returned four images,
  all assigned to the upstream train split;
- one inspected multi-card example contained exactly one `Back` box among face
  boxes; filenames indicate only a very small set of source examples;
- the exact YOLOv8 endpoint is also login-gated.

This candidate is **rejected** as `playing-cards-back-seed`. Four visible examples,
all in one upstream split, do not provide adequate BACK diversity or the three
independent grouped partitions required by CardViper's preflight. The project also
shows inconsistent class surfaces (`CardBack` in the overview versus `Back`,
`Card`, and `null` in the image filter), and provides no scene/session lineage.
CardViper therefore does not register it or map any non-`Back` alias to `BACK`.

## Exact unblock sequence

1. A user with lawful Roboflow access downloads the selected face project **v2**
   as YOLOv8 and places the archive/extracted export under ignored
   `ml/local/sources/playing-cards-seed/`.
2. Record the archive SHA-256 and retain its original `data.yaml`.
3. Verify the 52 numeric class indices against the exact canonical face set.
4. Audit hashes and perceptual near-duplicates, then assign conservative
   source/scene/session groups so every original and generated variant stays
   together.
5. Acquire a separately licensed or CardViper-owned BACK corpus with independent
   train/validation/test capture groups and diverse back designs.
6. Normalize, combine, generate new CardViper grouped splits and crops, then run
   `cardviper-classifier-preflight`.
7. Promote sources and enable training permissions only after the preflight is
   `READY` with every one of the 53 labels present in train, validation and test.

No raw dataset, model binary, generated split or machine-specific path is tracked
by this checkpoint. No classifier training or tensor contract can begin yet.
