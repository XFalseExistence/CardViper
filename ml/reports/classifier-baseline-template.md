# CardViper 53-way classifier baseline template

Create `classifier-baseline.md` only after training and final TEST evaluation on verified, grouped real data. Copy the sections below and fill them from generated reports; leave this template unchanged.

## Provenance and grouping

- Exact source IDs, versions, licenses, archive SHA-256 and original class-index mapping
- Image-family audit method, reviewed augmentation families and unresolved cases
- Counts by split, independent group and all 53 classes; BACK source and count
- Preflight report path and READY result

## Training

- MobileNetV3Small/ImageNet and input size
- RGB preprocessing, train augmentation, frozen-head and selective fine-tuning phases
- Seed, epochs, batch size, learning rates, selected validation checkpoint, elapsed duration
- Split/crop manifest SHA-256 values and label hash

## Held-out TEST results

- Exact top-1 accuracy, BACK precision/recall, per-class recall/precision
- Worst classes and common confusion pairs
- 2C/2D/2H/2S and red-versus-black 2 analysis
- Confidence distribution and threshold/coverage table
- Known failures and limitations

## Export

- Selected model path and SHA-256, `.tflite` byte size
- Actual input/output tensor names, shapes, dtypes and quantization
- Canonical `labels.txt` hash and frozen `classifier.json` path
- Synthetic smoke outcome separately labeled; never report synthetic accuracy as performance
