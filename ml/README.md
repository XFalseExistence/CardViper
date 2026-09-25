# CardViper ML/data tooling

B1 checkpoint: canonical labels, provenance manifests, local YOLO import and
validated grouped splits. Crop generation and final B1 documentation are pending.
No neural training or Android runtime integration is included.

From the repository root, using Python 3.11:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e './ml[test]'
python -m pytest ml/tests -q
```

Keep raw/private datasets and generated files in ignored `ml/local/` or outside
this repository. `ml/datasets/sources.json` is metadata only. The seed named by
the approved plan lacks an exact URL/version; it remains unusable for training
until provenance and license are verified. BACK needs separately licensed data;
no-card negatives are required for later detector training.
