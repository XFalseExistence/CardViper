# Offline Joshua family review

Build the pack from the audited Joshua ZIP and the matching local reports:

```sh
cardviper-build-family-review \
  --archive '/Volumes/Toshi4TB/Playing Cards.v2-initial.yolov8.zip' \
  --audit '/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-artifact-audit-pushed.json' \
  --family-audit '/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-family-audit-pushed.json' \
  --proposal '/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-family-proposal-pushed.json' \
  --output '/Volumes/Toshi4TB/DEV_CACHE/CardViper/family-review'
```

Open `index.html` directly in a browser. It uses only local thumbnails and
embedded data; it sends nothing to a server. `S`, `I`, and `U` choose same
family, independent, or unsure for the current candidate bucket. The page shows
machine recommendations where evidence is strong, but the recommendation button
records a human click. It shows reviewed/unresolved counts, filters, and a next
unresolved button. Download decisions JSON periodically and use the resume
control to load it again if the browser closes. The notes
field, reviewer, date, method, and approval checkbox are part of the downloaded
`review-decisions.json`. Keep that file private with the images. The initial
`review.json` is an unapproved draft with every decision set to unsure.

Each candidate is a coarse dimensions-and-label-set bucket whose members still
span proposed families. Its image hashes, annotation hashes, report hashes, and
archive hash bind the decision to the exact source. **SAME FAMILY** conservatively
joins every current family in that bucket. **INDEPENDENT** asserts that every
cross-family relationship in that bucket was inspected and is independent;
review all displayed images before using it. **UNSURE** leaves the bucket
unresolved. Difference-hash and rounded annotation geometry are review clues,
not proof of shared origin or independence. Missing decisions and stale source
or report evidence are rejected.

Apply only a completed human decision file:

```sh
cardviper-apply-family-review \
  --archive '/Volumes/Toshi4TB/Playing Cards.v2-initial.yolov8.zip' \
  --audit '/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-artifact-audit-pushed.json' \
  --family-audit '/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-family-audit-pushed.json' \
  --proposal '/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-family-proposal-pushed.json' \
  --review '/Volumes/Toshi4TB/DEV_CACHE/CardViper/family-review/review-decisions.json' \
  --output '/Volumes/Toshi4TB/DEV_CACHE/CardViper/family-review/reviewed-groups.json'
```

The output remains NOT READY if any bucket is unresolved or approval is absent.
An approved file containing any `unsure` decision is rejected. It never changes
the ZIP or source permissions. All 224 known exact-duplicate sets are already
within single proposed groups; the importer keeps those images together and
rejects any exact set that crosses proposed or reviewed groups. Roboflow's
upstream train/valid/test folders are not used as independent split evidence.

The review pack reports what would happen if **every** unresolved bucket were
merged. This is analysis only. In the current Joshua export that creates a
pathological giant component, so the tool does not expose an automatic merge-all
action.
