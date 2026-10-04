import hashlib
import json
import zipfile

from PIL import Image

from cardviper_ml.build_family_review import build_review_pack


def review_fixture(tmp_path):
    source = tmp_path / "source"
    names = []
    for index, color in enumerate(("red", "blue")):
        image = source / "train" / "images" / f"card-{index}_jpg.rf.{index:032x}.jpg"
        label = source / "train" / "labels" / f"card-{index}_jpg.rf.{index:032x}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 16), color).save(image)
        label.write_text("0 .5 .5 .5 .5\n")
        names.append(image.relative_to(source).as_posix())
    archive = tmp_path / "faces.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                packed.write(path, path.relative_to(source).as_posix())
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"artifact_status": "SOURCE_ARTIFACT_VALID",
                                 "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                 "exact_duplicate_files": []}))
    family = tmp_path / "family.json"
    family.write_text(json.dumps({"unresolved_similarity_candidates": [names]}))
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"schema_version": 1, "source_id": "playing-cards-seed",
                                    "artifact_fingerprint": "a" * 64,
                                    "groups": {names[0]: "g0", names[1]: "g1"}}))
    return archive, audit, family, proposal


def test_offline_pack_binds_image_hashes_and_starts_unapproved(tmp_path):
    args = review_fixture(tmp_path)
    output = tmp_path / "review"
    report = build_review_pack(*args, output)
    assert report["candidate_count"] == 1
    assert (output / "index.html").is_file()
    assert len(list((output / "images").glob("*.jpg"))) == 2
    data = json.loads((output / "review.json").read_text())
    assert data["approved"] is False
    assert data["decisions"][0]["decision"] == "unsure"
    candidate = data["candidates"][0]
    assert len(candidate["images"]) == 2
    assert all(len(image["sha256"]) == 64 for image in candidate["images"])
    assert candidate["candidate_id"] == data["decisions"][0]["candidate_id"]
    assert "http://" not in (output / "index.html").read_text()
