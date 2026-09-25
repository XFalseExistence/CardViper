import hashlib
import json
from dataclasses import replace

from PIL import Image
import pytest

from test_manifest import record


@pytest.mark.parametrize("box,padding,want", [
    ((2, 2, 6, 6), 0, (2, 2, 6, 6)),
    ((0, 2, 4, 6), .25, (0, 1, 5, 7)),
    ((2, 0, 6, 4), .25, (1, 0, 7, 5)),
    ((6, 2, 10, 6), .25, (5, 1, 10, 7)),
    ((2, 4, 6, 8), .25, (1, 3, 7, 8)),
    ((0, 0, 4, 4), .25, (0, 0, 5, 5)),
    ((2, 2, 6, 6), .25, (1, 1, 7, 7)),
    ((2.2, 2.3, 5.2, 5.8), 0, (2, 2, 6, 6)),
])
def test_padded_pixel_bounds_at_each_edge(box, padding, want):
    from cardviper_ml.build_classifier_crops import crop_bounds
    assert crop_bounds(box, 10, 8, padding) == want


@pytest.mark.parametrize("box", [(1, 1, 1, 2), (2, 2, 1, 3), (20, 20, 25, 25), (-1, 0, 2, 2), (0, 0, float("nan"), 5)])
def test_degenerate_crop_rejected_with_reason(box):
    from cardviper_ml.build_classifier_crops import crop_bounds
    with pytest.raises(ValueError, match="bbox"):
        crop_bounds(box, 10, 8)


@pytest.mark.parametrize("padding", [-.1, .51, float("inf"), float("nan")])
def test_padding_is_bounded_and_finite(padding):
    from cardviper_ml.build_classifier_crops import crop_bounds
    with pytest.raises(ValueError, match="padding"):
        crop_bounds((1, 1, 2, 2), 10, 8, padding)


def image_record(tmp_path, label="2D"):
    from cardviper_ml.manifest import Annotation
    root = tmp_path / "source"
    root.mkdir()
    path = root / "a.png"
    image = Image.new("RGB", (10, 8))
    image.putdata([(x, y, 50) for y in range(8) for x in range(10)])
    image.save(path)
    row = record(image_path="a.png", width=10, height=8,
                 image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 objects=(Annotation("7", label, (2, 2, 6, 6)),))
    return root, row


@pytest.mark.parametrize("label", ["2D", "2S", "10H", "BACK"])
def test_generated_pixels_identity_and_provenance_are_exact(tmp_path, label):
    from cardviper_ml.build_classifier_crops import build_crops
    root, row = image_record(tmp_path, label)
    output = tmp_path / "crops"
    crops = build_crops([row], {"seed": root}, output, padding=.25)
    assert len(crops) == 1
    crop = crops[0]
    assert crop["source_id"] == row.source_id
    assert crop["scene_id"] == row.scene_id
    assert crop["session_id"] == row.session_id
    assert crop["source_image"] == row.image_path
    assert crop["source_bbox"] == [2, 2, 6, 6]
    assert crop["source_image_sha256"] == row.image_sha256
    assert crop["source_annotation"] == row.annotation_path
    assert crop["annotation_id"] == "7"
    assert crop["label"] == label
    assert crop["crop_bbox"] == [1, 1, 7, 7]
    with Image.open(output / crop["crop_path"]) as image:
        assert image.size == (6, 6)
        assert image.getpixel((0, 0)) == (1, 1, 50)
        assert image.getpixel((5, 5)) == (6, 6, 50)
    assert json.loads((output / "crops.jsonl").read_text()) == crop
    second = tmp_path / "second"
    assert build_crops([row], {"seed": root}, second, padding=.25) == crops
    assert (second / crop["crop_path"]).read_bytes() == (output / crop["crop_path"]).read_bytes()


def test_negative_images_produce_no_classifier_crops(tmp_path):
    from cardviper_ml.build_classifier_crops import build_crops
    root, row = image_record(tmp_path)
    output = tmp_path / "negative"
    assert build_crops([replace(row, objects=())], {"seed": root}, output) == []
    assert (output / "crops.jsonl").read_text() == ""


def test_source_hash_and_dimensions_guard_against_stale_annotations(tmp_path):
    from cardviper_ml.build_classifier_crops import build_crops
    root, row = image_record(tmp_path)
    with pytest.raises(ValueError, match="dimensions"):
        build_crops([replace(row, width=20)], {"seed": root}, tmp_path / "bad-size")
    Image.new("RGB", (10, 8), "red").save(root / "a.png")
    with pytest.raises(ValueError, match="hash"):
        build_crops([row], {"seed": root}, tmp_path / "bad-hash")


def test_source_symlink_escape_and_output_overwrite_are_rejected(tmp_path):
    from cardviper_ml.build_classifier_crops import build_crops
    root, row = image_record(tmp_path)
    output = tmp_path / "crops"
    build_crops([row], {"seed": root}, output)
    with pytest.raises(FileExistsError):
        build_crops([row], {"seed": root}, output)
    (tmp_path / "outside.png").write_bytes((root / "a.png").read_bytes())
    (root / "link.png").symlink_to(tmp_path / "outside.png")
    with pytest.raises(ValueError, match="escapes"):
        build_crops([replace(row, image_path="link.png")], {"seed": root}, tmp_path / "escape")


def test_cli_workflow_import_split_validate_and_crop(tmp_path):
    """Exercise installed-style module CLIs on local synthetic data end to end."""
    import subprocess
    import sys
    from pathlib import Path
    from cardviper_ml.manifest import read_manifest
    root = tmp_path / "export"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    groups = {}
    for index, label in enumerate(("AC", "BACK", "2S")):
        Image.new("RGB", (8, 8), (index * 50, 20, 30)).save(root / f"images/{index}.png")
        (root / f"labels/{index}.txt").write_text(f"{index} .5 .5 .5 .5\n")
        groups[f"images/{index}.png"] = {"scene_id": f"scene-{index}"}
    classes = tmp_path / "classes.json"
    classes.write_text('["AC", "BACK", "2S"]')
    group_file = tmp_path / "groups.json"
    group_file.write_text(json.dumps(groups))
    sources = Path(__file__).parents[1] / "datasets/sources.json"
    manifest = tmp_path / "all.jsonl"
    split_dir = tmp_path / "splits"
    roots = tmp_path / "roots.json"
    roots.write_text(json.dumps({"playing-cards-seed": str(root)}))

    def run(module, *args):
        return subprocess.run([sys.executable, "-m", f"cardviper_ml.{module}", *map(str, args)],
                              capture_output=True, text=True)

    result = run("labels", tmp_path / "labels.txt")
    assert result.returncode == 0, result.stderr
    assert len((tmp_path / "labels.txt").read_text().splitlines()) == 53
    result = run("import_roboflow", "--root", root, "--source-id", "playing-cards-seed",
                 "--sources", sources, "--classes", classes, "--groups", group_file, "--output", manifest)
    assert result.returncode == 0, result.stderr
    assert [r.objects[0].label for r in read_manifest(manifest)] == ["AC", "BACK", "2S"]
    for command in ("generate", "validate"):
        result = run("splits", command, "--manifest", manifest, "--directory", split_dir)
        assert result.returncode == 0, result.stderr
    result = run("build_classifier_crops", "--manifest", split_dir / "train.jsonl",
                 "--roots", roots, "--output", tmp_path / "crops")
    assert result.returncode == 0, result.stderr
    crop = json.loads((tmp_path / "crops/crops.jsonl").read_text())
    assert crop["label"] == read_manifest(split_dir / "train.jsonl")[0].objects[0].label
    (split_dir / "val.jsonl").write_bytes((split_dir / "train.jsonl").read_bytes())
    result = run("splits", "validate", "--manifest", manifest, "--directory", split_dir)
    assert result.returncode != 0
    assert "Duplicate" in result.stderr
