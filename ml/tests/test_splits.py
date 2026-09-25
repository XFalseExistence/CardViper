from dataclasses import replace
import hashlib

import pytest

from test_manifest import record


def dataset(count=12):
    return [record(scene_id=f"scene-{i // 2}", session_id=f"session-{i // 4}",
                   image_path=f"images/{i}.png", image_sha256=hashlib.sha256(str(i).encode()).hexdigest())
            for i in range(count)]


def test_grouped_split_deterministic_order_independent_and_all_records_once(tmp_path):
    from cardviper_ml.splits import split_records, validate_splits, write_splits, read_splits
    rows = dataset(24)
    first = split_records(rows, seed=42)
    second = split_records(list(reversed(rows)), seed=42)
    assert first == second
    assert all(first[name] for name in ("train", "val", "test"))
    assert sorted(r.image_path for records in first.values() for r in records) == sorted(r.image_path for r in rows)
    ownership = {}
    for name, records in first.items():
        for r in records:
            for key in ((r.source_id, "scene", r.scene_id), (r.source_id, "session", r.session_id)):
                assert ownership.setdefault(key, name) == name
    validate_splits(first, expected=rows)
    for path, value in ((tmp_path / "one", first), (tmp_path / "two", second)):
        write_splits(path, value)
    assert read_splits(tmp_path / "one") == first
    for path in (tmp_path / "one").iterdir():
        assert path.read_bytes() == (tmp_path / "two" / path.name).read_bytes()


def test_transitive_scene_and_session_connections_cannot_leak():
    from cardviper_ml.splits import split_records
    rows = dataset(24)
    # Bridge two sessions via one shared scene; union must be transitive.
    rows[4] = replace(rows[4], scene_id=rows[0].scene_id)
    result = split_records(rows)
    owners = {r.image_path: name for name, group in result.items() for r in group}
    assert len({owners[r.image_path] for r in rows[:8]}) == 1


def test_holdout_reserves_entire_related_sequence():
    from cardviper_ml.splits import split_records, validate_splits
    rows = dataset(24)
    rows[0] = replace(rows[0], held_out=True)
    result = split_records(rows)
    assert {r.image_path for r in result["holdout"]} == {f"images/{i}.png" for i in range(4)}
    validate_splits(result, expected=rows)
    result["train"].extend(result.pop("holdout"))
    result["holdout"] = []
    with pytest.raises(ValueError, match="held.out"):
        validate_splits(result, expected=rows)


@pytest.mark.parametrize("kind", ["same_record", "same_path", "same_content"])
def test_duplicate_images_rejected_instead_of_leaking(kind):
    from cardviper_ml.splits import split_records
    rows = dataset()
    duplicate = rows[0]
    if kind == "same_path":
        duplicate = replace(duplicate, scene_id="different", image_sha256="f" * 64)
    if kind == "same_content":
        duplicate = replace(duplicate, source_id="other", image_path="other.png", scene_id="other")
    with pytest.raises(ValueError, match="Duplicate"):
        split_records(rows + [duplicate])


def test_validator_rejects_cross_split_groups_missing_extra_or_modified_records():
    from cardviper_ml.splits import split_records, validate_splits
    rows = dataset(24)
    good = split_records(rows)
    bad = {key: list(values) for key, values in good.items()}
    bad["val"].append(bad["train"].pop())
    with pytest.raises(ValueError, match="group"):
        validate_splits(bad)
    bad = {key: list(values) for key, values in good.items()}
    bad["test"].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_splits(bad, expected=rows)
    bad = {key: list(values) for key, values in good.items()}
    bad["train"][0] = replace(bad["train"][0], objects=())
    with pytest.raises(ValueError, match="coverage"):
        validate_splits(bad, expected=rows)


def test_small_dataset_fails_explicitly_but_three_groups_is_valid():
    from cardviper_ml.splits import split_records
    with pytest.raises(ValueError, match="at least 3"):
        split_records(dataset(8))
    assert [len(v) for k, v in split_records(dataset(12)).items() if k != "holdout"] == [4, 4, 4]


@pytest.mark.parametrize("ratios", [(1, 0, 0), (.8, .1, float("nan")), (.8, .1, -.1), (.2, .2, .2)])
def test_invalid_split_ratios_fail(ratios):
    from cardviper_ml.splits import split_records
    with pytest.raises(ValueError):
        split_records(dataset(), ratios=ratios)


def test_missing_group_metadata_fails_when_loading_split_files(tmp_path):
    from cardviper_ml.manifest import write_manifest
    from cardviper_ml.splits import read_splits
    for name in ("train", "val", "test", "holdout"):
        write_manifest(tmp_path / f"{name}.jsonl", dataset(4))
    path = tmp_path / "train.jsonl"
    path.write_text(path.read_text().replace('"scene_id": "scene-0"', '"scene_id": ""'))
    with pytest.raises(ValueError, match="scene_id"):
        read_splits(tmp_path)
