"""Deterministic source/scene/session-connected splits with independent validation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .manifest import ImageRecord, read_manifest, write_manifest

SPLIT_NAMES = ("train", "val", "test", "holdout")


def record_key(row):
    return row.source_id, row.image_path


def group_keys(row):
    keys = [(row.source_id, "scene", row.scene_id)]
    if row.session_id is not None:
        keys.append((row.source_id, "session", row.session_id))
    return keys


def unique_records(rows):
    keys, digests = set(), set()
    for row in rows:
        if not isinstance(row, ImageRecord):
            raise ValueError("Expected validated ImageRecord with scene metadata")
        if record_key(row) in keys or row.image_sha256 in digests:
            raise ValueError("Duplicate image path or content; reconcile provenance before splitting")
        keys.add(record_key(row))
        digests.add(row.image_sha256)


def connected_groups(rows):
    """Union both scene and session edges, including transitive connections."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in group_keys(row):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = {}
    for i, row in enumerate(rows):
        components.setdefault(find(i), []).append(row)
    return [sorted(group, key=record_key) for group in components.values()]


def validate_splits(splits, *, expected=None):
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Expected train, val, test and holdout split files")
    rows = [row for name in SPLIT_NAMES for row in splits[name]]
    unique_records(rows)
    if expected is not None:
        expected = list(expected)
        unique_records(expected)
        if sorted(rows, key=record_key) != sorted(expected, key=record_key):
            raise ValueError("Split coverage differs from original manifest (missing, extra or modified records)")
    owner = {}
    for name in SPLIT_NAMES:
        for row in splits[name]:
            if row.held_out and name != "holdout":
                raise ValueError("Pixel/held-out image is outside holdout")
            for key in group_keys(row):
                if owner.setdefault(key, name) != name:
                    raise ValueError(f"Leakage: group {key!r} appears in multiple splits")
    if any(not splits[name] for name in SPLIT_NAMES[:3]):
        raise ValueError("train, val and test must each contain at least one group")


def split_records(records, *, seed=42, ratios=(0.8, 0.1, 0.1)):
    if len(ratios) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Three positive finite split ratios must sum to one")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rows = sorted(records, key=record_key)
    unique_records(rows)
    result = {name: [] for name in SPLIT_NAMES}
    available = []
    for group in connected_groups(rows):
        if any(row.held_out for row in group):
            result["holdout"].extend(group)
        else:
            available.append(group)
    if len(available) < 3:
        raise ValueError("Need at least 3 independent non-held-out scene/session groups")

    def seeded_order(group):
        value = json.dumps([seed, [record_key(row) for row in group]], separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()

    available.sort(key=seeded_order)
    # Reserve one group for each split; apportion remaining groups by ratios.
    quota = [(len(available) - 3) * ratio for ratio in ratios]
    counts = [1 + math.floor(value) for value in quota]
    remainder = len(available) - sum(counts)
    order = sorted(range(3), key=lambda i: (-(quota[i] % 1), i))
    for i in order[:remainder]:
        counts[i] += 1
    start = 0
    for name, count in zip(SPLIT_NAMES, counts):
        result[name] = [row for group in available[start:start + count] for row in group]
        start += count
    for rows_in_split in result.values():
        rows_in_split.sort(key=record_key)
    validate_splits(result, expected=rows)
    return result


def write_splits(directory, splits):
    validate_splits(splits)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name in SPLIT_NAMES:
        write_manifest(directory / f"{name}.jsonl", splits[name])


def read_splits(directory):
    directory = Path(directory)
    result = {name: read_manifest(directory / f"{name}.jsonl") for name in SPLIT_NAMES}
    validate_splits(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--manifest", required=True, type=Path, help="Original manifest, for exact coverage validation")
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.command == "generate":
        write_splits(args.directory, split_records(rows, seed=args.seed))
    else:
        validate_splits(read_splits(args.directory), expected=rows)
    print("Split coverage and leakage validation passed")


if __name__ == "__main__":
    main()
