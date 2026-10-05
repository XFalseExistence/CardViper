"""Offline intake of real BACK photographs into the existing, unapproved capture schema."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from PIL import Image

from .import_back_captures import MODES, SOURCE_ID, audit_back_source
from .prepare_back_capture_manifest import IMAGE_SUFFIXES

CAPTURE_FIELDS = {"capture_date", "device", "session_id", "scene_id", "deck_design",
                  "annotation_provenance", "capture_mode", "full_image_back", "boxes", "held_out"}


def _photos(root, *, inbox_only=True):
    directory = root / "inbox" if inbox_only else root
    if not directory.is_dir():
        return []
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Intake cannot follow symlinks")
    return [path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(value):
    data = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else (value or {})
    if not isinstance(data, dict) or set(data) - {"common", "images"}:
        raise ValueError("Metadata needs only common and images objects")
    common, images = data.get("common", {}), data.get("images", {})
    if not isinstance(common, dict) or not isinstance(images, dict) or set(common) - (CAPTURE_FIELDS | {"owner", "rights"}):
        raise ValueError("Invalid common metadata")
    for path, fields in images.items():
        if not isinstance(path, str) or not path.startswith("inbox/") or not isinstance(fields, dict) or set(fields) - CAPTURE_FIELDS:
            raise ValueError(f"Invalid image metadata: {path}")
    if "rights" in common and (not isinstance(common["rights"], dict) or
                               set(common["rights"]) - {"classifier", "detector", "evidence", "detector_evidence"}):
        raise ValueError("Invalid rights metadata")
    return common, images


def _fill(target, updates, label):
    for key, value in updates.items():
        previous = target.get(key)
        if previous == value:
            continue
        if previous not in (None, "", False, []):
            raise ValueError(f"Metadata conflict for {label}.{key}; review captures.json explicitly")
        target[key] = value


def _atomic_json(path, data):
    descriptor, temporary = tempfile.mkstemp(prefix=".captures-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def intake(root, metadata=None):
    """Record measured facts and explicitly supplied metadata; never approve a batch."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("BACK root must exist; no source was created")
    photos = _photos(root)
    if not photos:
        return validate(root)
    common, images = _metadata(metadata)
    discovered = {path.relative_to(root).as_posix() for path in photos}
    if set(images) - discovered:
        raise ValueError("Metadata references an image not found in inbox")
    target = root / "captures.json"
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("source_id") != SOURCE_ID or not isinstance(data.get("captures"), list):
            raise ValueError("Existing manifest has another schema or source ID")
    else:
        data = {"schema_version": 1, "source_id": SOURCE_ID, "draft_status": "NOT READY",
                "owner": "", "rights": {"classifier": False, "detector": False, "evidence": ""},
                "captures": [], "review": {"approved": False, "session_independence_confirmed": False,
                                            "near_duplicate_clues_reviewed": False, "reviewer": "", "notes": ""}}
    if not isinstance(data.get("review"), dict) or not isinstance(data.get("rights"), dict):
        raise ValueError("Existing review and rights must be objects")
    if data["review"].get("approved") is True:
        raise ValueError("Approved manifest is immutable through intake")
    if "owner" in common:
        _fill(data, {"owner": common["owner"]}, "source")
    if "rights" in common:
        _fill(data["rights"], common["rights"], "rights")
    existing = {}
    for entry in data["captures"]:
        if not isinstance(entry, dict) or entry.get("image_path") in existing:
            raise ValueError("Invalid or duplicate existing image_path")
        existing[entry["image_path"]] = entry
    for photo in photos:
        relative = photo.relative_to(root).as_posix()
        digest = _hash(photo)
        with Image.open(photo) as raster:
            width, height = raster.size
            raster.verify()
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {relative}")
        entry = existing.get(relative)
        if entry is None:
            entry = {"image_path": relative, "image_sha256": digest, "width": width, "height": height,
                     "capture_date": "", "device": "", "session_id": "", "scene_id": "",
                     "deck_design": "", "annotation_provenance": "", "capture_mode": "",
                     "full_image_back": False}
            data["captures"].append(entry)
        elif (entry.get("image_sha256") != digest or
              ("width" in entry and entry["width"] != width) or
              ("height" in entry and entry["height"] != height)):
            raise ValueError(f"Image changed since intake: {relative}")
        supplied = {key: value for key, value in common.items() if key in CAPTURE_FIELDS}
        supplied.update(images.get(relative, {}))
        _fill(entry, supplied, relative)
        if entry.get("capture_mode") == "annotated-scene" and entry.get("full_image_back") is False:
            entry.pop("full_image_back")
    data["captures"].sort(key=lambda entry: entry["image_path"])
    rendered = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if not target.exists() or target.read_text(encoding="utf-8") != rendered:
        _atomic_json(target, data)
    return validate(root)


def validate(root):
    """Report gaps and use the existing audit for the sole READY decision."""
    root = Path(root).resolve()
    result = {"status": "NOT READY", "blocker": "BACK_SOURCE_MISSING", "image_files": 0,
              "manifest_entries": 0, "complete_entries": 0, "incomplete_entries": 0,
              "exact_duplicates": [], "missing_fields": {}, "missing_files": [],
              "hash_mismatches": [], "unmanifested_images": [], "declared_sessions": 0,
              "independent_sessions": 0, "session_independence_confirmed": False,
              "session_count_qualification": "Declared IDs and structural groups do not prove independent captures",
              "deck_designs": 0, "annotation_modes": {}}
    if not root.is_dir():
        return result
    try:
        photos = _photos(root, inbox_only=False)
    except ValueError as error:
        result.update(blocker="BACK_AUDIT_NOT_READY", error=str(error))
        return result
    result["image_files"] = len(photos)
    manifest = root / "captures.json"
    if not manifest.is_file():
        if photos:
            result["blocker"] = "BACK_AUDIT_NOT_READY"
        return result
    result["blocker"] = "BACK_AUDIT_NOT_READY"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or data.get("source_id") != SOURCE_ID or not isinstance(data.get("captures"), list):
            raise ValueError("Expected schema_version 1, source_id cardviper-back, and captures list")
        entries = data["captures"]
        result["manifest_entries"] = len(entries)
        review = data.get("review")
        result["session_independence_confirmed"] = (
            isinstance(review, dict) and review.get("session_independence_confirmed") is True)
        source_missing = []
        if not data.get("owner"):
            source_missing.append("owner")
        rights = data.get("rights")
        if not isinstance(rights, dict) or not rights.get("evidence"):
            source_missing.append("rights.evidence")
        if not isinstance(rights, dict) or type(rights.get("classifier")) is not bool:
            source_missing.append("rights.classifier")
        if not isinstance(rights, dict) or type(rights.get("detector")) is not bool:
            source_missing.append("rights.detector")
        if source_missing:
            result["missing_fields"]["source"] = source_missing
        digests = defaultdict(list)
        sessions, designs, modes, paths = set(), set(), Counter(), set()
        for index, entry in enumerate(entries, 1):
            if not isinstance(entry, dict):
                result["missing_fields"][f"entry-{index}"] = ["capture object"]
                continue
            relative = entry.get("image_path")
            label = relative if isinstance(relative, str) else f"entry-{index}"
            missing = [field for field in ("capture_date", "device", "session_id", "scene_id",
                       "deck_design", "annotation_provenance", "capture_mode", "image_path", "image_sha256")
                       if not entry.get(field)]
            mode = entry.get("capture_mode")
            if mode == "tight-back-crop" and entry.get("full_image_back") is not True:
                missing.append("full_image_back=true")
            elif mode == "annotated-scene" and not entry.get("boxes"):
                missing.append("boxes")
            elif mode and mode not in MODES:
                missing.append("valid capture_mode")
            if isinstance(relative, str):
                paths.add(relative)
                path = root / relative
                if (path.is_symlink() or not path.resolve().is_relative_to(root) or
                        not path.is_file()):
                    result["missing_files"].append(relative)
                elif _hash(path) != entry.get("image_sha256"):
                    result["hash_mismatches"].append(relative)
                if isinstance(entry.get("image_sha256"), str):
                    digests[entry["image_sha256"]].append(relative)
            if entry.get("session_id"):
                sessions.add(entry["session_id"])
            if entry.get("deck_design"):
                designs.add(entry["deck_design"])
            if mode:
                modes[mode] += 1
            if missing:
                result["missing_fields"][label] = missing
            if not missing and label not in result["missing_files"] and label not in result["hash_mismatches"]:
                result["complete_entries"] += 1
        result["incomplete_entries"] = len(entries) - result["complete_entries"]
        result["exact_duplicates"] = sorted(sorted(group) for group in digests.values() if len(group) > 1)
        result["unmanifested_images"] = sorted(path.relative_to(root).as_posix() for path in photos
                                                if path.relative_to(root).as_posix() not in paths)
        result["declared_sessions"] = len(sessions)
        result["deck_designs"] = len(designs)
        result["annotation_modes"] = dict(sorted(modes.items()))
        if not entries or source_missing or result["incomplete_entries"] or result["missing_files"] or result["hash_mismatches"] or result["unmanifested_images"]:
            return result
        audit = audit_back_source(root)
        result["independent_sessions"] = audit["independent_sessions"]
        result["status"] = audit["status"]
        result["blocker"] = None if audit["status"] == "READY" else "BACK_AUDIT_NOT_READY"
        result["audit_reasons"] = audit["reasons"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("intake", "validate"):
        command = subcommands.add_parser(name)
        command.add_argument("--root", required=True, type=Path)
        if name == "intake":
            command.add_argument("--metadata", type=Path, help="Explicit common/images JSON metadata")
    args = parser.parse_args()
    result = intake(args.root, args.metadata) if args.command == "intake" else validate(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "validate" and result["status"] != "READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
