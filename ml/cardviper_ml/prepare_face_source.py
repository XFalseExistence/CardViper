"""Audit and normalize the locked Joshua v2 face export with reviewed families."""

from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

from .audit_dataset_artifact import SOURCE_ID, audit_artifact
from .audit_image_families import audit_families, collect_rows, filename_family
from .import_roboflow import import_yolo
from .manifest import nonempty
from .splits import unique_records


def _zip_entries(archive):
    with zipfile.ZipFile(archive) as packed:
        entries = packed.infolist()
        if len(entries) > 100000 or sum(item.file_size for item in entries) > 4 * 1024**3:
            raise ValueError("Archive exceeds bounded member count or expanded size")
        seen = set()
        files = set()
        for item in entries:
            name = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            normalized = name.as_posix()
            if (name.is_absolute() or ".." in name.parts or "\\" in item.filename or
                    ":" in item.filename or normalized in ("", ".") or normalized in seen or
                    stat.S_ISLNK(mode) or item.file_size > 512 * 1024**2 or
                    (item.file_size and not item.compress_size) or
                    (item.compress_size and item.file_size > item.compress_size * 1000)):
                raise ValueError(f"Unsafe or oversized archive entry: {item.filename}")
            seen.add(normalized)
            if not item.is_dir():
                files.add(normalized)
        if any(any(parent.as_posix() in files for parent in PurePosixPath(path).parents)
               for path in seen):
            raise ValueError("Archive file and directory paths collide")
    return entries


def _export_base(root):
    yaml_files = sorted(Path(root).rglob("data.yaml"))
    if len(yaml_files) != 1:
        raise ValueError("Expected exactly one data.yaml")
    return yaml_files[0].parent


def _extract(archive, destination):
    _zip_entries(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as packed:
        packed.extractall(destination)
    return _export_base(destination)


def _proposal(rows, artifact):
    paths = [row["path"] for row in rows]
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first = {}
    for index, row in enumerate(rows):
        for clue in (("family", filename_family(row["path"])), ("sha256", row["sha256"])):
            if clue in first:
                parent[find(index)] = find(first[clue])
            else:
                first[clue] = index
    components = defaultdict(list)
    for index, path in enumerate(paths):
        components[find(index)].append(path)
    groups = {}
    for members in components.values():
        stable_id = "face-" + hashlib.sha256(min(members).encode()).hexdigest()[:16]
        for path in members:
            groups[path] = stable_id
    fingerprint_input = {"data_yaml_sha256": artifact["data_yaml_sha256"],
                         "archive_sha256": artifact["archive_sha256"],
                         "images": [{key: row.get(key) for key in
                                     ("path", "sha256", "annotation_sha256")}
                                    for row in sorted(rows, key=lambda item: item["path"])]}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "source_id": SOURCE_ID,
            "artifact_fingerprint": fingerprint,
            "groups": dict(sorted(groups.items())),
            "review": {"approved": False,
                       "note": "Review all filename, exact hash, and similarity clues against source provenance."}}


def _review_groups(path, proposal):
    if path is None:
        raise ValueError("A reviewed family-group map is required")
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("source_id") != SOURCE_ID:
        raise ValueError("Reviewed group map has wrong schema or source ID")
    if value.get("artifact_fingerprint") != proposal["artifact_fingerprint"]:
        raise ValueError("Reviewed group map does not match this export fingerprint")
    groups = value.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(proposal["groups"]):
        raise ValueError("Reviewed group map must cover exactly every export image")
    for scene_id in groups.values():
        nonempty(scene_id, "reviewed scene_id")
    review = value.get("review")
    if (not isinstance(review, dict) or review.get("approved") is not True or
            review.get("similarity_candidates_reviewed") is not True):
        raise ValueError("A human family/similarity review is required")
    nonempty(review.get("reviewer"), "reviewer")
    nonempty(review.get("method"), "review method")
    review_date = review.get("review_date")
    if not isinstance(review_date, str) or date.fromisoformat(review_date).isoformat() != review_date:
        raise ValueError("review_date must be YYYY-MM-DD")
    connected = defaultdict(set)
    for image_path, proposed_group in proposal["groups"].items():
        connected[proposed_group].add(groups[image_path])
    if any(len(values) > 1 for values in connected.values()):
        raise ValueError("A filename or exact-hash family was split across reviewed scene groups")
    return groups, hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_face_source(source_path, workspace_source_path, reviewed_groups_path=None, *,
                        dry_run=False, class_map_path=None):
    """Return (report, B1 ImageRecords or None, durable root or None)."""
    source_path = Path(source_path)
    report = {"schema_version": 1, "source_id": SOURCE_ID, "status": "NOT READY",
              "reasons": [], "artifact_audit": None, "family_audit": None,
              "group_proposal": None, "reviewed_groups_sha256": None,
              "data_yaml_sha256": None}
    class_map = None
    try:
        if class_map_path is not None:
            class_map = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
        if source_path.is_file():
            _zip_entries(source_path)
        artifact = audit_artifact(source_path, class_map=class_map)
        report["artifact_audit"] = artifact
        if artifact["artifact_status"] != "SOURCE_ARTIFACT_VALID":
            raise ValueError("; ".join(artifact["errors"]))
        with tempfile.TemporaryDirectory() as temporary:
            base = _extract(source_path, Path(temporary) / "export") if source_path.is_file() else _export_base(source_path)
            rows = collect_rows(base)
            report["family_audit"] = audit_families(rows)
            proposal = _proposal(rows, artifact)
            report["group_proposal"] = proposal
            report["data_yaml_sha256"] = artifact["data_yaml_sha256"]
            if report["family_audit"]["exact_duplicates"]:
                raise ValueError("Exact duplicate images must be reconciled before splitting")
            groups, groups_sha = _review_groups(reviewed_groups_path, proposal)
            report["reviewed_groups_sha256"] = groups_sha
            names = [artifact["source_class_index_to_name"][str(i)] for i in range(52)]
            import_groups = {path: {"scene_id": groups[path]} for path in proposal["groups"]}
            def normalize(root):
                result = []
                for split in ("train", "valid", "val", "test"):
                    if (root / split / "images").is_dir():
                        result.extend(import_yolo(root, SOURCE_ID, names, import_groups,
                                                  class_map=class_map,
                                                  images=f"{split}/images",
                                                  annotations=f"{split}/labels"))
                unique_records(result)
                return result
            records = normalize(base)
            observed = {record.image_path: record.image_sha256 for record in records}
            if observed != {row["path"]: row["sha256"] for row in rows}:
                raise ValueError("Normalized records differ from audited export images")
            report["normalized_image_count"] = len(records)
            if dry_run:
                report["status"] = "READY"
                return report, None, None
            if source_path.is_file():
                if Path(workspace_source_path).exists():
                    raise ValueError("Destination source directory already exists")
                durable_root = _extract(source_path, workspace_source_path)
                if hashlib.sha256(source_path.read_bytes()).hexdigest() != artifact["archive_sha256"]:
                    raise ValueError("Archive bytes changed after audit")
            else:
                durable_root = base
            if source_path.is_file():
                extracted_rows = collect_rows(durable_root)
                if _proposal(extracted_rows, artifact)["artifact_fingerprint"] != proposal["artifact_fingerprint"]:
                    raise ValueError("Extracted export differs from reviewed artifact")
            report["status"] = "READY"
            return report, records, durable_root.resolve()
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        report["reasons"].append(str(error))
        return report, None, None
