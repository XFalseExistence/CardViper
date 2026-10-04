"""Apply complete, source-bound human family decisions without granting training rights."""

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path

from .audit_image_families import filename_family, video_recording_family
from .build_family_review import analyze_review
from .manifest import nonempty

DECISIONS = {"same_family", "independent", "unsure"}


def apply_review(archive, audit_path, family_path, proposal_path, review_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing group proposal: {output}")
    expected = analyze_review(archive, audit_path, family_path, proposal_path)
    review_path = Path(review_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    for key in ("schema_version", "source_id", "archive_sha256", "artifact_fingerprint",
                "artifact_audit_sha256", "family_audit_sha256", "base_proposal_sha256"):
        if review.get(key) != expected[key]:
            raise ValueError(f"Review is stale or source evidence changed: {key}")
    choices = review.get("decisions")
    required = {candidate["candidate_id"] for candidate in expected["candidates"]}
    if (not isinstance(choices, list) or len(choices) != len(required) or
            {entry.get("candidate_id") for entry in choices if isinstance(entry, dict)} != required):
        raise ValueError("Review must contain exactly one decision for every current candidate")
    if any(not isinstance(entry, dict) or entry.get("decision") not in DECISIONS or
           not isinstance(entry.get("notes"), str) for entry in choices):
        raise ValueError("Invalid family review decision")
    if review.get("approved") is not False and review.get("approved") is not True:
        raise ValueError("Human approval must be an explicit boolean")
    if review["approved"]:
        nonempty(review.get("reviewer"), "reviewer")
        nonempty(review.get("method"), "review method")
        try:
            review_date = review["review_date"]
            if date.fromisoformat(review_date).isoformat() != review_date:
                raise ValueError("Invalid review date")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Review date must be YYYY-MM-DD") from error
    base = json.loads(Path(proposal_path).read_text(encoding="utf-8"))
    base_groups = base["groups"]
    parent = {group: group for group in set(base_groups.values())}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    choice_by_id = {item["candidate_id"]: item for item in choices}
    for candidate in expected["candidates"]:
        if choice_by_id[candidate["candidate_id"]]["decision"] == "same_family":
            first = candidate["group_ids"][0]
            for group in candidate["group_ids"][1:]:
                parent[find(group)] = find(first)
    components = defaultdict(list)
    for path, group in base_groups.items():
        components[find(group)].append(path)
    groups = {}
    for members in components.values():
        identifier = "face-" + hashlib.sha256(min(members).encode()).hexdigest()[:16]
        for path in members:
            groups[path] = identifier
    final_by_base = {base_groups[path]: final_group for path, final_group in groups.items()}
    for candidate in expected["candidates"]:
        if choice_by_id[candidate["candidate_id"]]["decision"] == "independent":
            final_ids = [final_by_base[base_group] for base_group in candidate["group_ids"]]
            if len(set(final_ids)) != len(final_ids):
                raise ValueError("Independent decision conflicts with a same-family union")
    # A decision about one bucket never silently resolves a separate bucket.
    unresolved = [candidate["candidate_id"] for candidate in expected["candidates"]
                  if choice_by_id[candidate["candidate_id"]]["decision"] == "unsure" and
                  len({groups[item["path"]] for item in candidate["images"]}) > 1]
    exact = json.loads(Path(audit_path).read_text(encoding="utf-8"))["exact_duplicate_files"]
    if any(len({groups[path] for path in paths}) > 1 for paths in exact):
        raise ValueError("Exact duplicate images cross reviewed groups")
    for key_function in (filename_family, video_recording_family):
        families = defaultdict(set)
        for path, group in groups.items():
            key = key_function(path)
            if key is not None:
                families[key].add(group)
        if any(len(assigned) > 1 for assigned in families.values()):
            raise ValueError("Roboflow or video lineage crosses reviewed groups")
    ready = review["approved"] and not unresolved
    result = {
        "schema_version": 1, "source_id": "playing-cards-seed",
        "artifact_fingerprint": expected["artifact_fingerprint"],
        "groups": dict(sorted(groups.items())),
        "review": {"approved": ready, "similarity_candidates_reviewed": ready,
                   "reviewer": review.get("reviewer", ""), "review_date": review.get("review_date", ""),
                   "method": review.get("method", ""),
                   "candidate_decisions_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest()},
        "status": "READY" if ready else "NOT READY",
        "unresolved_candidate_count": len(unresolved),
        "unresolved_candidate_ids": unresolved,
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"status": result["status"], "output": str(output),
            "group_count": len(set(groups.values())),
            "unresolved_candidate_count": len(unresolved)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--family-audit", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(apply_review(args.archive, args.audit, args.family_audit,
                                  args.proposal, args.review, args.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
