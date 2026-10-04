"""Build an offline, source-bound Joshua family review pack; never approve it."""

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import statistics
import zipfile

from PIL import Image

from .audit_image_families import filename_family, video_recording_family
from .yolo_annotations import parse_yolo_row


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inputs(archive, audit_path, family_path, proposal_path):
    paths = [Path(item) for item in (audit_path, family_path, proposal_path)]
    audit, family, proposal = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    archive_sha = _sha256(archive)
    if audit.get("artifact_status") != "SOURCE_ARTIFACT_VALID" or audit.get("archive_sha256") != archive_sha:
        raise ValueError("Source archive changed or artifact audit is not valid")
    if (proposal.get("schema_version") != 1 or proposal.get("source_id") != "playing-cards-seed" or
            not isinstance(proposal.get("groups"), dict) or
            not isinstance(family.get("unresolved_similarity_candidates"), list)):
        raise ValueError("Invalid Joshua family evidence")
    for candidate in family["unresolved_similarity_candidates"]:
        if not isinstance(candidate, list) or len(candidate) < 2 or any(path not in proposal["groups"] for path in candidate):
            raise ValueError("Family candidate differs from proposal image set")
    return audit, family, proposal, archive_sha, {
        "artifact_audit_sha256": _sha256(paths[0]),
        "family_audit_sha256": _sha256(paths[1]),
        "base_proposal_sha256": _sha256(paths[2]),
    }


def _label_path(image_path):
    path = PurePosixPath(image_path)
    if path.is_absolute() or ".." in path.parts or "/images/" not in f"/{image_path}" or "\\" in image_path:
        raise ValueError(f"Unsafe export image path: {image_path}")
    return image_path.replace("/images/", "/labels/", 1).rsplit(".", 1)[0] + ".txt"


def _dhash(raster):
    reduced = raster.convert("L").resize((9, 8))
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (reduced.getpixel((x, y)) > reduced.getpixel((x + 1, y)))
    return value


def _image_evidence(packed, path, group, thumbnails=None):
    raw = packed.read(path)
    sha = hashlib.sha256(raw).hexdigest()
    annotation = packed.read(_label_path(path))
    objects = [parse_yolo_row(line, 52) for line in annotation.decode("utf-8").splitlines() if line.strip()]
    with Image.open(BytesIO(raw)) as raster:
        width, height = raster.size
        dhash = _dhash(raster)
        if thumbnails is not None:
            thumbnail = thumbnails / f"{sha}.jpg"
            if not thumbnail.exists():
                small = raster.convert("RGB")
                small.thumbnail((320, 240))
                small.save(thumbnail, "JPEG", quality=78)
    return {
        "path": path, "sha256": sha,
        "annotation_sha256": hashlib.sha256(annotation).hexdigest(),
        "split": path.split("/", 1)[0], "group_id": group,
        "lineage_key": filename_family(path), "video_key": video_recording_family(path),
        "width": width, "height": height, "object_count": len(objects),
        "labels": sorted({obj.class_id for obj in objects}),
        "geometry": sorted({(obj.class_id, *(round(value, 2) for value in obj.bbox)) for obj in objects}),
        "dhash": f"{dhash:016x}", "thumbnail": f"images/{sha}.jpg",
    }


def _jaccard(left, right):
    a, b = set(map(tuple, left)), set(map(tuple, right))
    return len(a & b) / len(a | b) if a or b else 1.0


def _candidate_evidence(images, artifact_fingerprint):
    ordered = sorted(images, key=lambda item: item["path"])
    source_bindings = [(item["path"], item["sha256"], item["annotation_sha256"]) for item in ordered]
    candidate_id = hashlib.sha256(json.dumps([artifact_fingerprint, source_bindings],
                                             separators=(",", ":")).encode()).hexdigest()
    groups = sorted({item["group_id"] for item in images})
    closest = None
    max_geometry = 0.0
    distances = []
    proven = False
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if left["group_id"] == right["group_id"]:
                continue
            distance = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
            geometry = _jaccard(left["geometry"], right["geometry"])
            distances.append(distance)
            max_geometry = max(max_geometry, geometry)
            if (left["sha256"] == right["sha256"] or left["lineage_key"] == right["lineage_key"] or
                    left["video_key"] is not None and left["video_key"] == right["video_key"]):
                proven = True
            score = (distance, -geometry, left["path"], right["path"])
            if closest is None or score < closest[0]:
                closest = (score, left, right, geometry)
    if not distances:
        raise ValueError("Candidate has no cross-family image pair")
    best_left, best_right = closest[1:3]
    min_distance = closest[0][0]
    if proven:
        classification = "PROVEN_SAME_FAMILY"
    elif len(groups) == 2 and min_distance <= 4 and closest[3] >= .8:
        classification = "STRONG_SAME_FAMILY_CANDIDATE"
    elif min_distance >= 24 and max_geometry <= .2:
        classification = "STRONG_INDEPENDENT_CANDIDATE"
    else:
        classification = "AMBIGUOUS"
    return {
        "candidate_id": candidate_id, "classification": classification,
        "group_ids": groups, "group_count": len(groups), "images": ordered,
        "cross_group_pair_count": len(distances),
        "min_dhash_distance": min_distance,
        "median_dhash_distance": statistics.median(distances),
        "max_geometry_jaccard": round(max_geometry, 4),
        "closest_pair": {
            "paths": [best_left["path"], best_right["path"]],
            "thumbnails": [best_left["thumbnail"], best_right["thumbnail"]],
            "dhash_distance": min_distance,
            "geometry_jaccard": round(closest[3], 4),
            "label_overlap": round(_jaccard([(v,) for v in best_left["labels"]],
                                             [(v,) for v in best_right["labels"]]), 4),
            "same_exact_hash": best_left["sha256"] == best_right["sha256"],
            "same_lineage": best_left["lineage_key"] == best_right["lineage_key"],
            "same_video": best_left["video_key"] is not None and best_left["video_key"] == best_right["video_key"],
        },
        "evidence_summary": "Perceptual distance and rounded-box geometry are clues, not provenance proof.",
    }


def _merge_all_statistics(proposal, candidates, packed):
    groups = proposal["groups"]
    parent = {group: group for group in set(groups.values())}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    for candidate in candidates:
        first = candidate["group_ids"][0]
        for group in candidate["group_ids"][1:]:
            parent[find(group)] = find(first)
    components = defaultdict(list)
    classes = defaultdict(set)
    for path, group in groups.items():
        root = find(group)
        components[root].append(path)
        labels = packed.read(_label_path(path)).decode("utf-8")
        classes[root].update(int(line.split()[0]) for line in labels.splitlines() if line.strip())
    sizes = sorted((len(paths) for paths in components.values()), reverse=True)
    class_group_counts = Counter(label for labels in classes.values() for label in labels)
    largest = sorted(components, key=lambda key: (-len(components[key]), key))[:10]
    giant = sizes[0] > max(1000, len(groups) // 10)
    return {
        "resulting_group_count": len(components), "largest_component_size": sizes[0],
        "median_component_size": statistics.median(sizes),
        "largest_component_class_counts": [len(classes[key]) for key in largest],
        "pathological_giant_component": giant,
        "all_52_faces_in_three_groups": all(class_group_counts[label] >= 3 for label in range(52)),
        "full_53_class_feasibility": "UNVERIFIED: BACK absent; grouped split and coverage not run",
        "merge_all_option_safe": not giant and all(class_group_counts[label] >= 3 for label in range(52)),
    }


def analyze_review(archive, audit_path, family_path, proposal_path, *, thumbnails=None):
    audit, family, proposal, archive_sha, report_hashes = _inputs(
        archive, audit_path, family_path, proposal_path)
    candidates_paths = [sorted(paths) for paths in family["unresolved_similarity_candidates"]
                        if len({proposal["groups"][path] for path in paths}) > 1]
    needed = sorted({path for paths in candidates_paths for path in paths})
    if thumbnails is not None:
        thumbnails.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as packed:
        evidence = {path: _image_evidence(packed, path, proposal["groups"][path], thumbnails)
                    for path in needed}
        candidates = [_candidate_evidence([evidence[path] for path in paths],
                                          proposal["artifact_fingerprint"])
                      for paths in candidates_paths]
        merge_all = _merge_all_statistics(proposal, candidates, packed)
    candidates.sort(key=lambda item: item["candidate_id"])
    return {
        "schema_version": 1, "source_id": "playing-cards-seed",
        "archive_sha256": archive_sha,
        "artifact_fingerprint": proposal["artifact_fingerprint"],
        **report_hashes,
        "candidate_count": len(candidates),
        "classification_counts": dict(sorted(Counter(item["classification"] for item in candidates).items())),
        "merge_all": merge_all,
        "candidates": candidates,
    }


def _render_html(pack):
    embedded = json.dumps(pack, separators=(",", ":")).replace("<", "\\u003c")
    return """<!doctype html><html lang="en"><meta charset="utf-8"><title>CardViper Joshua family review</title>
<style>body{font:16px system-ui;background:#111827;color:#f3f4f6;margin:2rem;max-width:1300px}
button,input{font:inherit;margin:.25rem;padding:.45rem}button{cursor:pointer}.row{display:flex;gap:1rem;flex-wrap:wrap}
.card{background:#1f2937;padding:1rem;border-radius:.5rem;flex:1;min-width:260px}img{max-width:280px;max-height:210px}
.gallery{display:flex;gap:.5rem;overflow-x:auto}.thumb{min-width:170px;font-size:.75rem;overflow-wrap:anywhere}
.thumb img{max-width:160px;max-height:120px}small{color:#cbd5e1}label{display:block}</style>
<h1>Joshua family review</h1><p>Offline pack. No network or automatic approval. Inspect every image in a bucket before asserting all its groups are independent.</p>
<div class="row"><label>Reviewer <input id="reviewer"></label><label>Review date <input id="date" type="date"></label>
<label>Method <input id="method" size="42" placeholder="How provenance was checked"></label></div>
<label><input type="checkbox" id="approved">I explicitly approve these decisions after reviewing the source images and provenance.</label>
<p id="progress"></p><label>Filter <select id="filter"><option value="all">all</option><option value="strong-same">strong-same</option>
<option value="ambiguous">ambiguous</option><option value="strong-independent">strong-independent</option>
<option value="undecided">undecided</option></select></label>
<button id="previous">Previous</button><span id="position"></span><button id="next">Next</button>
<button id="next-unresolved">Next unresolved</button>
<h2 id="title"></h2><p id="recommendation"></p><p id="evidence"></p><div class="row" id="pair"></div>
<h3>All images in this candidate bucket, grouped by current family</h3><div id="groups"></div>
<p><button data-decision="same_family">S: SAME FAMILY</button><button data-decision="independent">I: INDEPENDENT</button>
<button data-decision="unsure">U: UNSURE</button><button id="accept-recommendation">Accept shown recommendation</button></p>
<p id="saved-decision"></p><label>Decision notes <input id="notes" size="75"></label>
<p><button id="download">Download decisions JSON</button><label>Resume from decisions JSON <input id="load" type="file" accept="application/json,.json"></label>
<small>Keep the exported JSON beside the pack; apply it with cardviper-apply-family-review.</small></p>
<script>const pack = """ + embedded + """;
let at=0;const choices=Object.fromEntries(pack.candidates.map(c=>[c.candidate_id,{candidate_id:c.candidate_id,decision:'unsure',notes:''}]));
const q=id=>document.getElementById(id);function node(tag,text){const x=document.createElement(tag);x.textContent=text;return x}
function image(src){const x=document.createElement('img');x.src=src;x.loading='lazy';return x}
const recommendations={STRONG_SAME_FAMILY_CANDIDATE:'same_family',STRONG_INDEPENDENT_CANDIDATE:'independent'};
function unresolvedCount(){return pack.candidates.filter(c=>choices[c.candidate_id].decision==='unsure').length}
function matches(c){const f=q('filter').value;return f==='all'||
(f==='undecided'&&choices[c.candidate_id].decision==='unsure')||
(f==='strong-same'&&c.classification==='STRONG_SAME_FAMILY_CANDIDATE')||
(f==='ambiguous'&&c.classification==='AMBIGUOUS')||
(f==='strong-independent'&&c.classification==='STRONG_INDEPENDENT_CANDIDATE')}
function show(){const c=pack.candidates[at];if(!c)return;q('position').textContent=(at+1)+' / '+pack.candidates.length;
const remaining=unresolvedCount();q('progress').textContent=(pack.candidates.length-remaining)+' reviewed / '+pack.candidates.length+'; '+remaining+' unresolved';
q('title').textContent=c.classification+' • '+c.group_count+' groups • '+c.images.length+' images';
const suggestion=recommendations[c.classification];q('recommendation').textContent='RECOMMENDATION: '+
(suggestion==='same_family'?'SAME FAMILY':suggestion==='independent'?'INDEPENDENT':'NO RECOMMENDATION')+
'. Machine suggestion only; you must inspect the images and choose explicitly.';
q('accept-recommendation').disabled=!suggestion;
q('evidence').textContent='Closest dHash '+c.min_dhash_distance+'; median '+c.median_dhash_distance+
'; max rounded-box geometry overlap '+c.max_geometry_jaccard+'. Same / independent decisions apply to every cross-group relationship shown in this bucket.';
q('pair').replaceChildren();for(let j=0;j<2;j++){let x=node('div',c.closest_pair.paths[j]);x.className='card';x.prepend(image(c.closest_pair.thumbnails[j]));q('pair').append(x)}
q('groups').replaceChildren();const groups={};for(const v of c.images)(groups[v.group_id]??=[]).push(v);
for(const [id,items] of Object.entries(groups)){let box=node('div','Family '+id+' ('+items.length+')');box.className='card';let gallery=document.createElement('div');gallery.className='gallery';
for(const v of items){let t=node('div',v.path+' | split '+v.split+' | lineage '+v.lineage_key+' | video '+(v.video_key||'none')+' | labels '+v.labels.join(',')+' | objects '+v.object_count+' | '+v.width+'×'+v.height);t.className='thumb';t.prepend(image(v.thumbnail));gallery.append(t)}box.append(gallery);q('groups').append(box)}
q('notes').value=choices[c.candidate_id].notes;q('saved-decision').textContent='Saved decision: '+choices[c.candidate_id].decision.toUpperCase();
document.querySelectorAll('[data-decision]').forEach(b=>{const selected=choices[c.candidate_id].decision===b.dataset.decision;
b.style.fontWeight=selected?'bold':'normal';b.setAttribute('aria-pressed',String(selected))})}
function saveNotes(){choices[pack.candidates[at].candidate_id].notes=q('notes').value}
function move(direction){saveNotes();for(let n=1;n<=pack.candidates.length;n++){const i=(at+direction*n+pack.candidates.length)%pack.candidates.length;
if(matches(pack.candidates[i])){at=i;show();return}}alert('No candidates match this filter.')}
q('previous').onclick=()=>move(-1);q('next').onclick=()=>move(1);q('filter').onchange=()=>{if(!matches(pack.candidates[at]))move(1);else show()};
q('next-unresolved').onclick=()=>{saveNotes();for(let n=1;n<=pack.candidates.length;n++){const i=(at+n)%pack.candidates.length;
if(choices[pack.candidates[i].candidate_id].decision==='unsure'){at=i;q('filter').value='all';show();return}}alert('No unresolved candidates remain.')};
function decide(value){saveNotes();choices[pack.candidates[at].candidate_id].decision=value;show()}
document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=()=>decide(b.dataset.decision));
q('accept-recommendation').onclick=()=>{const choice=recommendations[pack.candidates[at].classification];if(choice)decide(choice)};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;
const d={s:'same_family',i:'independent',u:'unsure'}[e.key.toLowerCase()];if(d)decide(d)});
q('approved').onchange=()=>{if(q('approved').checked&&unresolvedCount()){q('approved').checked=false;
alert('Completion warning: resolve every unsure decision before approval.')}};
q('download').onclick=()=>{saveNotes();if(q('approved').checked&&unresolvedCount()){
alert('Cannot export approved=true while unresolved decisions remain.');return}const out={schema_version:1,source_id:pack.source_id,archive_sha256:pack.archive_sha256,
artifact_fingerprint:pack.artifact_fingerprint,artifact_audit_sha256:pack.artifact_audit_sha256,
family_audit_sha256:pack.family_audit_sha256,base_proposal_sha256:pack.base_proposal_sha256,
reviewer:q('reviewer').value,review_date:q('date').value,method:q('method').value,approved:q('approved').checked,
decisions:pack.candidates.map(c=>choices[c.candidate_id])};const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)+'\\n'],{type:'application/json'}));a.download='review-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
q('load').onchange=async()=>{try{const raw=JSON.parse(await q('load').files[0].text());
for(const key of ['archive_sha256','artifact_fingerprint','artifact_audit_sha256','family_audit_sha256','base_proposal_sha256'])
if(raw[key]!==pack[key])throw Error('Stale or mismatched '+key);
if(!Array.isArray(raw.decisions)||raw.decisions.length!==pack.candidates.length)throw Error('Incomplete decisions');
const ids=new Set(pack.candidates.map(c=>c.candidate_id));const seen=new Set();
for(const d of raw.decisions){if(!ids.has(d.candidate_id)||seen.has(d.candidate_id)||!['same_family','independent','unsure'].includes(d.decision))
throw Error('Invalid candidate decision');seen.add(d.candidate_id)}
for(const d of raw.decisions)choices[d.candidate_id]={candidate_id:d.candidate_id,decision:d.decision,notes:d.notes||''};
q('reviewer').value=raw.reviewer||'';q('date').value=raw.review_date||'';q('method').value=raw.method||'';
q('approved').checked=raw.approved===true&&unresolvedCount()===0;show()}
catch(error){alert('Cannot load decisions: '+error.message)}};show();</script></html>"""


def build_review_pack(archive, audit_path, family_path, proposal_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing review pack: {output}")
    output.mkdir(parents=True)
    try:
        pack = analyze_review(archive, audit_path, family_path, proposal_path,
                              thumbnails=output / "images")
        review = {**pack, "reviewer": "", "review_date": "", "method": "", "approved": False,
                  "decisions": [{"candidate_id": item["candidate_id"], "decision": "unsure", "notes": ""}
                                for item in pack["candidates"]]}
        (output / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output / "index.html").write_text(_render_html(pack), encoding="utf-8")
        return {"status": "REVIEW REQUIRED", "output": str(output),
                "candidate_count": pack["candidate_count"],
                "classification_counts": pack["classification_counts"],
                "merge_all": pack["merge_all"]}
    except BaseException:
        # Do not leave a pack that looks complete after an interrupted build.
        (output / "INCOMPLETE").write_text("Review pack build failed; discard this directory.\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--family-audit", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_review_pack(args.archive, args.audit, args.family_audit,
                                       args.proposal, args.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
