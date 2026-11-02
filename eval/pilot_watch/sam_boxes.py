"""The pilot watch with SAM 3.1 finding the captain, instead of a hand-drawn seat box per image set.

For every labelled image (eval/pilot_watch/images/<set>/<state>.jpg): SAM 3.1 finds the person,
SAM 3D Body poses them inside that box (Modal), and senses/pilot_pose.py says slumped or not;
scored against the state each image was made to show. Writes eval/results/pilot_sam_boxes.json.

  uv run --group eval python eval/pilot_watch/sam_boxes.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from senses.pilot_pose import posture  # noqa: E402
from senses.sam31 import pilot_box  # noqa: E402

IMAGES = ROOT / "eval" / "pilot_watch" / "images"
SLUMPED = {"upright": False, "looking_down": False, "slumped": True, "slumped_side": True}


def main() -> int:
    import modal
    todo, boxes = {}, {}
    for p in sorted(IMAGES.glob("*/*.jpg")):
        if p.stem not in SLUMPED:
            continue  # the halfway photos are keyframes for the video, not labelled test images
        data = p.read_bytes()
        boxes[p] = pilot_box(data)
        todo[f"{p.parent.name}/{p.name}"] = (data, list(boxes[p]))
    res = modal.Cls.from_name("crosscheck-sam3d-body", "Pose")().many.remote(todo)
    rows, right = [], 0
    for p, box in boxes.items():
        r = res[f"{p.parent.name}/{p.name}"]
        got = posture(r["pred_keypoints_3d"]) if "pred_keypoints_3d" in r else None
        ok = got is not None and got["slumped"] == SLUMPED[p.stem]
        right += ok
        rows.append({"image": f"{p.parent.name}/{p.name}", "sam31_box": box, "expected_slumped": SLUMPED[p.stem],
                     "posture": got, "right": ok})
        print(f"{p.parent.name}/{p.stem:13s} box {box}  {got and (got['neck_flex'], got['wrists'], got['slumped'])}  {'ok' if ok else 'WRONG'}")
    out = {"right": right, "of": len(rows), "rows": rows}
    (ROOT / "eval" / "results" / "pilot_sam_boxes.json").write_text(json.dumps(out, indent=1))
    print(f"{right}/{len(rows)} right with SAM 3.1 boxes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
