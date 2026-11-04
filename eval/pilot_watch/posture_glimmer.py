"""Can the local Muse Glimmer judge the captain's posture by itself, without SAM 3D Body?

The 24 labelled images (6 synthetic pilots x upright, looking down, slumped, slumped sideways),
cropped to each camera's seat box, one question each. The answer decides whether the on-board
pilot watch can do without SAM 3D Body (it cannot: onboard/README.md).

Run (Ollama with muse-glimmer:30b up):  uv run python eval/pilot_watch/posture_glimmer.py
"""
import asyncio
import json
import pathlib
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from senses.onboard import glimmer_json, local_client  # noqa: E402
from senses.panel_glimmer import jpeg_b64  # noqa: E402

D = ROOT / "eval" / "pilot_watch"
TRUTH = {"upright": "alert", "looking_down": "alert", "slumped": "incapacitated", "slumped_side": "incapacitated"}
PROMPT = ("A cockpit camera image of an airline captain in the left seat. Judge only his posture. Is he "
          "\"alert\" (upright or looking down at something, head up or deliberately looking down, hands active), or "
          "\"incapacitated\" (slumped: head dropped onto the chest or leaning against the side window, eyes closed, hands limp)? "
          'JSON only: {"state": "alert" | "incapacitated"}')
SCHEMA = {"type": "object", "properties": {"state": {"type": "string", "enum": ["alert", "incapacitated"]}}, "required": ["state"]}


async def main() -> int:
    boxes = json.loads((D / "boxes.json").read_text())
    rows = []
    async with local_client(300) as c:
        for s in sorted(p.name for p in (D / "images").iterdir() if p.is_dir()):
            for state, want in TRUTH.items():
                img = Image.open(D / "images" / s / f"{state}.jpg")
                crop = img.crop(tuple(boxes[s])) if s in boxes else img
                crop.thumbnail((768, 768))
                text, _ = await glimmer_json(PROMPT, SCHEMA, jpeg_b64(crop), num_predict=20, client=c)
                got = json.loads(text).get("state")
                rows.append({"image": f"{s}/{state}", "truth": want, "glimmer": got, "right": got == want})
    out = {"right": sum(r["right"] for r in rows), "of": len(rows), "wrong": [r["image"] for r in rows if not r["right"]], "rows": rows}
    (ROOT / "eval" / "results" / "day_one" / "pilot_posture_glimmer.json").write_text(json.dumps(out, indent=1))
    print(out["right"], "of", out["of"], "wrong:", out["wrong"])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
