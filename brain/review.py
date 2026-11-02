"""A second opinion on the pilot camera from Muse Wonder, independent of SAM 3D Body.

The pilot watch decides live from SAM 3D Body's pose of every frame and a measured threshold.
After the run, Muse Wonder (Meta's reasoning model that reads video) watches the whole recording,
camera.mp4, without seeing the poses, the thresholds or what Crosscheck said, and answers the
same question in its own terms. Two different models, two different methods: when they agree it
is evidence; when they disagree it is the first thing a reviewer should look at.

  uv run python cli.py review [replay/pilot_* ...]     # writes <recording>/review.json
"""
import base64
import json
import os
import pathlib
import re
import time

import httpx
from dotenv import load_dotenv

MODEL = "muse-wonder-1.0"
QUESTION = (
    "This video is from a camera on the glareshield of an airliner flight deck, looking at the captain in the left "
    "seat. It is {seconds:.0f} seconds long. Watch all of it. Does the captain become incapacitated (unconscious, "
    "asleep or unresponsive: slumped, eyes closed, hands off the controls) and stay that way, or recover, or stay "
    "normal throughout (reading a checklist or looking down is normal)? Reply with JSON only: "
    '{{"state": "normal" | "incapacitated" | "recovered", "from_s": <seconds when it began, or null>, '
    '"evidence": "<one sentence on what you see>"}}'
)


def review(video: pathlib.Path, seconds: float) -> dict:
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
    data = base64.b64encode(video.read_bytes()).decode()
    body = {"model": MODEL, "reasoning_effort": "low", "max_completion_tokens": 4000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": QUESTION.format(seconds=seconds)},
                                                      {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + data}}]}]}
    t = time.time()
    r = httpx.post(os.environ.get("META_API_BASE", "https://api.meta.ai/v1") + "/chat/completions", json=body, timeout=300,
                   headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}, trust_env=False)
    r.raise_for_status()
    d = r.json()
    text = d["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        answer = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        answer = {}
    return {"model": MODEL, "state": answer.get("state"), "from_s": answer.get("from_s"), "evidence": answer.get("evidence"),
            "s": round(time.time() - t, 1), "usage": d.get("usage", {}), "raw": text if not answer else None}


def review_recording(rec: pathlib.Path) -> dict | None:
    meta = json.loads((rec / "meta.json").read_text())
    if not meta.get("video") or not (rec / meta["video"]).exists():
        return None
    import subprocess
    seconds = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(rec / meta["video"])], capture_output=True, text=True).stdout.strip() or 0)
    r = review(rec / meta["video"], seconds)
    truth = {"slumped": "incapacitated", "asleep": "incapacitated", "checklist": "normal", "answers": "recovered"}.get(meta.get("variant"))
    r["expected_state"] = truth
    r["agrees_with_truth"] = r["state"] == truth
    if r["from_s"] is not None and meta.get("t_change") and meta.get("t_start"):
        r["change_s"] = round(meta["t_change"] - meta["t_start"], 1)
    (rec / "review.json").write_text(json.dumps(r, indent=1))
    return r


def run(recs: list[pathlib.Path]) -> None:
    for rec in recs:
        r = review_recording(rec)
        if r:
            print(f"{rec.name}: {r['state']} from {r['from_s']} s (expected {r['expected_state']}; "
                  f"{'agrees' if r['agrees_with_truth'] else 'DISAGREES'}) {r['s']} s\n  {r['evidence']}")
