"""Scenario 4: pilot incapacitation, through the opt-in pilot-watch camera.

The camera's view is a sequence of images of one pilot (generated with Muse Image, labelled by
the state they show, eval/pilot_watch/): upright until T_CHANGE, then the variant's state.
Every frame goes to SAM 3D Body on Modal as it is taken; senses/pilot_pose.py turns the body
keypoints into "slumped or not". A slumped posture alone never alerts: Crosscheck asks first.
  slumped    - set e, slumped sideways, no answer  -> "Captain, are you OK?", then "...take control."
  asleep     - set f, asleep, head rolled sideways  -> the same
  checklist  - set b, reading a checklist           -> silence
  answers    - set d, slumped, answers when asked   -> the question only, no alert
The answer is spoken (macOS voice) and heard through Muse Voice Transcribe, like all crew speech.
No simulator is needed.

--video: the camera sees moving footage instead of still images: clips made from the same photos
with LTX-2.5 (eval/pilot_watch/make_video.py), cut together as the run goes (idle until T_CHANGE,
then the change, then the state held; in "answers" the pilot lifts his head when asked). Every
captured frame goes to SAM 3D Body. The run also writes camera.mp4, exactly what the camera saw.
Generated footage: no real person is filmed.

--webcam: the Mac's own camera instead of the image sets, with you as the captain. A voice tells
you what to do 3 s before T_CHANGE ("slump forward and stay still", "look down at your lap as if
reading a checklist"); the ground truth is that instruction. With --mic you answer Crosscheck
yourself (in the "answers" variant); otherwise the scripted answer is played. Recorded as
pilot_<variant>_webcam. The frames show your face: they stay on this Mac (git-ignored) unless
you choose to publish them.

Needs:  uv run modal deploy cloud/sam3d_body_app.py
Run:    uv run --group eval python eval/scenarios/pilot_watch.py [slumped|asleep|checklist|answers] [--speak] [--video|--webcam] [--mic]
"""
import asyncio
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import soundfile as sf
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.ledger import Fact  # noqa: E402
from brain.live import Crosscheck  # noqa: E402
from brain.rules import RESPONSE_WAIT_S  # noqa: E402
from eval.scoring import score  # noqa: E402
from senses.mic import Mic  # noqa: E402
from senses.pilot_pose import posture  # noqa: E402
from senses.speech import transcribe  # noqa: E402
from voice.crew import SR, say  # noqa: E402

IMAGES = ROOT / "eval" / "pilot_watch" / "images"
VIDEO = ROOT / "eval" / "pilot_watch" / "video"
CHANGE_S = 5.0  # length of the "change" clip
BOXES = json.loads((ROOT / "eval" / "pilot_watch" / "boxes.json").read_text())
VARIANTS = {  # image set, state after the change, what the pilot says when asked
    "slumped": ("e", "slumped", None),
    "asleep": ("f", "slumped_side", None),
    "checklist": ("b", "looking_down", None),
    "answers": ("d", "slumped", "I'm fine, just reaching for my pen."),
}
EXPECT = {"slumped": (True, True), "asleep": (True, True), "checklist": (False, False), "answers": (True, False)}
T_CHANGE, RUN_S = 10, 45
SCALE = 0.5  # frames go to the GPU at 800 x 800
WEBCAM_W = 800  # webcam frames are sent this wide
INSTRUCT_S = 3  # --webcam: the instruction is spoken this long before T_CHANGE
INSTRUCTIONS = {
    "slumped": "Now slump forward, head down, and stay still. Do not answer.",
    "asleep": "Now lean your head against the side, as if asleep, and stay still. Do not answer.",
    "checklist": "Now look down at your lap, as if reading a checklist.",
    "answers": "Now slump forward, head down. When Crosscheck asks, answer it.",
}


def frame_bytes(image_set: str, state: str) -> bytes:
    im = Image.open(IMAGES / image_set / f"{state}.jpg").convert("RGB")
    buf = io.BytesIO()
    im.resize((int(im.width * SCALE), int(im.height * SCALE))).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def play(pcm, sr: int) -> None:
    path = pathlib.Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(path, pcm, sr)
    subprocess.Popen(["afplay", str(path)])


class VideoFeed:
    """The pilot camera playing the generated clips: idle until T_CHANGE, the change, then the state
    held (idle and hold play back and forth, so the footage never jumps); recover() cuts to the
    pilot lifting his head. at(t) is the frame on screen t seconds into the run."""

    def __init__(self, variant: str):
        import cv2
        self.cv2 = cv2
        self.clips = {}
        for p in sorted((VIDEO / variant).glob("*.mp4")):
            cap, frames = cv2.VideoCapture(str(p)), []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 24
            self.clips[p.stem] = frames
        if not {"idle", "change", "hold"} <= set(self.clips):
            raise SystemExit(f"no clips in {VIDEO / variant}: run eval/pilot_watch/make_video.py {variant}")
        self.t_recover = None

    def _pingpong(self, name: str, t: float):
        f = self.clips[name]
        i = int(t * self.fps) % (2 * len(f) - 2)
        return f[i if i < len(f) else 2 * len(f) - 2 - i]

    def recover(self, t: float) -> None:
        if "recover" in self.clips and self.t_recover is None:
            self.t_recover = t

    def at(self, t: float):
        """(BGR frame, what it shows) at t seconds into the run."""
        if self.t_recover is not None and t >= self.t_recover:
            rec = self.clips["recover"]
            i = int((t - self.t_recover) * self.fps)
            if i < len(rec):
                return rec[i], "recover"
            return self._pingpong("idle", t - self.t_recover - len(rec) / self.fps), "idle"
        if t < T_CHANGE:
            return self._pingpong("idle", t), "idle"
        if t < T_CHANGE + CHANGE_S:
            ch = self.clips["change"]
            return ch[min(len(ch) - 1, int((t - T_CHANGE) * self.fps))], "change"
        return self._pingpong("hold", t - T_CHANGE - CHANGE_S), "hold"

    def jpeg(self, t: float) -> bytes:
        return self.cv2.imencode(".jpg", self.at(t)[0], [self.cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    def write(self, path: pathlib.Path, seconds: float) -> None:
        """What the camera saw, as an H.264 file a browser can play."""
        f0 = self.at(0)[0]
        h, w = f0.shape[:2]
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                              "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                              "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        for i in range(int(seconds * self.fps)):
            p.stdin.write(self.at(i / self.fps)[0].tobytes())
        p.stdin.close()
        p.wait()


class Webcam:
    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(0)
        for _ in range(10):  # the first frames are dark while the camera adjusts its exposure
            self.cap.read()

    def frame(self) -> bytes:
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("webcam: no frame (camera permission for this terminal app?)")
        h, w = bgr.shape[:2]
        bgr = self.cv2.resize(bgr, (WEBCAM_W, round(h * WEBCAM_W / w)))
        return self.cv2.imencode(".jpg", bgr, [self.cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


async def pose_onboard(data: bytes, box: list[float]) -> dict:
    """SAM 3D Body on this computer (onboard/pose_service.py), the same call as Modal's Pose.one."""
    import base64
    from senses.onboard import local_client
    async with local_client(60) as c:
        r = await c.post(os.environ.get("POSE_URL", "http://127.0.0.1:8772") + "/pose",
                         json={"jpeg": base64.b64encode(data).decode(), "box": [float(v) for v in box]})
        r.raise_for_status()
        return r.json()


async def main(variant: str, speak: bool, webcam: bool = False, mic: Mic | None = None, video: bool = False) -> int:
    from senses.onboard import ONBOARD
    image_set, state, answer = VARIANTS[variant]
    name = f"pilot_{variant}" + ("_webcam" if webcam else "") + ("_onboard" if ONBOARD else "")
    out = pathlib.Path(os.environ.get("CROSSCHECK_OUT", ROOT / "replay")) / name  # eval/repeat.py sets it
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cam = Webcam() if webcam else None
    feed = VideoFeed(variant) if video else None
    box_from = "hand-drawn seat box (eval/pilot_watch/boxes.json)"
    if feed:
        size = feed.at(0)[0].shape[1]
        frames = {"upright": feed.jpeg(0)}
        found = None
        if not ONBOARD:  # on board the seat box is the camera's installed calibration (no SAM 3.1 weights)
            from senses.sam31 import pilot_box  # SAM 3.1 finds the captain; SAM 3D Body poses inside that box
            found = pilot_box(frames["upright"])
        box = list(found) if found else [v * size / 1600 for v in BOXES[image_set]]
        box_from = "SAM 3.1, on the first frame" if found else box_from
    elif cam:
        first = cam.frame()
        w, h = Image.open(io.BytesIO(first)).size
        box = [0, 0, w, h]  # a laptop camera sees one person, filling the frame
        frames = {"upright": first}
    else:
        frames = {s: frame_bytes(image_set, s) for s in ("upright", state)}
        box = [v * SCALE for v in BOXES[image_set]]
    if ONBOARD:
        pose_one = pose_onboard
    else:
        import modal
        pose_one = modal.Cls.from_name("crosscheck-sam3d-body", "Pose")().one.remote.aio
    await pose_one(frames["upright"], box)  # wake the model (on Modal, the GPU container) before the clock starts
    live_answer = mic is not None and variant == "answers"  # the person at the microphone answers for the captain

    say_cc = mic.speak if mic else None
    if speak and not mic:
        from voice.tts import speak as say_cc
    cc = Crosscheck(out, speak=say_cc, explain=True)
    truth, heard, n = [], [], 0
    answer_task = None

    async def respond_live():
        """The person at the microphone answers."""
        mic.flush()
        line = await mic.next_line(RESPONSE_WAIT_S)
        if line is None:
            return
        if feed:  # the captain on screen lifts his head as the person answers
            feed.recover(time.time() - t0)  # from now: the camera.mp4 must match the frames it captured
        f = f"mic{len(heard)}.wav"
        sf.write(out / f, line["pcm"], SR)
        heard.append({"speaker": "capt", "spoken": None, "audio_file": f, "transcript": line["text"],
                      "s_from_start": round(line["t"] - line["t_start"], 2)})
        cc.event("heard", line["t"], speaker="capt", transcript=line["text"], spoken=None, mic=True)
        cc.said(Fact("said", "pilot.voice", line["text"], line["t"], "capt", source="muse-voice-transcribe"),
                transcript=line["text"])

    async def instruct():
        """--webcam: tell the person what to do; the instruction is the ground truth."""
        await asyncio.sleep(max(0.0, t0 + T_CHANGE - INSTRUCT_S - time.time()))
        text = INSTRUCTIONS[variant]
        print(">>>", text, flush=True)
        audio = say(text, "Daniel")
        if mic:
            await mic.play(audio, SR)
        else:
            play(audio, SR)
        cc.event("action", time.time(), line=f"Instruction to the person: {text}")

    async def respond():
        """The pilot answers 1.5 s after Crosscheck asks."""
        if feed:  # he lifts his head first, then speaks
            feed.recover(time.time() - t0)
            cc.event("action", time.time(), line="The captain lifts his head")
        await asyncio.sleep(1.5)
        audio = say(answer)
        t_start = time.time()
        if speak:
            play(audio, SR)
        r = await transcribe(audio)
        t = time.time()
        heard.append({"speaker": "capt", "spoken": answer, "transcript": r["text"], "s_from_start": round(t - t_start, 2)})
        cc.event("heard", t, speaker="capt", transcript=r["text"], spoken=answer)
        if r["text"].strip():
            cc.said(Fact("said", "pilot.voice", r["text"].strip(), t, "capt", source="muse-voice-transcribe"),
                    transcript=r["text"])

    t0 = time.time()
    instructing = asyncio.create_task(instruct()) if cam else None
    while time.time() - t0 < RUN_S:
        captured = time.time()
        now_state = "upright" if captured - t0 < T_CHANGE else state
        if feed:
            data = feed.jpeg(captured - t0)
            now_state = {"recover": "upright", "idle": "upright"}.get(feed.at(captured - t0)[1], now_state)
        else:
            data = cam.frame() if cam else frames[now_state]
        fname = f"frames/{n:05d}.jpg"
        (out / fname).write_bytes(data)
        n += 1
        res = await pose_one(data, box)
        t = time.time()
        truth.append({"t": captured, "state": now_state})
        if "pred_keypoints_3d" not in res:
            cc.event("pilot_error", t, frame=fname, error=res.get("error"))
            continue
        cc.observe_pilot(posture(res["pred_keypoints_3d"]), t, fname, captured=round(captured, 3))
        new = cc.decide(t)
        if answer and answer_task is None and any(f.key == "challenge" for f in new):
            answer_task = asyncio.create_task(respond_live() if live_answer else respond())
    if answer_task:
        await answer_task
    if instructing:
        await instructing
    if feed:
        feed.write(out / "camera.mp4", time.time() - t0)

    with (out / "truth.jsonl").open("w") as f:
        for row in truth:
            f.write(json.dumps(row) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "scenario": name, "scorer": "pilot_watch", "variant": variant, "image_set": image_set, "state": state,
        "answer": answer, "expect_ask": EXPECT[variant][0], "expect_alert": EXPECT[variant][1],
        "t_start": t0, "t_change": t0 + T_CHANGE, "frame_box": [0, 0, size, size] if feed else [0, 0, 800, 800], "pilot_camera": True,
        "pose": "SAM 3D Body (facebook/sam-3d-body-vith) on this computer (onboard/pose_service.py), one call per frame" if ONBOARD
                else "SAM 3D Body (facebook/sam-3d-body-vith) on Modal T4, one call per frame",
        "pilot_box": [round(v) for v in box], "pilot_box_from": box_from,
        "camera": "the Mac's webcam, a person following spoken instructions" if cam else
                  f"generated video: LTX-2.5 clips anchored on Muse Image set {image_set}" if feed else f"Muse Image set {image_set}",
        "video": "camera.mp4" if feed else None,
        "speech": ("Muse Voice Transcribe, live: the person on the Mac's microphone" if live_answer else
                   "Muse Voice Transcribe, live (macOS voice for the pilot's answer)") if answer else "none",
        "heard": heard, "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    if feed and not ONBOARD:  # Muse Wonder watches the recording, independently of SAM 3D Body (on board: after landing)
        from brain.review import review_recording
        try:
            rv = review_recording(out)
            print("Muse Wonder:", rv["state"], "from", rv["from_s"], "s:", rv["evidence"])
        except Exception as e:  # the review is a second opinion; the run stands without it
            print("Muse Wonder review failed:", e)
    result = score(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


async def _run(variant: str, speak: bool, webcam: bool, video: bool) -> int:
    if "--mic" not in sys.argv:
        return await main(variant, speak, webcam, video=video)
    async with Mic() as mic:
        return await main(variant, speak, webcam, mic, video=video)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(asyncio.run(_run(args[0] if args else "slumped", speak="--speak" in sys.argv, webcam="--webcam" in sys.argv, video="--video" in sys.argv)))
