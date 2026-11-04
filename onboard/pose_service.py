"""The on-board pilot watch: SAM 3D Body (Meta open weights) on this computer, behind a loopback port.

The same model and call as cloud/sam3d_body_app.py on Modal, so eval/scenarios/pilot_watch.py can
pose each frame without a network. Runs in its own environment with Meta's code
(github.com/facebookresearch/sam-3d-body, unmodified) and the checkpoint downloaded once
(setup below):

  HF_HUB_OFFLINE=1 ~/llm-stack/sam3db/.venv/bin/python onboard/pose_service.py      # 127.0.0.1:8772

POST /pose with {"jpeg": <base64>, "box": [l, t, r, b]} -> the keypoints Modal's Pose.one returns.
GET /health says whether the model is loaded and warmed up.

Meta's code is written for a CUDA GPU. On a Mac three things are pointed elsewhere, here, without
editing it: its hard-coded "cuda" (batches moved with recursive_to, and a few .cuda() calls) goes to
this computer's GPU (MPS); and the MHR body model (a TorchScript file) runs on the CPU, because it
computes the skeleton in float64, which MPS does not have. Everything else runs on the GPU.

The backbone runs in float32 here (POSE_PRECISION=fp32), not the checkpoint config's bfloat16 that
Modal's T4 used: on the 24 labelled images Modal had posed (the image sets
before the 5 Nov remake, eval/pilot_watch/images_before_remake/), float32 on this GPU gave the cloud's
posture flag on 24 of 24 (wrists within 0.003 torso lengths, neck within 0.2 degrees); bfloat16 on
this GPU gave 23 of 24, one image just at the wrists threshold moving 0.01 across it. bfloat16 was
1.7-1.9 s an image against float32's 2.1-2.3 s, with X-Plane open but not busy (POSE_PRECISION=bf16).
With X-Plane flying on the same GPU, float32 took 4.4-6.7 s an image. The CPU alone (POSE_DEVICE=cpu)
works but took 12-13 s.

Setup (once): the environment is ~/llm-stack/sam3db/.venv (Python 3.11, torch, and the packages
in cloud/sam3d_body_app.py's image; no detectron2, MoGe or SAM 3: the seat box is given and the
default field of view used, as on Modal), Meta's code cloned to ~/llm-stack/sam3db/sam-3d-body,
and the gated checkpoint in ~/models/sam-3d-body-vith (hf download facebook/sam-3d-body-vith
--local-dir ~/models/sam-3d-body-vith).
"""
import base64
import contextlib
import io
import json
import os
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

CODE = pathlib.Path(os.environ.get("SAM3DB_CODE", pathlib.Path.home() / "llm-stack" / "sam3db" / "sam-3d-body"))
MODEL = pathlib.Path(os.environ.get("SAM3DB_MODEL", pathlib.Path.home() / "models" / "sam-3d-body-vith"))
PORT = int(os.environ.get("POSE_PORT", "8772"))
DEVICE = os.environ.get("POSE_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
PRECISION = os.environ.get("POSE_PRECISION", "fp32")  # fp32, or bf16 (the checkpoint's config, as on Modal)
KEYS = ("pred_keypoints_2d", "pred_keypoints_3d", "bbox", "focal_length", "pred_cam_t")  # what Modal's Pose.one returns
LOCK = threading.Lock()  # one GPU: one frame at a time
EST = {}


class OnCpu(torch.nn.Module):
    """Runs a module on the CPU and hands its outputs back on the caller's device."""

    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m.cpu()

    def forward(self, *xs):
        dev = next(x.device for x in xs if x is not None)
        return tuple(o.to(dev) for o in self.m(*[None if x is None else x.cpu() for x in xs]))


def load() -> None:
    t = time.time()
    sys.path.insert(0, str(CODE))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    if DEVICE != "cuda":
        import sam_3d_body.models.meta_arch.sam3d_body as arch
        import sam_3d_body.sam_3d_body_estimator as estimator
        from sam_3d_body.utils import dist
        to = lambda x, target: dist.recursive_to(x, DEVICE if target == "cuda" else target)  # noqa: E731
        estimator.recursive_to = arch.recursive_to = to
        torch.Tensor.cuda = lambda self, *a, **k: self.to(DEVICE)  # this process serves only this model
    if PRECISION == "fp32":
        import sam_3d_body.build_models as build
        get_config = build.get_config

        def fp32(path):
            cfg = get_config(path)
            cfg.defrost()
            cfg.TRAIN.USE_FP16 = False
            cfg.freeze()
            return cfg
        build.get_config = fp32
    model, cfg = load_sam_3d_body(checkpoint_path=str(MODEL / "model.ckpt"), device=DEVICE, mhr_path=str(MODEL / "assets" / "mhr_model.pt"))
    if DEVICE == "mps":
        for head in (model.head_pose, model.head_pose_hand):
            head.mhr = OnCpu(head.mhr)
    EST["est"] = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    loaded = time.time() - t
    blank = np.zeros((800, 800, 3), np.uint8)  # the first frame compiles the GPU kernels (about 6 s): do it before /health says ok
    _run(cv_encode(blank), [100.0, 100.0, 700.0, 700.0])
    print(f"pose service: SAM 3D Body on {DEVICE} ({cfg.TRAIN.USE_FP16 and cfg.TRAIN.FP16_TYPE or 'float32'}), loaded in {loaded:.0f} s, warmed up in {time.time() - t - loaded:.0f} s, "
          f"port {PORT}", flush=True)
    EST["ready"] = True


def cv_encode(rgb: np.ndarray) -> bytes:
    import cv2
    return cv2.imencode(".jpg", rgb)[1].tobytes()


def _run(jpeg: bytes, box: list[float]) -> list:
    import cv2
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("not a JPEG")
    with LOCK, torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):  # it prints a line per image
        return EST["est"].process_one_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bboxes=np.array([box], dtype=np.float32))


def pose(jpeg: bytes, box: list[float]) -> dict:
    t = time.time()
    people = _run(jpeg, box)
    print(f"pose {time.time() - t:.2f} s", flush=True)
    if not people:
        return {"error": "no person"}
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in people[0].items() if k in KEYS}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        ready = bool(EST.get("ready"))
        self._reply(200 if ready else 503, {"ok": ready, "device": DEVICE, "precision": PRECISION, "model": MODEL.name})

    def do_POST(self):
        if self.path != "/pose":
            return self._reply(404, {"error": "POST /pose"})
        if not EST.get("ready"):
            return self._reply(503, {"error": "loading"})
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._reply(200, pose(base64.b64decode(body["jpeg"]), [float(v) for v in body["box"]]))
        except Exception as e:  # a bad frame is an error for that frame, not for the service
            self._reply(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)  # bind first: /health answers 503 while loading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    load()
    threading.Event().wait()
