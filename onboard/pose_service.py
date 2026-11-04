"""The on-board pilot watch: SAM 3D Body (Meta open weights) on this computer, behind a loopback port.

The same model and call as cloud/sam3d_body_app.py on Modal, so eval/scenarios/pilot_watch.py can
pose each frame without a network. Runs in its own environment with Meta's code
(github.com/facebookresearch/sam-3d-body) and the checkpoint downloaded once:

  ~/llm-stack/sam3db/.venv/bin/python onboard/pose_service.py      # 127.0.0.1:8772

POST /pose with {"jpeg": <base64>, "box": [l, t, r, b]} -> the keypoints Modal's Pose.one returns.
"""
import base64
import json
import os
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

CODE = pathlib.Path(os.environ.get("SAM3DB_CODE", pathlib.Path.home() / "llm-stack" / "sam3db" / "sam-3d-body"))
MODEL = pathlib.Path(os.environ.get("SAM3DB_MODEL", pathlib.Path.home() / "models" / "sam-3d-body-vith"))
PORT = int(os.environ.get("POSE_PORT", "8772"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
LOCK = threading.Lock()
EST = {}


def load() -> None:
    sys.path.insert(0, str(CODE))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    model, cfg = load_sam_3d_body(checkpoint_path=str(MODEL / "model.ckpt"), device=DEVICE, mhr_path=str(MODEL / "assets" / "mhr_model.pt"))
    EST["est"] = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    print(f"pose service: SAM 3D Body on {DEVICE}, port {PORT}", flush=True)


def pose(jpeg: bytes, box: list[float]) -> dict:
    import cv2
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    with LOCK, torch.inference_mode():
        people = EST["est"].process_one_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bboxes=np.array([box], dtype=np.float32))
    if not people:
        return {"error": "no person"}
    p = people[0]
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in p.items()
            if k in ("pred_keypoints_2d", "pred_keypoints_3d", "bbox", "focal_length", "pred_cam_t")}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply(200 if EST else 503, {"ok": bool(EST), "device": DEVICE})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._reply(200, pose(base64.b64decode(body["jpeg"]), body["box"]))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    load()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
