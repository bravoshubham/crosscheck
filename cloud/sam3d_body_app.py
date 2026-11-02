"""SAM 3D Body on Modal (T4): 3D body keypoints for the pilot-watch camera.

The checkpoint (facebook/sam-3d-body-vith, gated) is downloaded on the Mac and uploaded to the
private volume `crosscheck-models` at /sam-3d-body-vith, so no Hugging Face token goes to Modal.
No person detector or field-of-view model: the camera is fixed, so the captain's seat is a
known box, and the default field of view is used.

  uv run modal run cloud/sam3d_body_app.py
writes <image>.pose.json next to each eval/pilot_watch/images/<set>/<state>.jpg (70 keypoints,
2D and 3D), using that set's box from eval/pilot_watch/boxes.json.
  uv run modal deploy cloud/sam3d_body_app.py
deploys Pose for the live scenario (eval/scenarios/pilot_watch.py), which calls it per frame.
"""
import json
import pathlib

import modal

app = modal.App("crosscheck-sam3d-body")
vol = modal.Volume.from_name("crosscheck-models")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1")
    .pip_install("pytorch-lightning", "opencv-python-headless", "yacs", "scikit-image", "einops", "timm", "dill",
                 "pandas", "rich", "hydra-core", "pyrootutils", "roma", "joblib", "loguru", "optree", "fvcore",
                 "pycocotools", "huggingface_hub", "networkx==3.2.1", "numpy<2", "braceexpand", "omegaconf")
    .run_commands("git clone --depth 1 https://github.com/facebookresearch/sam-3d-body /opt/sam3db")
    .env({"PYTHONPATH": "/opt/sam3db"})
)
CKPT = "/models/sam-3d-body-vith/model.ckpt"
MHR = "/models/sam-3d-body-vith/assets/mhr_model.pt"


@app.cls(image=image, gpu="T4", volumes={"/models": vol}, timeout=1200, scaledown_window=300)
class Pose:
    """Loads the model once per container; the live pilot-watch scenario calls it once a second."""

    @modal.enter()
    def load(self):
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        model, cfg = load_sam_3d_body(checkpoint_path=CKPT, device="cuda", mhr_path=MHR)
        self.est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

    @modal.method()
    def one(self, data: bytes, box: list[float]) -> dict:
        import cv2
        import numpy as np
        import torch
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        with torch.inference_mode():
            people = self.est.process_one_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bboxes=np.array([box], dtype=np.float32))
        if not people:
            return {"error": "no person"}
        p = people[0]
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in p.items()
                if k in ("pred_keypoints_2d", "pred_keypoints_3d", "bbox", "focal_length", "pred_cam_t")}

    @modal.method()
    def many(self, images: dict[str, tuple[bytes, list[float]]]) -> dict[str, dict]:
        """images: name -> (jpeg bytes, captain's seat box)."""
        return {name: self.one.local(data, box) for name, (data, box) in images.items()}


@app.local_entrypoint()
def main(images: str = "eval/pilot_watch/images"):
    """Every <set>/<state>.jpg without a .pose.json yet, with the set's box from boxes.json."""
    d = pathlib.Path(images)
    boxes = json.loads((d.parent / "boxes.json").read_text())
    todo = {f"{p.parent.name}/{p.name}": (p.read_bytes(), boxes[p.parent.name])
            for p in sorted(d.glob("*/*.jpg")) if not p.with_suffix(".pose.json").exists()}
    if not todo:
        print("nothing to do")
        return
    for name, r in Pose().many.remote(todo).items():
        (d / name).with_suffix(".pose.json").write_text(json.dumps(r))
        print(name, r.get("error") or f"{len(r['pred_keypoints_3d'])} keypoints")
