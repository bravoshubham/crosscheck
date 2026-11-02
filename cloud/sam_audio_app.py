"""SAM Audio on a Modal GPU: pull one voice out of cockpit noise by text prompt.

Weights live in the private Modal volume `crosscheck-models` (uploaded from a local
Hugging Face download, so no token is stored in Modal). The volume's config.json sets
visual_ranker, text_ranker and span_predictor to null (original kept as config.orig.json):
we separate with one candidate and no span prediction, so the rankers would never run, and
loading them would pull another gated 12 GB model (facebook/sam-audio-judge) into the container.

Batch the T2 radio set:
  uv run modal run cloud/sam_audio_app.py --set eval/audio/radio_set --prompt "a person speaking"
"""
import io
import json
import pathlib

import modal

app = modal.App("crosscheck-sam-audio")
models = modal.Volume.from_name("crosscheck-models")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "ffmpeg")
         .pip_install("torch", "torchaudio", "soundfile", "numpy")
         .pip_install("git+https://github.com/facebookresearch/sam-audio")
         # sam-audio's dependencies pull protobuf 3.19, which breaks Modal's own client in the container
         .pip_install("protobuf>=4.25,<6",
                      # sam-audio's from_pretrained passes proxies/resume_download, removed in huggingface_hub 1.0
                      "huggingface_hub>=0.30,<1.0", "transformers>=4.45,<5"))


@app.cls(image=image, gpu="T4", volumes={"/models": models}, timeout=1800, scaledown_window=120)
class SamAudio:
    @modal.enter()
    def load(self):
        import torch
        from sam_audio import SAMAudio, SAMAudioProcessor
        self.torch = torch
        self.model = SAMAudio.from_pretrained("/models/sam-audio-base").eval().cuda()
        self.proc = SAMAudioProcessor.from_pretrained("/models/sam-audio-base")

    @modal.method()
    def separate(self, wavs: list[bytes], prompt: str) -> list[bytes]:
        import soundfile as sf
        import torchaudio
        out = []
        for raw in wavs:
            x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            t = self.torch.from_numpy(x.T.mean(0, keepdims=True))
            if sr != self.proc.audio_sampling_rate:
                t = torchaudio.functional.resample(t, sr, self.proc.audio_sampling_rate)
            batch = self.proc(audios=[t], descriptions=[prompt]).to("cuda")
            with self.torch.inference_mode():
                res = self.model.separate(batch, predict_spans=False, reranking_candidates=1)
            target = res.target[0] if isinstance(res.target, (list, tuple)) else res.target
            y = target.float().cpu().reshape(1, -1)
            y = torchaudio.functional.resample(y, self.proc.audio_sampling_rate, sr)
            buf = io.BytesIO()
            sf.write(buf, y[0].numpy(), sr, format="WAV")
            out.append(buf.getvalue())
        return out


@app.local_entrypoint()
def main(set: str = "eval/audio/radio_set", prompt: str = "a person speaking"):
    root = pathlib.Path(set)
    manifest = json.loads((root / "manifest.json").read_text())
    files = [root / m["file"] for m in manifest]
    outs = SamAudio().separate.remote([f.read_bytes() for f in files], prompt)
    for f, b in zip(files, outs):
        f.with_suffix(".sam.wav").write_bytes(b)
    print(f"separated {len(outs)} files with prompt {prompt!r}")
