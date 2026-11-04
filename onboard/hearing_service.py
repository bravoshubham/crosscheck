"""The on-board ears: two Meta open-weights recognizers, loaded once, behind a loopback HTTP port.

  SeamlessM4T v2 large   (CC-BY-NC 4.0)  fast, and right on clean speech; drops words it cannot place
  Omnilingual ASR 7B     (Apache 2.0)    hears every word, sometimes as the wrong word

senses/speech.py sends each line here when CROSSCHECK_ONBOARD=1 and gets both transcripts;
senses/speech_spark.py keeps a value only when the two agree, or when only one heard one
(eval/audio/wer_local.py: 45 of 51 scenario facts right, none wrong). Runs in its own environment
because Omnilingual ASR's fairseq2 pins torch 2.8 (see onboard/README.md):

  FAIRSEQ2_USER_ASSET_DIR=~/llm-stack/omniasr/cards \
    ~/llm-stack/omniasr/.venv/bin/python onboard/hearing_service.py      # 127.0.0.1:8771

POST /transcribe with raw 16 kHz mono float32 samples; the reply is
{"seamless": text, "omni": text, "seconds": {...}}. GET /health says whether both are loaded.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

PORT = int(os.environ.get("HEARING_PORT", "8771"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
OMNI_CARD = os.environ.get("OMNI_CARD", "omniASR_LLM_7B_local")
SEAMLESS = "facebook/seamless-m4t-v2-large"
LOCK = threading.Lock()  # one GPU: one line at a time
MODELS: dict = {}


def load() -> None:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    t = time.time()
    MODELS["omni"] = ASRInferencePipeline(model_card=OMNI_CARD, device=DEVICE, dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32)
    MODELS["proc"] = AutoProcessor.from_pretrained(SEAMLESS)
    MODELS["seamless"] = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        SEAMLESS, dtype=torch.float16 if DEVICE != "cpu" else torch.float32).to(DEVICE).eval()
    print(f"hearing service: both recognizers on {DEVICE} in {time.time() - t:.0f} s, port {PORT}", flush=True)


def seamless(x: np.ndarray) -> str:
    proc, model = MODELS["proc"], MODELS["seamless"]
    inp = proc(audio=x, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    inp = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inp.items()}
    with torch.no_grad():
        ids = model.generate(**inp, tgt_lang="eng", num_beams=4, max_new_tokens=96)
    return proc.decode(ids[0], skip_special_tokens=True)


def omni(x: np.ndarray) -> str:
    return MODELS["omni"].transcribe([{"waveform": x, "sample_rate": 16000}], lang=["eng_Latn"], batch_size=1)[0]


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply(200 if "seamless" in MODELS else 503, {"ok": "seamless" in MODELS, "device": DEVICE, "omni": OMNI_CARD})

    def do_POST(self):
        if self.path != "/transcribe":
            return self._reply(404, {"error": "POST /transcribe"})
        x = np.frombuffer(self.rfile.read(int(self.headers["Content-Length"])), dtype="<f4").copy()
        out, secs = {}, {}
        with LOCK:
            for name, fn in (("seamless", seamless), ("omni", omni)):
                t = time.time()
                out[name] = fn(x)
                secs[name] = round(time.time() - t, 2)
        self._reply(200, {**out, "seconds": secs, "audio_s": round(len(x) / 16000, 2)})

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    load()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
