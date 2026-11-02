"""Crosscheck's voice: Meta MMS-TTS (facebook/mms-tts-eng, CC-BY-NC 4.0; swap before any
commercial launch). One calm voice, played through the Mac's speakers or a chosen device."""
import pathlib
import subprocess
import tempfile
import threading

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

_lock = threading.Lock()
_model = None


def _load():
    global _model
    if _model is None:
        tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        vits = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
        _model = (tok, vits)
    return _model


def synth(text: str) -> tuple[np.ndarray, int]:
    tok, vits = _load()
    # MMS reads words, not punctuation-heavy text: "two-four-zero" -> "two four zero"
    clean = text.replace("-", " ").replace(".", "").replace(",", "").lower()
    torch.manual_seed(0)
    with torch.inference_mode():
        wav = vits(**tok(clean, return_tensors="pt")).waveform[0].numpy()
    return wav, vits.config.sampling_rate


def speak(text: str, block: bool = False, save_to: pathlib.Path | None = None) -> threading.Thread | None:
    """Synthesize and play. Non-blocking by default so the monitoring loop never waits on audio."""
    def run():
        with _lock:
            wav, sr = synth(text)
            path = save_to or pathlib.Path(tempfile.mkstemp(suffix=".wav")[1])
            sf.write(path, wav, sr)
            subprocess.run(["afplay", str(path)], check=False)
    if block:
        run()
        return None
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t
