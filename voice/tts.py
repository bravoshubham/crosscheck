"""Crosscheck's voice. Callouts come from the recorded bank (voice/bank.py) when it can say them,
like an aircraft's aural alerts; anything else, and the synthetic radio voices of the test sets,
from Meta MMS-TTS (facebook/mms-tts-eng, CC-BY-NC 4.0; swap before any commercial launch)."""
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


def callout(text: str) -> tuple[np.ndarray, int]:
    """A callout as Crosscheck says it: from the bank if it can, else synthesised."""
    from voice.bank import SR, audio
    wav = audio(text)
    return (wav, SR) if wav is not None else synth(text)


def speak(text: str, block: bool = False, save_to: pathlib.Path | None = None) -> threading.Thread | None:
    """Play a callout. Non-blocking by default so the monitoring loop never waits on audio."""
    def run():
        with _lock:
            wav, sr = callout(text)
            path = save_to or pathlib.Path(tempfile.mkstemp(suffix=".wav")[1])
            sf.write(path, wav, sr)
            subprocess.run(["afplay", str(path)], check=False)
    if block:
        run()
        return None
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t
