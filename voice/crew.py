"""Voices for the simulated crew and ATC in scenarios: the Mac's built-in speech (`say`).

Meta MMS-TTS stays Crosscheck's own voice. For the people around it, Voice Transcribe hears the
Mac voices far better: "flaps fifteen" in MMS-TTS came back as "at fifteen" / "Twites 15", in
"Samantha" as "Flaps 15." on every try. A different voice also makes it obvious in a demo who is
speaking. macOS only, like the scenarios themselves.
"""
import pathlib
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

CREW = "Samantha"
ATC = "Samantha"
SR = 16000


def say(text: str, voice: str = CREW) -> np.ndarray:
    """16 kHz mono float audio of `text` in a macOS voice, with a short lead-in like a keyed mic."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "line.wav"
        subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@22050", text], check=True)
        x, sr = sf.read(path)
    x = resample_poly(x, SR, sr) if sr != SR else x
    return np.concatenate([np.zeros(SR // 4), x, np.zeros(SR // 4)])
