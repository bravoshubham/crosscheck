"""Build the T2 radio test set: known ATC and crew lines, radio-filtered, mixed with engine noise.

Voices: Meta MMS-TTS (facebook/mms-tts-eng, CC-BY-NC 4.0), synthetic until self-recorded lines
replace them. Noise: X-Plane's own engine loops, read from the local install and never written
into the repo (Laminar's assets). Output goes to eval/audio/radio_set/ (git-ignored wavs) with a
manifest of the true text, so WER can be scored with and without SAM Audio.

Run:  uv run --group eval python eval/audio/make_radio_set.py            # X-Plane noise (local only)
      uv run --group eval python eval/audio/make_radio_set.py --synthetic  # radio_set_open/, publishable
"""
import json
import pathlib

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, resample_poly, sosfilt
from transformers import AutoTokenizer, VitsModel

from synth_noise import ENGINES as SYNTH

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "radio_set"
OUT_OPEN = ROOT / "radio_set_open"  # synthetic engine noise (synth_noise.py): safe to publish
SR = 16000
ENGINES = {"jet": pathlib.Path.home() / "X-Plane 12/Resources/sounds/engine/ENGINE_HI_BYPASS_JET_inn.wav",
           "piston": pathlib.Path.home() / "X-Plane 12/Resources/sounds/engine/ENGINE_RECIPROCATING_inn.wav"}
SNRS_DB = [5, 0, -5]

# Spoken forms, as a controller or pilot says them; the manifest keeps this exact text for WER.
LINES = [
    ("atc", "alaska four two one climb and maintain flight level two four zero"),
    ("atc", "alaska four two one turn left heading two seven zero"),
    ("atc", "alaska four two one descend and maintain one one thousand"),
    ("atc", "alaska four two one contact portland approach one two four point three five"),
    ("atc", "alaska four two one reduce speed to two one zero knots"),
    ("atc", "alaska four two one cleared ils runway one zero right approach"),
    ("atc", "cessna three four five charlie papa squawk four three two one"),
    ("atc", "cessna three four five charlie papa maintain three thousand five hundred"),
    ("crew", "flaps fifteen"),
    ("crew", "positive rate gear up"),
    ("crew", "set heading two seven zero"),
    ("crew", "climb flight level two four zero alaska four two one"),
    ("crew", "mayday mayday mayday cessna three four five charlie papa engine failure"),
    ("crew", "one thousand stabilized"),
]


def radio(x: np.ndarray) -> np.ndarray:
    """300-3400 Hz band-pass with light clipping, like an aircraft radio."""
    sos = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
    y = sosfilt(sos, x)
    return np.tanh(2.5 * y / (np.abs(y).max() + 1e-9)) * 0.8


def load_noise(path: pathlib.Path, n: int) -> np.ndarray:
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps, pn = np.mean(speech ** 2), np.mean(noise ** 2) + 1e-12
    y = speech + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return y / (np.abs(y).max() + 1e-9) * 0.9


def main(synthetic: bool = False) -> None:
    out = OUT_OPEN if synthetic else OUT
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    tts = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
    manifest = []
    for i, (who, text) in enumerate(LINES):
        torch.manual_seed(i)
        with torch.inference_mode():
            wav = tts(**tok(text, return_tensors="pt")).waveform[0].numpy()
        wav = resample_poly(wav, SR, tts.config.sampling_rate)
        pad = np.zeros(int(0.4 * SR))
        clean = np.concatenate([pad, radio(wav) if who == "atc" else wav * 0.8, pad])
        sf.write(out / f"{i:02d}_clean.wav", clean, SR)
        engine = "jet" if "alaska" in text or who == "crew" and "cessna" not in text else "piston"
        noise = SYNTH[engine](len(clean), seed=i) if synthetic else load_noise(ENGINES[engine], len(clean))
        for snr in SNRS_DB:
            name = f"{i:02d}_{engine}_snr{snr:+d}.wav"
            sf.write(out / name, mix(clean, noise, snr), SR)
            manifest.append({"file": name, "clean": f"{i:02d}_clean.wav", "speaker": who,
                             "engine": engine, "snr_db": snr, "text": text})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(LINES)} lines x {len(SNRS_DB)} SNRs = {len(manifest)} mixes in {out}")


if __name__ == "__main__":
    import sys
    main(synthetic="--synthetic" in sys.argv)
