"""The scenarios' own radio and intercom lines, voiced exactly as the scenarios voice them, as a test
set for the recognizers: ATC in MMS-TTS through the radio filter, the readback in MMS-TTS, the
flap calls in the Mac's "Samantha". Three renders of each line (MMS-TTS varies its timing).

Run:  uv run python eval/audio/make_scenario_set.py   -> eval/audio/scenario_set/ (+ manifest.json)
"""
import json
import pathlib
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.audio.make_radio_set import radio  # noqa: E402
from senses.speech import said_facts  # noqa: E402
from voice.crew import say  # noqa: E402
from voice.tts import synth  # noqa: E402

OUT = ROOT / "eval" / "audio" / "scenario_set"
RENDERS = 3
ALT = {11000: "one one thousand", 12000: "one two thousand", 6000: "six thousand", 5000: "five thousand",
       4000: "four thousand", 3000: "three thousand", 2000: "two thousand"}


def lines() -> list[tuple[str, str]]:
    out = [("atc", "alaska four two one climb and maintain one one thousand"),
           ("pm", "climb and maintain one one thousand alaska four two one"),
           ("pm", "climb and maintain one two thousand alaska four two one")]
    for v in (6000, 5000, 4000, 3000, 2000):  # the 15-minute flight's descents
        out += [("atc", f"alaska four two one descend and maintain {ALT[v]}"), ("pm", f"descend and maintain {ALT[v]} alaska four two one")]
    for v, spoken in ((120, "one two zero"), (300, "three zero zero")):
        side = "left" if v == 120 else "right"
        out += [("atc", f"alaska four two one turn {side} heading {spoken}"), ("pm", f"{side} heading {spoken} alaska four two one")]
    out += [("pf", f"flaps {w}") for w in ("one", "five", "ten", "fifteen")]
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (who, text) in enumerate(lines()):
        for r in range(RENDERS):
            torch.manual_seed(1000 * i + r)
            if who == "pf":
                pcm = say(text)
            else:
                pcm, sr = synth(text)
                assert sr == 16000, sr
                pcm = radio(pcm) if who == "atc" else pcm * 0.9 / (np.abs(pcm).max() + 1e-9)
            f = f"{i:02d}_{who}_{r}.wav"
            sf.write(OUT / f, pcm.astype(np.float32), 16000)
            manifest.append({"file": f, "clean": f, "snr_db": 0, "speaker": who, "text": text,
                             "facts": [[x.key, x.value] for x in said_facts(text, who, 0, "421")]})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} clips in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
