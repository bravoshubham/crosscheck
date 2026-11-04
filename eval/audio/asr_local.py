"""Transcribe the radio set with a Meta open-weights recognizer running on this machine.

The on-board candidates for hearing the radio with no network:
  omni-ctc-1b, omni-llm-1b, omni-llm-3b, omni-llm-7b   Omnilingual ASR (Apache 2.0, fairseq2)
  seamless                                             SeamlessM4T v2 large (CC-BY-NC 4.0, transformers)

Standalone on purpose (numpy and soundfile only, besides the engine), because Omnilingual ASR
needs its own environment (fairseq2 pins torch 2.8):

  ~/llm-stack/omniasr/.venv/bin/python eval/audio/asr_local.py omni-llm-7b [--open | --scenario]
  uv run python eval/audio/asr_local.py seamless [--open | --scenario]

Writes eval/results/day_one/t2_<engine>[_open]_cache.json in the same shape as the Voice
Transcribe cache ({clip: {"text", "s"}}), so eval/audio/wer_local.py scores them side by side.
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENGINE = sys.argv[1]
OPEN = "--open" in sys.argv
SCEN = "--scenario" in sys.argv  # the scenarios' own lines (eval/audio/make_scenario_set.py)
SET = ROOT / "eval" / "audio" / ("scenario_set" if SCEN else "radio_set_open" if OPEN else "radio_set")
TAG = "_scenario" if SCEN else "_open" if OPEN else ""
OUT = ROOT / "eval" / "results" / "day_one" / f"t2_{ENGINE}{TAG}_cache.json"
OMNI = {"omni-ctc-1b": "omniASR_CTC_1B", "omni-llm-1b": "omniASR_LLM_1B",
        "omni-llm-3b": "omniASR_LLM_3B", "omni-llm-7b": "omniASR_LLM_7B"}


def recognizer(engine: str):
    """A function from 16 kHz mono float32 audio to text."""
    import torch
    device = os.environ.get("ASR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    if engine in OMNI:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        card = os.environ.get("OMNI_CARD", OMNI[engine])  # e.g. a local card pointing at downloaded weights
        pipe = ASRInferencePipeline(model_card=card, device=device,
                                    dtype=torch.float32 if device == "cpu" else torch.bfloat16)
        return lambda x: pipe.transcribe([{"waveform": x, "sample_rate": 16000}], lang=["eng_Latn"], batch_size=1)[0]
    if engine == "seamless":
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        name = "facebook/seamless-m4t-v2-large"
        proc = AutoProcessor.from_pretrained(name)
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()

        def run(x):
            inp = proc(audio=x, sampling_rate=16000, return_tensors="pt").to(device)
            inp = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inp.items()}
            with torch.no_grad():
                ids = model.generate(**inp, tgt_lang="eng", num_beams=4, max_new_tokens=96)
            return proc.decode(ids[0], skip_special_tokens=True)
        return run
    raise SystemExit(f"unknown engine {engine}")


def main() -> int:
    manifest = json.loads((SET / "manifest.json").read_text())
    files = sorted({m["clean"] for m in manifest} | {m["file"] for m in manifest})
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [f for f in files if f not in cache]
    print(f"{ENGINE}: {len(files)} clips, {len(todo)} to transcribe")
    if not todo:
        return 0
    t = time.time()
    run = recognizer(ENGINE)
    print(f"  loaded in {time.time() - t:.0f} s")
    for f in todo:
        x, sr = sf.read(SET / f, dtype="float32")
        assert sr == 16000, sr
        t = time.time()
        text = run(np.ascontiguousarray(x if x.ndim == 1 else x.mean(axis=1)))
        cache[f] = {"text": text, "s": round(time.time() - t, 2), "audio_s": round(len(x) / sr, 2)}
        print(f"  {f} ({cache[f]['s']} s): {text}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(cache, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
