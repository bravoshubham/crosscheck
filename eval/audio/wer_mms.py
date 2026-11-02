"""Word error rate on the radio set with Meta's open MMS ASR (facebook/mms-1b-all, English adapter).

Interim scorer that costs no API credits; Muse Voice Transcribe replaces it for the reported
numbers. Scores clean speech, raw mixes, and (if present) SAM Audio outputs named
<mix>.sam.wav, grouped by SNR.

Run:  uv run --group eval python eval/audio/wer_mms.py
"""
import collections
import json
import pathlib
import sys

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

SET = pathlib.Path(__file__).resolve().parent / "radio_set"
MODEL = "facebook/mms-1b-all"


def main() -> int:
    proc = AutoProcessor.from_pretrained(MODEL, target_lang="eng")
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, target_lang="eng", ignore_mismatched_sizes=True).eval()

    def transcribe(path: pathlib.Path) -> str:
        x, sr = sf.read(path)
        assert sr == 16000, sr
        with torch.inference_mode():
            logits = model(**proc(x.astype(np.float32), sampling_rate=sr, return_tensors="pt")).logits
        return proc.decode(logits[0].argmax(-1))

    manifest = json.loads((SET / "manifest.json").read_text())
    groups: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    rows, seen_clean = [], set()
    for m in manifest:
        if m["clean"] not in seen_clean:
            seen_clean.add(m["clean"])
            hyp = transcribe(SET / m["clean"])
            groups["clean"].append((m["text"], hyp))
        hyp = transcribe(SET / m["file"])
        groups[f"raw snr{m['snr_db']:+d}"].append((m["text"], hyp))
        row = {"file": m["file"], "ref": m["text"], "raw": hyp}
        sam = SET / m["file"].replace(".wav", ".sam.wav")
        if sam.exists():
            row["sam"] = transcribe(sam)
            groups[f"sam snr{m['snr_db']:+d}"].append((m["text"], row["sam"]))
        rows.append(row)
    summary = {k: round(jiwer.wer([r for r, _ in v], [h for _, h in v]), 3) for k, v in sorted(groups.items())}
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "day_one"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t2_wer_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "t2_wer.json").write_text(json.dumps({"asr": MODEL, "wer": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
