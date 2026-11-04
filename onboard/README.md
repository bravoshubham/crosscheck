# On board: Crosscheck with no network

In the air there may be no connection, so the full system runs every model on one computer in the
aircraft and needs nothing else. Everything here is Meta open weights:

| Job | Model (open weights) | Runs as | Cloud mode uses |
| --- | --- | --- | --- |
| Hear the radio and the intercom | SeamlessM4T v2 large + Omnilingual ASR 7B, both on every line | `onboard/hearing_service.py`, 127.0.0.1:8771 | Muse Voice Transcribe |
| Read the PFD (speed, selected speed and altitude, altitude, flight modes) | Muse Glimmer 30B, vision | Ollama (or llama.cpp), 127.0.0.1:11434 | Apple on-device text recognition, or Muse Spark |
| Read the analog flap gauge | Muse Glimmer 30B: "which marking does the needle point at" | the same | Muse Spark |
| Second reader for radio lines code cannot parse | Muse Glimmer 30B | the same | Muse Spark |
| The sentence under each callout | Muse Glimmer 30B | the same | Muse Spark |
| Crosscheck's voice | MMS-TTS | in process | (the same) |
| Pre-flight camera check | Muse Glimmer 30B must read values from two of three places on the PFD (or the gauge) | the same | SAM 3.1 (its open weights are gated) |

The rules that decide when to speak are the same plain code in both modes (`brain/rules.py`).

## Three ways to run it

| | Needs | What runs where |
| --- | --- | --- |
| Replay (`cli.py replay --all`) | Any computer with Python | The recordings, through the same ledger, rules and scorer; no models |
| Mac mode (`cli.py onboard ...`) | An Apple Silicon Mac, X-Plane 12; the models and the simulator took about 40 GB here (tested on 128 GB) | This page's system, every model on the Mac, the network refused |
| The aircraft system | One computer with a GPU, a camera on the displays, the audio panel's feeds | The same software and models; nothing has to leave the aircraft |

Mac mode is the aircraft system on the computer it was tested on. The models run under Ollama /
llama.cpp and PyTorch, which also run on Linux with an NVIDIA GPU, but that was not tested here;
the macOS sandbox that proves a run used no network is macOS-only.

## Hearing: two recognizers, and a value only when they do not disagree

Neither open recognizer is as good as Voice Transcribe alone, and they fail differently:
SeamlessM4T v2 is right on clean speech but drops words it cannot place ("Ten of the flaps."),
Omnilingual ASR hears every word but sometimes as the wrong one ("maintain won one thousand").
So both hear every line. A value that contradicts the aircraft (a climb to below it) is dropped
from either; then a value counts when both heard it, or when only one heard a value at all; if
they heard different values, neither is kept and the local Glimmer is asked, with the same guard
as in cloud mode (the value must be in a transcript, and the line must carry our callsign).

On the scenarios' own lines, voiced as the scenarios voice them (`eval/audio/make_scenario_set.py`,
17 lines x 3 renders, 51 with a value):

| | Right | Missed | Wrong |
| --- | --- | --- | --- |
| Muse Voice Transcribe (cloud) | 49 | 0 | 2 |
| SeamlessM4T v2 alone | 36 | 15 | 0 |
| Omnilingual ASR 7B alone | 45 | 6 | 0 |
| **The on-board pair** | **48** | **3** | **0** |

`uv run python eval/audio/wer_local.py --scenario` reproduces the table from the cached
transcripts. On the noisy radio set (synthetic engine noise at +5 / 0 / -5 dB) both open models
fall behind Voice Transcribe (`eval/audio/wer_local.py --open`): on board, the radio should be a
wired audio feed, not a microphone in the cockpit.

## Seeing

Glimmer reads three parts of the PFD stacked into one image (the mode boxes, the speed tape, the
altitude tape: half the image tokens of the whole display, for the same accuracy on the recorded
frames). The numbers come back as one short string and the three flight modes as a list of
exactly three 737 mode names, because the answer is most of a read's time. On final approach two
reads in three are only the speed tape and the mode boxes. One read is in flight at a time: on a
busy GPU a second one doubles the delay instead of the rate.

On the flap gauge, asking for the clock position of the needle was right on 8 of 20 frames;
asking which marking it points at, with each marking's place on the dial, at 2x zoom: 19 of 20
shipped frames (Muse Spark, cloud: 18 of 20).

**Accuracy and speed** (`eval/panel_onboard.py`, the recorded frames, X-Plane closed): within
tolerance on 57 of 63 selected altitudes, 30 of 36 airspeeds and 53 of 77 altitudes (the rolling
last digits are the hard part), in a median 3.9 s a read on this computer's GPU.

**The limit.** That is enough for the altitude and flap checks. It is not enough for the
autothrottle alert, which needs a fresh reading every second or two: in flight, with X-Plane also
on the GPU, Glimmer read every 4-6 s, and on board the callout came after the speed was 10 kt low
every time (the counts are in the test report). Even alone on this GPU a read takes about 4 s, so
this alert needs a faster GPU or a smaller, faster reader for the speed tape. For the runs X-Plane's
scenery settings were lowered (objects, trees, clouds and shadows off: the view through the
windows, not the panel) and restored afterwards.

## Results, flown in X-Plane with the network refused

Repeat runs with the final code (`eval/results/repeat/*_onboard.json`; earlier runs that found
problems are kept beside them as `*.before_*.json` and not counted):

| Scenario | Right | Crosscheck spoke | Deadline |
| --- | --- | --- | --- |
| Altitude bust | 3 of 3 | 22.7-23.0 s after the MCP was set | the 737 passed 11,000 at 26.0-26.1 s |
| Readback error | 3 of 3 | 5.4-5.8 s after the MCP was set | |
| Correct setting | 3 of 3 | silent, as it should be | |
| Flaps: wrong setting | 3 of 3 | 28.4-33.2 s after the call | |
| Flaps: not moved | 3 of 3 | 32.9-33.3 s after the call | |
| Flaps: correct | 3 of 3 | silent | |
| Autothrottle on final | **0 of 3** | 17.3-22.8 s after the fault | speed 10 kt low at 11.2-12.3 s |
| 15 minutes of normal flying | 1 of 1 | no false alarm; 12 of 14 spoken values right | |

Each scenario also has a shipped recording (`replay/*_onboard/`), counted in the test report: with them, 25 of 29 on-board runs were right, and the four misses are the autothrottle. The first on-board 15-minute flight had four false alarms; the rules were fixed for a slow reader and the flight flown again (the report's findings list both).
The flap callouts take longer than in cloud mode (about 20 s with Muse Spark) because the rule
waits for five gauge readings that agree, and on board they come every few seconds.

## Proving it used no network

`onboard/run.sh` runs any scenario under a macOS sandbox (`onboard/no_network.sb`) that refuses
every connection that is not to this computer. Each run's `meta.json` records the proof: a
connection to api.meta.ai attempted from inside the run, and refused.

    onboard/start.sh                                         # the hearing service + Glimmer, checked
    onboard/run.sh eval/scenarios/altitude_bust.py bust --speak

## Setup (once)

    ollama pull muse-glimmer:30b                             # or meta-models/Muse-Glimmer-30B-GGUF with llama.cpp
    # Omnilingual ASR needs its own environment: fairseq2 pins torch 2.8
    uv venv -p 3.11 ~/llm-stack/omniasr/.venv
    uv pip install --python ~/llm-stack/omniasr/.venv/bin/python omnilingual-asr "torchaudio==2.8.0" "pyarrow<20" \
        soundfile "transformers>=4.57" sentencepiece protobuf
    brew install libsndfile
    hf download facebook/omniASR-LLM-7B && hf download facebook/seamless-m4t-v2-large --exclude "*.pt"
    # point a fairseq2 card at the downloaded checkpoint (Meta's own download URL stalled here):
    # ~/llm-stack/omniasr/cards/local.yaml, name omniASR_LLM_7B_local, checkpoint file://.../omniASR-LLM-7B.pt

Memory on the test computer (Apple M5 Max, 128 GB), measured during the runs: Glimmer 21 GB,
X-Plane 9 GB, the two recognizers a few GB more; macOS reported 80% of memory free. The code uses PyTorch and llama.cpp-compatible servers, which
also run on Linux with an NVIDIA GPU; that was not tested here.

## Not on board yet

- **The pilot watch** still measures posture with SAM 3D Body on a cloud GPU (Modal). Asking
  Glimmer directly was right on 19 of 24 test images (it took five of six forward slumps for a pilot
  looking down; `eval/pilot_watch/posture_glimmer.py`); SAM 3D Body's measurements were right on
  24 of 24, so it should run here. `onboard/pose_service.py` and the scenario's on-board path are
  written; running SAM 3D Body's own code on this computer has not been done yet.
- **Muse Wonder's review** of the pilot-camera recording is a second opinion after the flight; it
  needs the network, which an aircraft has again on the ground.
