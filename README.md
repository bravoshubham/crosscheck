# Crosscheck

An AI that monitors the pilots. It listens to the crew and the radio, reads the panel through a
camera, and calls out when what was **said**, what was **set** and what the aircraft is **doing**
don't match — on Meta's models, proven in X-Plane against the simulator's own truth.

> In the NTSB's review of 37 crew-involved US airline accidents (1978–1990), failures to monitor
> and challenge were found in 31. ([NTSB SS-94/01](https://www.ntsb.gov/safety/safety-studies/Documents/SS9401.pdf))

**Status (2 Nov 2026): work in progress, built during the Meta AI Developer Hackathon.**
Numbers below are from single runs unless stated; the evaluation section says what each was
measured against.

## What it catches

| Scenario | What happens | Crosscheck says | First result |
| --- | --- | --- | --- |
| Altitude bust | Cleared to 11,000, MCP set to 12,000 | "Cleared one-one thousand, set one-two thousand." | Called 9.0 s after the mis-set, 17.2 s before the 737 passed 11,000 (earlier run: 10.7 s / 15.9 s) |
| Readback error | Cleared 11,000, read back 12,000 | "Readback one-two thousand, cleared one-one thousand." | Called as soon as the readback was transcribed |
| Correct setting | Cleared and set 11,000 | (silence) | No callout |
| Autothrottle not in speed on final (the Asiana 214 pattern) | 737 on approach to Portland 10R; autothrottle drops out, speed decays | "Speed. Autothrottle not in speed." | Called 9.9 s after the fault, 3.1 s before speed was 10 kt below target; no false callouts in the 25 s before |
| Pilot incapacitated (pilot camera, 4 variants) | Captain slumps forward / falls asleep / reads a checklist / slumps but answers | "Captain, are you OK?" then "Captain not responding. First officer, take control." / silence / the question only | Asked 8.5–11.1 s after the slump, alert ~10 s later; silent for the checklist; no alert once the captain answered |
| Flaps not set as called (3 variants) | "Flaps fifteen" called; lever to 10 / not moved / to 15 | "Flaps fifteen called, ten set." / "…five set." / silence | Called 17.6 s and 16.2 s after the call; silent while the needle took 13 s to reach 15 |

The pilot watch is an opt-in camera on the pilot. Its images are generated with Muse Image (no
real person is filmed): one photo per pilot, edited into upright, slumped forward, asleep against
the window, and reading a checklist. Every frame goes to SAM 3D Body (on a Modal GPU, ~2 s a
frame); code turns its 3D keypoints into neck flexion and hand height. A slumped posture alone
never alerts: Crosscheck asks first, and an answer heard on the intercom clears it. The
thresholds were chosen on one image set; a first guess flagged 3 of 6 normal images on three
held-out sets, so they were re-derived from the first set alone and tested on two sets made
afterwards: 8/8 (24/24 over all six sets).

In the flaps scenarios the camera looks at the centre panel and Muse Spark reads the analog flap
gauge (text recognition cannot read a needle): Spark says where the needle points, code turns
that into a detent from a table measured in X-Plane, and the rule waits until the needle has
stopped — five readings over 5 s — and gives the crew 5 s to move the lever, so a correct
selection is never flagged mid-travel. That makes it slower than the altitude checks (~16 s).
The callout itself is spoken in a macOS voice on the intercom and heard by Voice Transcribe.
The settle rule and the dial table were tuned on a first set of runs (which exposed a false
alarm on the correct selection and a misread 5-as-2); the recordings shipped are fresh runs,
one per variant, made after those fixes.

In the altitude scenarios both sides are real. The clearance (radio-filtered) and the readback
are spoken by a synthetic voice, each streamed on its own channel to Muse Voice Transcribe, and
only what comes back is parsed into the ledger (`senses/speech.py`); the MCP altitude is read
from the screen. The final transcript arrives ~1.3 s after each line ends. Recognition is not
perfect and the design assumes so: the readback came back as "Plan een mensen one two thousand
Alaska four two one" — the verb is lost but the value and callsign digits survive, and those are
all Crosscheck takes. A line without our callsign digits is ignored. `--scripted` runs the same
scenario with the values typed in (no API key).

## How it works

```
panel camera ─ panel reader ─┐
                             ├─► ledger of said / set / doing facts ─► rules ─► callout (MMS-TTS)
crew + ATC audio ─ speech ───┘                                          │
                                                                        └─► Muse model decides in non-standard cases
X-Plane truth (10 Hz) ─────────────────────────────────────────────────► evaluator only
```

- **It never reads the aircraft's data.** It only sees and hears, like a human monitoring pilot.
  The simulator's data is used only to score it.
- **Rules fire instantly; standard callouts use fixed phraseology** (`brain/rules.py`,
  `brain/phraseology.py`). A Muse model chooses the action only when the situation is not standard.
- **Spoken values are checked against the transcript** before they enter the ledger
  (`brain/speech_guard.py`): in testing, the model once copied an example instruction onto a
  garbled radio line; the guard drops anything the transcript does not support.

## Meta models

| Model | Job | State |
| --- | --- | --- |
| Muse Glimmer 30B (open weights, on the Mac) | Decisions on board with no network; clearance extraction | Working: decisions in 2.5 s; extraction 7/7 on clean text |
| Muse Spark | Reading the panel (any OS); decisions through the API | Panel reader working and more accurate than the on-device one (below); live in all four scenarios — fast enough for clearance checks, ~3–5 s too slow for the autothrottle callout. Decisions not yet wired in |
| SAM 3.1 | Locating each display for calibration | Key works; not yet wired in |
| Muse Voice Transcribe | Crew and ATC speech, one channel each | Live in the altitude scenarios; radio set scored (below) |
| SAM 3D Body | Pilot watch: body pose from the pilot camera | Working on a Modal T4, ~2 s a frame warm (29 s cold start); posture test 8/8 on clean held-out images; live in the pilot-watch scenario |
| Muse Image | Generating the pilot-camera test images (and edits of each) | Working, ~$0.01 an image; the content filter refused a "passed out" prompt, and some edits at random (a retry worked) |
| SAM Audio | Separating overlapping speakers | Runs on Modal (T4); cleans the noise (+5.8 to +10.7 dB SI-SDR) but makes transcription worse, so it is kept out of the speech path |
| MMS-TTS, MMS ASR | Crosscheck's voice; interim speech baseline | Working |

## See it

```bash
uv run python cli.py debrief      # then open site/index.html
```

The debrief viewer replays each recorded scenario: the cockpit frames Crosscheck's camera saw,
what was said (as Voice Transcribe heard it), what was set and what the aircraft was doing (as
read from the panel), and each callout, spoken at the moment it fired — on one timeline with
X-Plane's own values drawn dashed, for scoring only. A static page; no server or simulator.

During a live run, `uv run python cli.py live` serves the tablet display on port 8770 (open it
from an iPad on the same network): the current callout in one band, what was said, set and done
in three columns, the camera's latest frame and a log. It follows whichever scenario is
recording by tailing its event log, so it cannot slow down or change the monitoring loop.

## Run it yourself

```bash
uv run python cli.py replay --all
```

Replays every recorded scenario in `replay/` from the panel readings saved with it, through the
same ledger, rules and scorer as the live run, and checks the result matches the live one. No
simulator, GPU or API key needed. `--recompute` re-reads the saved panel images (macOS);
`--recompute --reader spark` re-reads them with Muse Spark on any OS (needs `MODEL_API_KEY`;
~$0.003 a frame, ~$0.60 for all frames of the four on-device recordings).
Eleven recordings ship today (2 Nov): the four scenarios with the on-device panel reader
(`autothrottle_hold`, `altitude_bust`, `altitude_readback`, `altitude_correct`), the same four
with Muse Spark reading the panel live (`*_spark`), and the three flap variants (`flaps_*`, flap
gauge read by Spark). All eleven replay exactly as they ran live. Ten pass; `autothrottle_hold_spark` fails as recorded — Spark's readings arrive ~5 s after
capture, so the callout came after speed was already 10 kt low — and is kept as evidence. The
on-device four also match with `--recompute`, and with `--recompute --reader spark`.

Live scenarios need X-Plane 12 on the same machine. The free demo works; `bridge/demo_dialogs.py`
clicks the demo's own "Use Demo" / "Understood" / "OK" dialogs so runs can be unattended (macOS
must allow the app to control the computer):

```bash
uv run python eval/scenarios/autothrottle_hold.py --silent
uv run python eval/scenarios/altitude_bust.py bust      # or readback, correct
# add --spark to read the panel with Muse Spark instead (recorded as *_spark)
uv run python eval/scenarios/flaps.py wrong             # or forgot, correct
uv run modal deploy cloud/sam3d_body_app.py             # once, for the pilot watch (Modal GPU)
uv run --group eval python eval/scenarios/pilot_watch.py slumped   # or asleep, checklist, answers; no simulator
```

Tests: `uv run pytest`.

## Evaluation so far

| Test | Result | Measured against |
| --- | --- | --- |
| Panel reading, Muse Spark vs on-device, every 3rd frame of the four recordings (67 frames) | Altitude ±20 ft: **64/67** vs 56/67 · Selected altitude exact: **49/49** vs 44/49 · Speed ±2 kt: **18/18** vs 17/18 · Thrust mode: **17/17** vs 16/17 · Spark median 2.8 s/frame, $0.20 in all (`eval/panel_spark.py`) | X-Plane's values in the second before each capture; thrust mode against the scripted fault time |
| Panel reading (737, screen capture, 60 frames of a climbing turn) | Speed ±2 kt: 57/60 · Altitude ±20 ft: 54/60 · Track ±2°: 54/60 · ~0.6 s/frame | X-Plane's values logged around each capture |
| X-Plane link | 10.0 Hz truth; autopilot, transponder and failure commands work | Read-back of each write |
| On-board decisions (Glimmer) | 8/8 correct, median 2.5 s | Hand-labelled flags |
| Clearance extraction (Glimmer) | Exact text 7/7; noisy recognizer output 4/7 (+5 dB) to 0/7 (−5 dB); 0 wrong values after the guard | Known script of each line |
| Radio speech, Muse Voice Transcribe (14 lines × clean, +5/0/−5 dB jet or piston noise) | Lines with every number exact: clean 13/13; +5 dB 12/13; 0 dB 9/13; −5 dB 10/13. After SAM Audio: 11, 8, 7. Meta's open MMS ASR: 5, 3, 1. 98 clips cost ≈ $0.02 | Known script of each line (`eval/audio/wer_muse.py`) |
| Altitude facts from those transcripts (`senses/speech.py`) | 0 wrong altitudes into the ledger from 98 transcripts; misses where "flight level" became "white level" or the callsign digits were lost | Known script |
| Callsign recognition (same set) | "Alaska" missed in 2/7 clean, 5/21 noisy lines ("Oscar", "Alas, for"); the guard drops it rather than misattribute | Known script |

## Built during the hackathon

Everything in this repository was written during the hackathon (27 Oct – 4 Nov 2026):
the X-Plane bridge, panel reader, ledger and rules, speech guard, live loop, replay mode,
scenarios, evaluation scripts and the GPU apps for SAM Audio and SAM 3D Body.

## Licences and data

Code: MIT. MMS-TTS is CC-BY-NC 4.0 (fine for a hackathon; to be replaced before any commercial
use). SAM Audio and SAM 3D Body weights are gated on Hugging Face and are not redistributed here.
X-Plane's engine sounds are used locally to build test audio and are not included. All procedures
are written from public FAA handbooks; no airline data is used.
