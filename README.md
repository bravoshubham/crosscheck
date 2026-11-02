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
| Altitude bust | Cleared to 11,000, MCP set to 12,000 | "Cleared one-one thousand, set one-two thousand." | Called 8.1 s after the mis-set, 17.7 s before the 737 passed 11,000 |
| Readback error | Cleared 11,000, read back 12,000 | "Readback one-two thousand, cleared one-one thousand." | Called right after the readback |
| Correct setting | Cleared and set 11,000 | (silence) | No callout |
| Autothrottle not in speed on final (the Asiana 214 pattern) | 737 on approach to Portland 10R; autothrottle drops out, speed decays | "Speed. Autothrottle not in speed." | Called 9.9 s after the fault, 3.1 s before speed was 10 kt below target; no false callouts in the 25 s before |

In the altitude scenarios the clearance and readback are supplied by the scenario script; the
panel side (reading the MCP altitude from the screen) is real. Speech recognition joins when the
Muse Voice Transcribe path is connected.

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
| Muse Spark | The same decisions through the API; reading panel crops on non-Macs | Waiting for API key |
| SAM 3.1 | Locating each display for calibration | Waiting for API key |
| Muse Voice Transcribe | Crew and ATC speech with speaker labels | Waiting for API key |
| SAM Audio | Separating ATC and each pilot from cockpit noise | Weights ready; GPU run pending |
| SAM 3D Body | Opt-in pilot watch (slumping, unresponsive) | Weights ready; GPU run pending |
| MMS-TTS, MMS ASR | Crosscheck's voice; interim speech baseline | Working |

## Run it yourself

```bash
uv run python cli.py replay --all
```

Replays every recorded scenario in `replay/` from the panel readings saved with it, through the
same ledger, rules and scorer as the live run, and checks the result matches the live one. No
simulator, GPU or API key needed. `--recompute` re-reads the saved panel images (macOS).
*The first recordings are being made now; until they land, `replay/` is empty.*

Live scenarios need X-Plane 12 (the free demo works) on the same machine:

```bash
uv run python eval/scenarios/autothrottle_hold.py --silent
uv run python eval/scenarios/altitude_bust.py bust      # or readback, correct
```

Tests: `uv run pytest`.

## Evaluation so far

| Test | Result | Measured against |
| --- | --- | --- |
| Panel reading (737, screen capture, 60 frames of a climbing turn) | Speed ±2 kt: 57/60 · Altitude ±20 ft: 54/60 · Track ±2°: 54/60 · ~0.6 s/frame | X-Plane's values logged around each capture |
| X-Plane link | 10.0 Hz truth; autopilot, transponder and failure commands work | Read-back of each write |
| On-board decisions (Glimmer) | 8/8 correct, median 2.5 s | Hand-labelled flags |
| Clearance extraction (Glimmer) | Exact text 7/7; noisy recognizer output 4/7 (+5 dB) to 0/7 (−5 dB); 0 wrong values after the guard | Known script of each line |
| Radio word error rate, MMS ASR, no separation | 0.435 clean → 0.911 at −5 dB | Known script; SAM Audio comparison pending |

## Built during the hackathon

Everything in this repository was written during the hackathon (27 Oct – 4 Nov 2026):
the X-Plane bridge, panel reader, ledger and rules, speech guard, live loop, replay mode,
scenarios, evaluation scripts and the GPU apps for SAM Audio and SAM 3D Body.

## Licences and data

Code: MIT. MMS-TTS is CC-BY-NC 4.0 (fine for a hackathon; to be replaced before any commercial
use). SAM Audio and SAM 3D Body weights are gated on Hugging Face and are not redistributed here.
X-Plane's engine sounds are used locally to build test audio and are not included. All procedures
are written from public FAA handbooks; no airline data is used.
