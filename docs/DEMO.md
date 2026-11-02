# Presenting Crosscheck live

A 5-minute live demo on one Mac, with an iPad as the crew's tablet. Every step has a fallback
that needs no simulator.

## Before you go on (10 minutes)

1. **Restart X-Plane 12** so the demo's 60 minutes start fresh. Load any flight once, so the start-up
   dialogs are gone. Scenarios load their own flight.
2. **Mac:** Do Not Disturb on, volume about 60 %, nothing in front of X-Plane. The camera reads the screen,
   and SAM 3.1 refuses to start a run if it cannot see the flight displays.
3. **Microphone:** run `uv run python -m senses.mic` and say "flaps fifteen". It should print `Flaps 15.`
   within about a second. If it prints nothing, check System Settings, Privacy & Security, Microphone.
4. **iPad:** same Wi-Fi as the Mac. Any demo command prints the address to open, for example
   `http://192.168.29.42:8770`.
5. **Fallback page:** open `site/index.html` in a browser tab (`uv run python cli.py debrief` builds it).
6. Check `.env` has `MODEL_API_KEY`, and that the network works: `uv run python cli.py summary replay/flaps_wrong`
   answers in a few seconds.

## The demo

| Time | You do | They see |
| --- | --- | --- |
| 0:00 | Open `site/index.html`. "Said 11,000. Set 12,000." | A real recording: the clearance, the selector, the aircraft and the callout on one timeline |
| 0:40 | `uv run python cli.py demo altitude` | X-Plane at 9,000 ft. ATC on the speakers: "Alaska four two one, climb and maintain one one thousand" |
| 0:55 | Read it back **wrong**: "Climb and maintain one **two** thousand, Alaska four two one" | Tablet: your words as Voice Transcribe heard them, then "Readback one-two thousand, cleared one-one thousand", spoken, with Muse Spark's sentence under it |
| 2:00 | `uv run python cli.py demo flaps`; when prompted, say "flaps fifteen" | The pilot monitoring sets 10. The needle moves; Muse Spark reads the gauge; "Flaps fifteen called, ten set." |
| 3:15 | `uv run python cli.py demo pilot` | The captain on the pilot camera slumps. "Captain, are you OK?" Answer out loud: "I'm fine." No alert. Then Muse Wonder's independent verdict on the footage |
| 4:15 | Back to `site/index.html#report` | The test matrix, the 15-minute false-alarm flight, and what testing caught |

What to say about the Meta models, in one breath each:

- **Muse Voice Transcribe** hears every word, from the radio and from you, live.
- **Muse Spark** reads the panel through the camera, reads radio lines the parser cannot, and explains each callout.
- **SAM 3.1** checks the camera sees the displays before every run, and finds the pilot.
- **SAM 3D Body** gives the pilot's posture in 3D on every frame.
- **Muse Wonder** watches the footage afterwards and gives a second opinion that shares nothing with SAM 3D Body.
- **Muse Image** made the synthetic pilots, so no real person is filmed.

## When something goes wrong

| Problem | Do this |
| --- | --- |
| X-Plane is slow, crashes, or shows "demo time is up" | `uv run python cli.py demo replay altitude_readback` re-enacts the recording in real time: same tablet, same callouts, spoken |
| The microphone mishears you | Say it again. Only a line with our callsign's digits counts. Or re-enact `altitude_readback` |
| No network (no Voice Transcribe, Spark, SAM 3.1) | The debrief and re-enactments need no network: `site/index.html`, and `cli.py demo replay <recording>` |
| The camera check refuses to start | Something is in front of X-Plane, or the view moved. Click X-Plane and run again |
| The tablet shows "reconnecting" | Only one demo command at a time; stop the old one (Ctrl-C) and run again |

Recordings that can be re-enacted: `altitude_bust`, `altitude_readback`, `altitude_correct`,
`autothrottle_hold`, `flaps_wrong`, `flaps_forgot`, `flaps_correct`, `pilot_slumped`, `pilot_asleep`,
`pilot_checklist`, `pilot_answers`, `nominal` (15 minutes).

Live demos write to `demo_runs/`, which is git-ignored: your voice stays on the Mac.
