# Crosscheck: the film (about 4 minutes; the short cut is docs/film_script_short.md)

Narration for `uv run python cli.py film`. Each section is one scene; the pictures are real
recordings from `replay/` (the pilot camera is generated footage, and says so on screen).
To narrate it yourself, record each section as `docs/narration/NN.wav` (16-48 kHz mono) and the
film uses your recording instead of the synthetic voice.

## 01 · Title
Crosscheck. A monitoring pilot that only sees and hears.

## 02 · Why
When the NTSB reviewed thirty-seven airline accidents where the crew played a part, it found the
same thing in thirty-one of them. Someone didn't monitor, or didn't challenge. Crosscheck is the
colleague who always does.

## 03 · How it works
It listens to the radio and the intercom through Muse Voice Transcribe. It reads the flight
displays through a camera, with Muse Spark. It never touches the aircraft's data. And it checks
three things against each other: what was said, what was set, and what the aircraft is doing.

## 04 · Altitude bust
Here, air traffic control clears Alaska four two one to eleven thousand feet. The crew reads it
back correctly, then sets twelve thousand. Crosscheck hears the clearance, reads the selector, and
speaks up nine seconds later, seventeen seconds before the aircraft would have passed its altitude.

## 05 · Autothrottle
On final approach, the autothrottle drops out and the speed starts to decay: the pattern behind
Asiana two one four. Crosscheck watches the trend, and calls it before the speed is ten knots low.

## 06 · Flaps
The pilot flying calls flaps fifteen. The pilot monitoring selects ten. Muse Spark reads the analog
gauge, Crosscheck waits for the needle to stop, and calls the difference.

## 07 · Pilot watch
An optional camera watches the captain. SAM three-point-one finds him, SAM 3D Body measures his
posture on every frame. When he slumps, Crosscheck asks first. No answer, and it tells the first
officer to take control. Afterwards, Muse Wonder watches the same footage on its own, and reaches
the same verdict.

## 08 · Evidence
Every run is scored automatically against the simulator. Fifteen minutes of normal flying, with
every line spoken and transcribed live: no false alarm. And where testing found a problem, the
report says what it was, and what changed.

## 09 · Models
Seven Meta models, each doing one job: hearing, reading, watching, checking, and a second opinion.

## 10 · Close
One more pair of eyes and ears on the flight deck. Crosscheck.
