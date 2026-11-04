# Crosscheck's callout bank

Every phrase and word Crosscheck says, recorded once (`voice/render_bank.py`) and played as an
aircraft plays its aural alerts: no model in flight. `voice/bank.py` plays a whole phrase if it is
here, or puts a callout together from the words; anything else falls back to MMS-TTS.

The voice was designed from a description with Fish Audio Drama 3, run locally ("a calm, clear
woman's voice: an aircraft cockpit alerting system", take `alert_f_2`), and every clip cloned from
it. Each candidate and phrase was played to Muse Voice Transcribe and the take it heard best kept
(`report.json`). Played back the same way, nine callouts twice each: 15 of 18 came back word for
word, against 2 of 18 for MMS-TTS ("Fleered one 1000", "Hapting not responding"). One limit is the
recognizer's, not the voice's: "one-two thousand" is sometimes heard as "1000", as on real radio lines.

Fish Audio Community License (non-commercial use). Built with Fish Audio.
