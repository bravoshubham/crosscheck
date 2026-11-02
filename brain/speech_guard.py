"""Guard between the language model and the ledger: an extracted value is kept only if the
transcript supports it. Found necessary in T3, where Glimmer copied a worked example
("delta four, heading three one zero") onto a garbled line that said neither."""
import re

WORD_DIGITS = {"zero": "0", "oh": "0", "one": "1", "won": "1", "two": "2", "too": "2", "to": "2", "three": "3",
               "tree": "3", "four": "4", "for": "4", "fower": "4", "five": "5", "fife": "5", "six": "6",
               "seven": "7", "eight": "8", "nine": "9", "niner": "9"}
SCALE = {"thousand": 3, "hundred": 2}


def spoken_digits(transcript: str) -> str:
    """Digits in speaking order: 'one one thousand' -> '11000', 'point three five' -> '35'."""
    out = []
    for w in re.findall(r"[a-z]+|\d+", transcript.lower()):
        if w.isdigit():
            out.append(w)
        elif w in WORD_DIGITS:
            out.append(WORD_DIGITS[w])
        elif w in SCALE:
            out.append("0" * SCALE[w])
    return "".join(out)


def spoken_values(transcript: str) -> set[int]:
    """Compound numbers as spoken: 'three thousand five hundred' -> 3500, 'one one thousand' -> 11000."""
    vals, group, total = set(), "", 0
    for w in re.findall(r"[a-z]+|\d+", transcript.lower()) + ["."]:
        d = w if w.isdigit() else WORD_DIGITS.get(w)
        if d is not None:
            group += d
        elif w in SCALE and group:
            total += int(group) * 10 ** SCALE[w]
            group = ""
            vals.add(total)
        else:
            if group:
                vals.add(total + int(group))
            elif total:
                vals.add(total)
            group, total = "", 0
    return vals


def _supported_number(value, digits: str, values: set[int] | None = None) -> bool:
    if values and isinstance(value, (int, float)) and int(value) == value and int(value) in values:
        return True
    s = str(value).replace(".", "")
    s = s.rstrip("0") or "0"  # "24000" is spoken "two four zero" (flight level) or "two four thousand"
    return s in digits


def guard(extraction: dict, transcript: str, fields: list[str]) -> tuple[dict, list[str]]:
    """Return the extraction with unsupported values set to None, and the names of dropped fields."""
    digits = spoken_digits(transcript)
    values = spoken_values(transcript)
    words = set(re.findall(r"[a-z]+", transcript.lower()))
    kept, dropped = dict(extraction), []
    call = set(re.findall(r"[a-z]+", str(extraction.get("callsign") or "").lower()))
    if call and not call & words:
        kept["callsign"] = None
        dropped.append("callsign")
    for f in fields:
        v = extraction.get(f)
        if v is not None and not _supported_number(v, digits, values):
            kept[f] = None
            dropped.append(f)
    return kept, dropped
