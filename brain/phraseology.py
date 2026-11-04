"""Standard spoken forms for callouts: digits said one by one, as pilots and controllers do."""

DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
          "6": "six", "7": "seven", "8": "eight", "9": "niner"}
TRANSITION_FT = 18000  # US transition altitude: at or above, altitudes are flight levels


def digits(n: int) -> str:
    return "-".join(DIGITS[c] for c in str(n))


def altitude(ft: int) -> str:
    """24000 -> 'two-four-zero' (flight level); 11000 -> 'one-one thousand'; 3500 -> 'three thousand five hundred'."""
    ft = int(round(ft / 100.0) * 100)
    if ft >= TRANSITION_FT:
        return digits(ft // 100)
    thousands, hundreds = divmod(ft, 1000)
    parts = []
    if thousands:
        parts.append(f"{digits(thousands)} thousand")
    if hundreds:
        parts.append(f"{DIGITS[str(hundreds // 100)]} hundred")
    return " ".join(parts) or "zero"


def heading(deg: int) -> str:
    """270 -> 'two-seven-zero'; 5 -> 'zero-zero-five'; 0 -> 'three-six-zero'."""
    return "-".join(DIGITS[c] for c in f"{int(round(deg)) % 360 or 360:03d}")


def flaps(setting: int) -> str:
    words = {0: "up", 1: "one", 2: "two", 5: "five", 10: "ten", 15: "fifteen", 25: "twenty-five", 30: "thirty", 40: "forty"}
    return words.get(int(setting), str(setting))


def altimeter(hundredths: int) -> str:
    """3012 (30.12 inches) -> 'three-zero-one-two', as ATC gives it."""
    return digits(int(hundredths))
