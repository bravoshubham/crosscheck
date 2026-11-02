"""Synthetic cockpit engine noise, so the radio test set can be published (X-Plane's own engine
sounds cannot). Designed by hand, not derived from any recording; deterministic for a seed.

  jet    - broadband roar falling ~3 dB per octave above 300 Hz, a low rumble, and two fan tones
           (1.15 and 2.3 kHz) with slow wobble
  piston - 80 Hz firing frequency (2,400 rpm, 4 cylinders) with decaying harmonics and slight
           jitter, over low-frequency exhaust noise
"""
import numpy as np

SR = 16000


def _shaped(n: int, rng, tilt_db_per_oct: float, corner_hz: float, low_boost_db: float = 0.0) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    gain = np.where(f > corner_hz, (f / corner_hz) ** (tilt_db_per_oct / 6.02), 1.0)
    gain *= 1 + (10 ** (low_boost_db / 20) - 1) * np.exp(-f / 120)
    x = np.fft.irfft(spec * gain, n)
    return x / (np.std(x) + 1e-12)


def jet(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    x = _shaped(n, rng, -3.0, 300.0, low_boost_db=6.0)
    for hz, level in ((1150.0, 0.35), (2300.0, 0.18)):
        wobble = 1 + 0.004 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28))
        x += level * np.sin(2 * np.pi * hz * np.cumsum(wobble) / SR)
    return x / np.std(x)


def piston(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    f0 = 80.0 * (1 + 0.01 * np.sin(2 * np.pi * 0.5 * t) + 0.003 * rng.standard_normal(n).cumsum() / np.sqrt(SR))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum((1.0 / k) * np.sin(k * phase + rng.uniform(0, 6.28)) for k in range(1, 25))
    x = x / np.std(x) + 0.8 * _shaped(n, rng, -6.0, 150.0, low_boost_db=4.0)
    return x / np.std(x)


ENGINES = {"jet": jet, "piston": piston}
