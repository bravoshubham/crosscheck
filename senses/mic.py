"""Live microphone: a person's voice in, one transcript per spoken line out.

The Mac's microphone is read in 80 ms blocks. A line starts when the level rises well above the
room's noise floor (measured over the first second) and ends after QUIET_S of quiet; while the
person is still speaking, the audio is already streaming to Muse Voice Transcribe, so the
transcript is ready ~1 s after they stop. The channel decides who spoke: the scenario says
which crew seat the microphone is (the radio stays synthesised). While the speakers play
something (the ATC line, an instruction), the microphone is muted, like a radio that cannot
transmit while it receives, so the person's line is the only thing it transcribes.

  uv run python -m senses.mic          # prints what it hears until Ctrl-C
"""
import asyncio
import collections
import threading
import time

import numpy as np

from senses.speech import PACKET_S, SR, transcribe_stream

BLOCK = int(SR * PACKET_S)
QUIET_S = 0.8      # this much quiet ends a line
MAX_LINE_S = 12.0  # a line longer than this is cut (an open mic, not a callout)
PRE_ROLL = 4       # blocks kept from before the level rose, so the first syllable is not lost
START_BLOCKS = 2   # loud blocks in a row that start a line


def _rms(b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((b.astype(np.float32) / 32768) ** 2)))


async def lines(stop: asyncio.Event, device=None, audio: np.ndarray | None = None, muted: threading.Event | None = None):
    """Yield {"t_start", "t", "text", "audio_s", "pcm"} for every line spoken until `stop` is set.
    audio: int16 16 kHz samples played in at real time instead of the microphone (for tests)."""
    import contextlib
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def callback(indata, frames, t, status):
        loop.call_soon_threadsafe(q.put_nowait, (time.time(), indata[:, 0].copy()))

    async def feed():
        for i in range(0, len(audio) - BLOCK, BLOCK):
            q.put_nowait((time.time(), audio[i:i + BLOCK]))
            await asyncio.sleep(PACKET_S)
        stop.set()

    if audio is None:
        import sounddevice as sd
        source = sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=BLOCK, device=device, callback=callback)
    else:
        source = contextlib.nullcontext()
        feeder = asyncio.create_task(feed())
    with source:
        floor = []
        while len(floor) < int(1 / PACKET_S):  # noise floor: the first second, nobody speaking
            floor.append(_rms((await q.get())[1]))
        threshold = max(0.012, 4 * float(np.median(floor)))
        pre: collections.deque = collections.deque(maxlen=PRE_ROLL)
        loud = 0
        while not stop.is_set():
            try:
                t, b = await asyncio.wait_for(q.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            if muted is not None and muted.is_set():
                pre.clear()
                loud = 0
                continue
            pre.append(b)
            loud = loud + 1 if _rms(b) > threshold else 0
            if loud < START_BLOCKS:
                continue
            t_start = t - PACKET_S * (len(pre) - 1)

            got = list(pre)

            async def chunks():
                for c in list(got):
                    yield c.tobytes()
                quiet, t_line = 0.0, time.time()
                while quiet < QUIET_S and time.time() - t_line < MAX_LINE_S:
                    _, c = await q.get()
                    quiet = quiet + PACKET_S if _rms(c) <= threshold else 0.0
                    got.append(c)
                    yield c.tobytes()
            r = await transcribe_stream(chunks())
            pre.clear()
            loud = 0
            if r["text"].strip():
                yield {"t_start": t_start, "t": time.time(), "text": r["text"].strip(), "audio_s": r["audio_s"],
                       "pcm": np.concatenate(got)}


class Mic:
    """The microphone as one crew channel for a scenario: listens in the background; the
    scenario takes lines with next_line() and mutes it while it plays audio with play()."""

    def __init__(self):
        self.stop, self.muted = asyncio.Event(), threading.Event()  # muted is set from other threads too
        self.q: asyncio.Queue = asyncio.Queue()
        self.task = None

    async def __aenter__(self):
        async def run():
            async for line in lines(self.stop, muted=self.muted):
                await self.q.put(line)
        self.task = asyncio.create_task(run())
        await asyncio.sleep(1.2)  # the noise floor is measured over the first second
        return self

    async def __aexit__(self, *exc):
        self.stop.set()
        try:
            await asyncio.wait_for(self.task, 15)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self.task.cancel()

    def flush(self) -> None:
        """Forget lines heard before now (talk while the simulator was setting up)."""
        while not self.q.empty():
            self.q.get_nowait()

    async def next_line(self, timeout: float) -> dict | None:
        try:
            return await asyncio.wait_for(self.q.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def play(self, pcm: np.ndarray, sr: int) -> None:
        """Play through the speakers with the microphone muted, and wait until it has finished."""
        import pathlib
        import subprocess
        import tempfile
        import soundfile as sf
        path = pathlib.Path(tempfile.mkstemp(suffix=".wav")[1])
        sf.write(path, pcm, sr)
        self.muted.set()
        try:
            p = await asyncio.create_subprocess_exec("afplay", str(path))
            await p.wait()
            await asyncio.sleep(0.3)  # room echo
        finally:
            self.muted.clear()


    def speak(self, text: str) -> None:
        """Crosscheck's callout (voice/tts.py) with the microphone muted: for Crosscheck(speak=...),
        which calls it from the monitoring thread; plays in its own thread, like tts.speak."""
        from voice.tts import speak

        def run():
            self.muted.set()
            try:
                speak(text, block=True)
                time.sleep(0.3)
            finally:
                self.muted.clear()
        threading.Thread(target=run, daemon=True).start()


async def _main():
    from dotenv import load_dotenv
    load_dotenv()
    stop = asyncio.Event()
    print("listening (Ctrl-C to stop)")
    async for line in lines(stop):
        print(f"{line['t'] - line['t_start']:4.1f} s  {line['text']}")


if __name__ == "__main__":
    asyncio.run(_main())
