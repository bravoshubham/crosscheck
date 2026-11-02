"""Live tablet display: follows whichever recording is being written and streams it to a page.

  uv run python cli.py live              # http://localhost:8770 (or the Mac's address, from an iPad)

The monitoring loop already writes every event to replay/<scenario>/events.jsonl. This server
tails the most recently written one and pushes each new line to the page as a server-sent
event; it never talks to the loop, so the display cannot slow it down or change what it does.
"""
import asyncio
import json
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPLAY = ROOT / "replay"
DIRS = (REPLAY, ROOT / "demo_runs")  # recorded scenarios, and live demos (demo.py)
PAGE = pathlib.Path(__file__).resolve().parent / "live.html"

app = FastAPI()


def newest() -> pathlib.Path | None:
    logs = [p for d in DIRS for p in d.glob("*/events.jsonl")]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


@app.get("/")
def page():
    return FileResponse(PAGE)


@app.get("/frame/{scenario}/{name}")
def frame(scenario: str, name: str):
    for d in DIRS:
        p = d / scenario / "frames" / name
        # the frames folder of a re-enactment is a link to the recording's: check where it really points
        if p.is_file() and any(base.resolve() in p.resolve().parents for base in DIRS):
            return FileResponse(p, headers={"Cache-Control": "no-store"})
    raise HTTPException(404)


@app.get("/stream")
async def stream():
    async def gen():
        log, pos = None, 0
        while True:
            cur = newest()
            if cur != log or (cur and cur.stat().st_size < pos):  # another scenario started, or the file was rewritten
                log, pos = cur, 0
                if log:
                    meta = log.parent / "meta.json"
                    yield "event: reset\ndata: " + json.dumps({"scenario": log.parent.name, "meta": json.loads(meta.read_text()) if meta.exists() else {}}) + "\n\n"
            if log:
                with log.open() as f:
                    f.seek(pos)
                    chunk = f.read()
                complete = chunk[:chunk.rfind("\n") + 1]  # only whole lines; the writer may be mid-line
                pos += len(complete.encode())
                for line in complete.splitlines():
                    if line.strip():
                        yield f"data: {line}\n\n"
            yield ": keepalive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


def serve(host: str = "0.0.0.0", port: int = 8770) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
