"""On-board mode: every model runs on this computer, and nothing leaves it.

CROSSCHECK_ONBOARD=1 switches the senses from the Meta Model API to Meta open weights served on
the loopback interface:

  hearing   onboard/hearing_service.py (SeamlessM4T v2 + Omnilingual ASR 7B)   127.0.0.1:8771
  seeing    Muse Glimmer 30B with vision, through Ollama or llama.cpp           127.0.0.1:11434
  thinking  the same Glimmer: the second reader of radio lines, and the sentence under a callout

`onboard/run.sh` starts a scenario this way under a macOS sandbox that refuses every network
connection that is not to this computer, so a run proves it needed nothing else.
"""
import os

import httpx

ONBOARD = os.environ.get("CROSSCHECK_ONBOARD") == "1"
HEARING = os.environ.get("HEARING_URL", "http://127.0.0.1:8771")
GLIMMER_HOST = os.environ.get("GLIMMER_HOST", "http://127.0.0.1:11434")
GLIMMER_MODEL = os.environ.get("GLIMMER_MODEL", "muse-glimmer:30b")
ONBOARD_SPEECH = ("on board: SeamlessM4T v2 and Omnilingual ASR 7B on this computer (onboard/hearing_service.py), "
                  "values kept when the two agree, the local Muse Glimmer as second reader; parsed by senses/speech.py")


def local_client(timeout: float = 120) -> httpx.AsyncClient:
    """A client that never uses a proxy: every on-board call is to 127.0.0.1."""
    return httpx.AsyncClient(trust_env=False, timeout=timeout)


async def glimmer_json(prompt: str, schema: dict, image_b64: str | None = None, num_predict: int = 200,
                       client: httpx.AsyncClient | None = None) -> tuple[str, dict]:
    """One local Glimmer call held to a JSON schema, thinking off. Returns (reply text, counters)."""
    msg = {"role": "user", "content": prompt, **({"images": [image_b64]} if image_b64 else {})}
    body = {"model": GLIMMER_MODEL, "stream": False, "think": False, "keep_alive": "60m", "format": schema,
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 4096}, "messages": [msg]}
    own = client is None
    client = client or local_client()
    try:
        r = await client.post(GLIMMER_HOST + "/api/chat", json=body)
        r.raise_for_status()
    finally:
        if own:
            await client.aclose()
    d = r.json()
    text = (d["message"].get("content") or "").replace("<|eot|>", "")
    return text, {k: d.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration")}


def camera_check(frame, kind: str = "pfd") -> dict:
    """The on-board pre-flight camera check: SAM 3.1's open weights are gated, so instead the local
    Glimmer reads the calibrated crop once, and the camera counts as right if values came back
    from at least two of three different places on the display: speed, selected altitude, thrust
    mode (a moved camera or a window in front gives nulls everywhere). Glimmer sometimes leaves
    one value unread on a good frame, and the altitude's rolling digits are left out for that reason."""
    import asyncio
    import concurrent.futures
    if kind == "flaps":
        from senses.flaps_glimmer import read_flaps_glimmer as read
    else:
        from senses.panel_glimmer import read_frame_glimmer as read
    with concurrent.futures.ThreadPoolExecutor(1) as ex:  # callers are inside an event loop already
        r, _ = ex.submit(lambda: asyncio.run(read(frame))).result()
    values = {"flaps": r.flaps} if kind == "flaps" else {
        "ias": r.ias, "selected_altitude": r.selected_altitude, "thrust_mode": r.fma[0] or None}
    ok = sum(v is not None for v in values.values()) >= (1 if kind == "flaps" else 2)
    return {"ok": ok, "found": [], "matches": {kind: {"read": values, "ok": ok}}, "model": "muse-glimmer-30b"}


def network_proof() -> dict:
    """Evidence for the run's record that nothing could leave this computer: a connection to the
    Meta Model API is attempted and must be refused."""
    import socket
    proof = {"sandbox": os.environ.get("CROSSCHECK_SANDBOX"), "onboard": ONBOARD}
    try:
        socket.create_connection(("api.meta.ai", 443), timeout=3).close()
        proof["api.meta.ai:443"] = "connected"
    except OSError as e:
        proof["api.meta.ai:443"] = f"refused: {type(e).__name__}"
    return proof
