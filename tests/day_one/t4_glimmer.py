"""T4: Glimmer makes one decision in under 3 s on the M5 Max (on-board mode).

Findings (2 Nov): with tool calling, Glimmer spends ~140-270 hidden tokens before the call
(6-17 s with X-Plane sharing the GPU), even with think=False. A JSON-schema reply with
worked examples cuts output to ~30 tokens. Standard callout wording comes from templates,
the model chooses the action.

Run with Ollama up:  uv run python tests/day_one/t4_glimmer.py
"""
import json, time, httpx, sys, re
model = sys.argv[1] if len(sys.argv) > 1 else "muse-glimmer:30b"
schema = {"type": "object", "properties": {
    "action": {"type": "string", "enum": ["speak_callout", "stay_silent", "read_checklist", "draft_radio_call"]},
    "text": {"type": "string", "maxLength": 60}}, "required": ["action", "text"]}
sys_p = ("You are Crosscheck, a pilot-monitoring assistant. A deterministic rule has flagged something. "
         "Decide the single best action and reply with JSON only.\n"
         "speak_callout: a mismatch the crew must hear now; text states both values in standard phraseology, 2-6 words.\n"
         "stay_silent: the flag is already resolved or not worth interrupting.\n"
         "read_checklist: an emergency needs a checklist; text is the checklist name.\n"
         "draft_radio_call: ATC must be told; text is the radio call.")
shots = [("FLAG callout_vs_set: called flaps 15, lever at 5.", {"action": "speak_callout", "text": "Flaps fifteen called, five set."}),
         ("FLAG clearance_vs_set: cleared FL240, MCP 24000. RESOLVED 2 s ago.", {"action": "stay_silent", "text": ""}),
         ("FLAG engine_failure: N1 engine 2 below 20%, single-pilot no.", {"action": "read_checklist", "text": "Engine failure"})]
tests = [("FLAG clearance_vs_set: cleared FL240, MCP 25000.", "speak_callout"),
         ("FLAG readback_vs_clearance: cleared FL240, readback FL250.", "speak_callout"),
         ("FLAG mode_vs_phase: final 800 ft, autothrottle HOLD, speed 128 target 140.", "speak_callout"),
         ("FLAG callout_vs_set: called gear up, gear lever up. RESOLVED.", "stay_silent")]
msgs0 = [{"role": "system", "content": sys_p}]
for u, a in shots: msgs0 += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(a)}]
ok = 0; times = []
with httpx.Client(timeout=300) as c:
    for u, want in tests * 2:
        t = time.time()
        r = c.post("http://127.0.0.1:11434/api/chat", json={"model": model, "messages": msgs0 + [{"role": "user", "content": u}], "format": schema, "think": False, "stream": False, "keep_alive": "30m", "options": {"temperature": 0}}).json()
        dt = time.time() - t; times.append(dt)
        raw = r["message"]["content"]; d = json.loads(re.sub(r"<\|[a-z_]+\|>\s*$", "", raw.strip()))
        ok += d["action"] == want
        print(f"{dt:.2f}s {d} {'OK' if d['action']==want else 'WRONG (want '+want+')'}")
summary = {"model": model, "correct": ok, "runs": len(times), "median_s": round(sorted(times)[len(times)//2], 2), "max_s": round(max(times), 2), "pass": ok == len(times) and max(times) < 3}
print(summary)
import pathlib
out = pathlib.Path(__file__).resolve().parents[2] / "eval" / "results" / "day_one"
out.mkdir(parents=True, exist_ok=True)
(out / "t4_results.json").write_text(json.dumps(summary, indent=1))
