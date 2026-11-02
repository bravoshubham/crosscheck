"""Crosscheck command line.

  uv run python cli.py replay replay/altitude_bust        # one recording, saved panel readings
  uv run python cli.py replay --all                       # every recording under replay/
  uv run python cli.py replay --all --recompute           # re-read every saved frame (macOS)
  uv run python cli.py debrief                            # build the debrief viewer into site/
  uv run python cli.py live                               # tablet display of the run in progress (:8770)

Replay feeds the panel readings and spoken ("said") facts saved with each recording back
through the same ledger, rules and scorer as the live run, then compares the result with the
one recorded live. Needs no simulator, GPU or API key.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from brain.live import replay  # noqa: E402
from eval.scoring import score  # noqa: E402

COMPARE = ("pass", "callout", "callouts", "callout_s_after_fault", "false_callouts_before_fault",
           "false_callouts_before_clearance")


def replay_one(rec: pathlib.Path, recompute: bool) -> bool:
    replay(rec, rec, recompute=recompute)
    got = score(rec, "replay_events.jsonl")
    live = json.loads((rec / "result.json").read_text()) if (rec / "result.json").exists() else {}
    meta = json.loads((rec / "meta.json").read_text())
    same = all(got.get(k) == live.get(k) for k in COMPARE if k in live)
    (rec / "replay_result.json").write_text(json.dumps(got, indent=1))
    mode = "re-read frames" if recompute else "saved readings"
    print(f"REPLAY {meta['scenario']} ({mode}; speech: {meta.get('speech', 'n/a')}; recorded {meta.get('recorded', '?')})")
    print(f"  pass: {got['pass']}   matches live run: {same}")
    for k in COMPARE[1:]:
        if k in got and got[k] not in (None, [], ""):
            print(f"  {k}: {got[k]}")
    if not same:
        for k in COMPARE:
            if k in live and got.get(k) != live.get(k):
                print(f"  DIFFERS {k}: live={live.get(k)} replay={got.get(k)}")
    return got["pass"] and same


def main() -> int:
    ap = argparse.ArgumentParser(prog="crosscheck")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="re-run recorded scenarios without the simulator")
    rp.add_argument("dirs", nargs="*", type=pathlib.Path)
    rp.add_argument("--all", action="store_true")
    rp.add_argument("--recompute", action="store_true", help="re-read the saved frames (macOS: Apple Vision)")
    sub.add_parser("debrief", help="build the debrief viewer (site/index.html) from replay/")
    lv = sub.add_parser("live", help="tablet display of the scenario being recorded")
    lv.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    if args.cmd == "live":
        from ui.server import serve
        print(f"Crosscheck live display on http://localhost:{args.port}")
        serve(port=args.port)
        return 0
    if args.cmd == "debrief":
        from debrief.build import build
        print("wrote", build())
        return 0
    dirs = sorted(p.parent for p in (ROOT / "replay").glob("*/meta.json")) if args.all else args.dirs
    if not dirs:
        ap.error("give recording directories or --all")
    ok = [replay_one(d, args.recompute) for d in dirs]
    print(f"\n{sum(ok)}/{len(ok)} recordings pass and match their live runs")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
