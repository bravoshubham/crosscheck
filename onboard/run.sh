#!/bin/sh
# Run a Crosscheck command on board: every model on this computer, and a macOS sandbox that
# refuses any network connection that is not to this computer (onboard/no_network.sb).
#   onboard/run.sh eval/scenarios/altitude_bust.py bust
# Needs the local services first: onboard/start.sh
cd "$(dirname "$0")/.." || exit 1
export CROSSCHECK_ONBOARD=1 CROSSCHECK_SANDBOX=onboard/no_network.sb HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec sandbox-exec -f onboard/no_network.sb .venv/bin/python "$@"
