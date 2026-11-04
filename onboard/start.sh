#!/bin/sh
# Start the on-board services and check them: the hearing service (two Meta recognizers) on
# 127.0.0.1:8771, and Muse Glimmer 30B in Ollama on 127.0.0.1:11434. See onboard/README.md.
cd "$(dirname "$0")/.." || exit 1
OMNI_ENV=${OMNI_ENV:-$HOME/llm-stack/omniasr}
if ! curl -s -m 2 http://127.0.0.1:8771/health | grep -q '"ok": true'; then
  FAIRSEQ2_USER_ASSET_DIR=$OMNI_ENV/cards DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib HF_HUB_OFFLINE=1 \
    nohup "$OMNI_ENV/.venv/bin/python" onboard/hearing_service.py > /tmp/crosscheck-hearing.log 2>&1 &
  for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8771/health | grep -q '"ok": true' && break; sleep 2; done
fi
curl -s -m 2 http://127.0.0.1:8771/health || { echo "hearing service did not start: /tmp/crosscheck-hearing.log"; exit 1; }
echo
curl -s -m 5 http://127.0.0.1:11434/api/tags | grep -q 'muse-glimmer' || { echo "Ollama with muse-glimmer:30b is not running"; exit 1; }
curl -s -m 120 http://127.0.0.1:11434/api/chat -d '{"model":"muse-glimmer:30b","stream":false,"think":false,"keep_alive":"60m","messages":[{"role":"user","content":"Reply OK"}],"options":{"num_predict":3}}' > /dev/null \
  && echo "muse-glimmer:30b loaded"
