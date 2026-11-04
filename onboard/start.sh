#!/bin/sh
# Start the on-board services and check them: the hearing service (two Meta recognizers) on
# 127.0.0.1:8771, the pilot watch's pose service (SAM 3D Body) on 127.0.0.1:8772, and Muse
# Glimmer 30B in Ollama on 127.0.0.1:11434. See onboard/README.md.
#   onboard/start.sh                 # all three
#   onboard/start.sh pose            # only the ones named: hearing, pose, glimmer
cd "$(dirname "$0")/.." || exit 1
WANT=" ${*:-hearing pose glimmer} "
want() { case "$WANT" in *" $1 "*) return 0 ;; esac; return 1; }
ok() { curl -s -m 2 "$1/health" | grep -q '"ok": true'; }

if want hearing; then
  OMNI_ENV=${OMNI_ENV:-$HOME/llm-stack/omniasr}
  if ! ok http://127.0.0.1:8771; then
    FAIRSEQ2_USER_ASSET_DIR=$OMNI_ENV/cards DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib HF_HUB_OFFLINE=1 \
      nohup "$OMNI_ENV/.venv/bin/python" onboard/hearing_service.py > /tmp/crosscheck-hearing.log 2>&1 &
    for i in $(seq 1 60); do ok http://127.0.0.1:8771 && break; sleep 2; done
  fi
  ok http://127.0.0.1:8771 || { echo "hearing service did not start: /tmp/crosscheck-hearing.log"; exit 1; }
  curl -s -m 2 http://127.0.0.1:8771/health; echo
fi

if want pose; then
  SAM3DB_ENV=${SAM3DB_ENV:-$HOME/llm-stack/sam3db}
  if ! ok http://127.0.0.1:8772; then
    HF_HUB_OFFLINE=1 nohup "$SAM3DB_ENV/.venv/bin/python" onboard/pose_service.py > /tmp/crosscheck-pose.log 2>&1 &
    for i in $(seq 1 60); do ok http://127.0.0.1:8772 && break; sleep 2; done
  fi
  ok http://127.0.0.1:8772 || { echo "pose service did not start: /tmp/crosscheck-pose.log"; exit 1; }
  curl -s -m 2 http://127.0.0.1:8772/health; echo
fi

if want glimmer; then
  curl -s -m 5 http://127.0.0.1:11434/api/tags | grep -q 'muse-glimmer' || { echo "Ollama with muse-glimmer:30b is not running"; exit 1; }
  curl -s -m 120 http://127.0.0.1:11434/api/chat -d '{"model":"muse-glimmer:30b","stream":false,"think":false,"keep_alive":"60m","messages":[{"role":"user","content":"Reply OK"}],"options":{"num_predict":3}}' > /dev/null \
    && echo "muse-glimmer:30b loaded"
fi
