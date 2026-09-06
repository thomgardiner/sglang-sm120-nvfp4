#!/usr/bin/env bash
set -u
GPU=1; C=sglang-qwen$GPU; PORT=$((30000 + GPU))
cd "$SGLANG_LAUNCHER_DIR"
serve() {
  spec=$1
  sudo -n systemctl stop sglang-qwen@$GPU.service 2>/dev/null
  docker stop -t 20 $C >/dev/null 2>&1; sleep 3
  for i in $(seq 1 30); do ss -ltn | grep -qE ":$PORT " || break; sleep 2; done
  nohup python3 worker.py --gpu $GPU --spec "$spec" > "./gpu$GPU-dt.log" 2>&1 < /dev/null &
  for i in $(seq 1 90); do
    sleep 10
    code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${SGLANG_API_KEY:-}" http://127.0.0.1:$PORT/v1/models || true)
    [ "$code" = "200" ] && return 0
    docker ps --format '{{.Names}}' | grep -q "$C" || return 1
  done
  return 1
}
for n in 4 8 12 16; do
  sed "/^SPECULATIVE_NUM_DRAFT_TOKENS=/d" specs/qwen3.8-27b.env > specs/dt.env
  echo "SPECULATIVE_NUM_DRAFT_TOKENS=$n" >> specs/dt.env
  if serve specs/dt.env; then
    python3 ./step_time.py $PORT "draft-tokens-$n" "./step-dt-$n.json" 2>&1 | grep -v Traceback
  else
    echo "[dt=$n] FAILED: $(grep -iE 'error|assert|ValueError' ./gpu$GPU-dt.log | head -2 | cut -c1-160)"
  fi
done
docker stop -t 20 $C >/dev/null 2>&1
sudo -n systemctl start sglang-qwen@$GPU.service
echo "DRAFT TOKEN SWEEP DONE $(date -u +%H:%M:%S)"
