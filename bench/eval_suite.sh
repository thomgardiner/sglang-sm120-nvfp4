#!/usr/bin/env bash
set -u
GPU=$1; TAG=$2
D=docker
LM=lm_eval
PORT=$((30000 + GPU)); C=sglang-qwen$GPU
cd "$SGLANG_LAUNCHER_DIR"
python3 - <<'PY'
p="worker.py"; s=open(p).read()
if "DEFAULT_CHAT_TEMPLATE_KWARGS" not in s:
    old='''    fp4_backend = spec.get("FP4_GEMM_BACKEND")'''
    new='''    ctk = spec.get("DEFAULT_CHAT_TEMPLATE_KWARGS")
    if ctk:
        cmd.extend(["--default-chat-template-kwargs", ctk])
    fp4_backend = spec.get("FP4_GEMM_BACKEND")'''
    assert old in s; open(p,"w").write(s.replace(old,new)); print("worker.py: chat template kwargs key added")
PY
grep -v '^DEFAULT_CHAT_TEMPLATE_KWARGS' "specs/eval-$TAG.env" > "specs/eval-$TAG-nothink.env"
echo 'DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking": false}' >> "specs/eval-$TAG-nothink.env"
python3 worker.py --gpu $GPU --spec "specs/eval-$TAG-nothink.env" --dry-run | tr ' ' '\n' | grep -A1 "default-chat-template-kwargs" | tr '\n' ' '; echo
sudo -n systemctl stop sglang-qwen@$GPU.service
$D stop -t 20 $C >/dev/null 2>&1; sleep 3
for i in $(seq 1 30); do ss -ltn | grep -qE ":($PORT|8997) " || break; sleep 2; done
nohup python3 worker.py --gpu $GPU --spec "specs/eval-$TAG-nothink.env" > "./gpu$GPU-nothink-$TAG.log" 2>&1 < /dev/null &
for i in $(seq 1 90); do
  sleep 10
  code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${SGLANG_API_KEY:-}" http://127.0.0.1:$PORT/v1/models || true)
  [ "$code" = "200" ] && { echo "[$TAG] UP after $((i*10))s"; break; }
  $D ps --format '{{.Names}}' | grep -q "$C" || { echo "[$TAG] CONTAINER GONE"; tail -20 "./gpu$GPU-nothink-$TAG.log"; exit 1; }
done
cd "$EVAL_WORK_DIR"
probe=$(curl -s -H "Authorization: Bearer ${SGLANG_API_KEY:-}" -H "Content-Type: application/json" http://127.0.0.1:$PORT/v1/chat/completions -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":64}')
echo "[$TAG] probe: $(echo "$probe" | python3 -c 'import json,sys; d=json.load(sys.stdin)["choices"][0]["message"]; print("content=",repr(d.get("content")), "reasoning_len=", len(d.get("reasoning_content") or ""))')"
MA="model=qwen3.8-27b,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=8,max_retries=3,tokenized_requests=False"
for spec in "gsm8k:4096:" "minerva_math500:4096:" "gpqa_diamond_cot_zeroshot:4096:" "ifeval:2048:" "humaneval_instruct:2048:--confirm_run_unsafe_code"; do
  IFS=: read -r task toks extra <<<"$spec"
  gen='temperature=0'
  [ "$task" = humaneval_instruct ] && gen='temperature=0,until=["\n```"]'
  echo "[$TAG] $task start $(date -u +%H:%M:%S)"
  HF_ALLOW_CODE_EVAL=1 OPENAI_API_KEY=${SGLANG_API_KEY:-} $LM --model local-chat-completions --model_args "$MA,max_gen_toks=$toks" \
    --tasks "$task" --num_fewshot 0 --apply_chat_template --gen_kwargs "$gen" $extra \
    --output_path "./lmeval/nothink-$TAG" --log_samples > "./lmeval_nothink_${TAG}_${task}.log" 2>&1
  grep -A8 "|Tasks" "./lmeval_nothink_${TAG}_${task}.log" | grep "|" | grep -v "Tasks\|---" | head -6 | sed "s/^/[$TAG] /"
done
python3 "$(dirname "$0")/he_grade.py" "$TAG"
$D stop -t 20 $C >/dev/null 2>&1
sudo -n systemctl start sglang-qwen@$GPU.service
echo "NOTHINK$GPU DONE $(date -u +%H:%M:%S)"
