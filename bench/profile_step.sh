set -u
H="Authorization: Bearer ${SGLANG_API_KEY:-}"
P=${1:?port}; C=${2:?container}
docker exec ${C} sh -c 'rm -f /tmp/*.trace.json.gz /tmp/*.trace.json 2>/dev/null; ls /tmp'
curl -s -X POST -H "$H" -H 'Content-Type: application/json' http://127.0.0.1:$P/start_profile -d '{"num_steps": 40, "activities": ["CPU","GPU"], "profile_by_stage": false}'; echo
python3 - "$P" <<'PY'
import json, sys, urllib.request, time
port = sys.argv[1]
body = {"model":"qwen3.8-27b","stream":True,"max_tokens":400,"temperature":0.8,
 "messages":[{"role":"user","content":f"prof-{time.time_ns()} Write a detailed walkthrough of a mill cellar search: ten short paragraphs, no list, keep going."}],
 "chat_template_kwargs":{"enable_thinking":True}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
  headers={"Content-Type":"application/json","Authorization":"Bearer "+__import__("os").environ.get("SGLANG_API_KEY","")})
n=0
with urllib.request.urlopen(req, timeout=300) as r:
    for line in r: n+=1
print("chunks", n)
PY
sleep 20
docker exec ${C} sh -c 'ls -la /tmp/*trace* 2>/dev/null'
