#!/usr/bin/env bash
# ===========================================================================
# H100 runbook - run top-to-bottom once the VM is up, then SHUT IT DOWN.
# Everything here REQUIRES the real Qwen3-30B endpoint; all dev/debug was
# already done off-GPU so this session is pure execution.
#
# PREREQS (do these first, model not yet loaded):
#   git clone https://github.com/oreng72/nebius_ho_mlops.git && cd nebius_ho_mlops
#   uv sync                                  # Linux: installs vLLM too
#   cp .env.example .env                     # then edit .env:
#     VLLM_BASE_URL=http://localhost:8000/v1
#     VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
#     OPENAI_API_KEY=EMPTY
#     LANGFUSE_PUBLIC_KEY=pk-lf-course-public
#     LANGFUSE_SECRET_KEY=sk-lf-course-secret
#     LANGFUSE_HOST=http://localhost:3001
#   uv run python scripts/load_data.py       # BIRD; if Beijing URL stalls, use HF mirror
#   docker compose up -d                     # Prometheus/Grafana/Langfuse (Langfuse keys auto-seeded)
# Forward ports from your laptop: 3000 9090 3001 8000 8001
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

wait_healthy () {  # wait_healthy <url> <name>
  echo "waiting for $2 ..."
  until curl -sf "$1" >/dev/null 2>&1; do sleep 3; done
  echo "$2 up."
}

mkdir -p results screenshots

# --- Phase 1: serve the model -----------------------------------------------------
echo "== Phase 1: starting vLLM (first run pulls ~60GB weights) =="
./scripts/start_vllm.sh > vllm.log 2>&1 &
wait_healthy "http://localhost:8000/v1/models" "vLLM"

echo "-- Phase 2 check: confirm metric names match serving.json --"
curl -s http://localhost:8000/metrics | grep -E "^vllm:(gpu_cache_usage_perc|prefix_cache|num_preemptions|num_requests|e2e_request_latency|request_(prefill|decode|queue)_time|time_to_first_token|time_per_output_token|prompt_tokens|generation_tokens|request_success)" | sort -u || true
echo "-- if any names above differ from serving.json exprs, patch the dashboard --"

echo "-- manual sanity query -> screenshot screenshots/vllm_manual_query.png --"
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"Qwen/Qwen3-30B-A3B-Instruct-2507",
  "messages":[{"role":"user","content":"Write one SQLite query that lists the 5 largest tables by row count."}],
  "max_tokens":256,"temperature":0}' | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# --- start the agent server -------------------------------------------------------
echo "== starting agent server =="
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001 > agent.log 2>&1 &
wait_healthy "http://localhost:8001/health" "agent"

# --- Phase 4: fire 10 TAGGED questions (populates Langfuse with filterable metadata) ---
echo "== Phase 4: fire 10 tagged questions (-> Langfuse traces) =="
uv run python - <<'PY'
import json, urllib.request
rows = [json.loads(l) for l in open("evals/eval_set.jsonl")][:10]
for i, r in enumerate(rows, 1):
    body = json.dumps({"question": r["question"], "db": r["db_id"],
                       "tags": {"phase": "phase4", "split": "baseline", "db": r["db_id"]}}).encode()
    req = urllib.request.Request("http://localhost:8001/answer", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=180).read())
        print(f"[{i:2}/10] {r['db_id']:24} iters={d.get('iterations')} ok={d.get('ok')}")
    except Exception as e:
        print(f"[{i:2}/10] {r['db_id']:24} ERROR {type(e).__name__}: {e}")
PY
echo "-- open http://localhost:3001 -> Traces: screenshot langfuse_tags.png (list w/ tags)"
echo "-- open a 2-iteration trace: screenshot langfuse_trace.png (generate->verify->revise waterfall)"

# --- Phase 5: baseline eval (watch Grafana while it runs) --------------------------
echo "== Phase 5: baseline eval (full 30) -> watch Grafana, screenshot grafana_eval_run.png =="
uv run python evals/run_eval.py --out results/eval_baseline.json

# --- Phase 6: SLO load test (watch Grafana -> grafana_serving.png, then before/after) ---
echo "== Phase 6: load test @ 10 RPS for 5 min =="
uv run python load_test/driver.py --rps 10 --duration 300

echo "== runbook done. =="
echo "Now: Phase 6 tuning loop (edit start_vllm.sh ONE lever, restart vLLM, re-run driver,"
echo "log 'saw X->changed Z->result W' + before/after screenshots), then:"
echo "  uv run python evals/run_eval.py --out results/eval_after_tuning.json"
echo "Fill REPORT.md numbers, then SHUT DOWN THE VM."
