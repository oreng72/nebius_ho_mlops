#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Phase 1 - vLLM serving config for Qwen3-30B-A3B-Instruct-2507 on 1x H100 80GB
#
# Workload this config is tuned for:
#   - prompts:  1.5-3K tokens (SQL schema + question)
#   - outputs:  short, structured SQL (~50-300 tokens)
#   - shape:    2-3 dependent vLLM calls per agent run, same schema reused
#   - SLO:      P95 end-to-end agent latency < 5s @ 10+ RPS (~20-30 vLLM req/s)
#
# Central constraint: Qwen3-30B-A3B is an MoE - 30B total params (~61GB in BF16)
# but only ~3B active/token. On an 80GB H100 the weights eat most of VRAM, so
# concurrency is KV-cache-bound, not compute-bound. That drives the flags below
# and makes FP8 weights the #1 Phase-6 lever.
# ---------------------------------------------------------------------------

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${VLLM_PORT:-8000}"

# Speed up the first ~60GB weight pull from HF.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

exec uv run vllm serve "$MODEL" \
  --served-model-name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 96 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --disable-log-requests

# ---------------------------------------------------------------------------
# Why each non-default flag (Phase 1 rationale; mirror these in REPORT.md):
#   --dtype bfloat16           baseline precision; FP8 weights held as Phase-6 lever
#   --max-model-len 8192       prompts+output fit in 8K; smaller ctx => more KV slots => more concurrency
#   --gpu-memory-utilization 0.92  claw back VRAM for KV without OOM on the 80GB card
#   --max-num-seqs 96          cap concurrent sequences near the real concurrency target
#   --max-num-batched-tokens 8192  per-step token budget; bounds how much prefill starves decode
#   --enable-chunked-prefill   slice 3K-token prefills so in-flight decodes aren't blocked (protects TTFT/TPOT)
#   --enable-prefix-caching    schema prefix repeats across 2-3 calls/run -> cache it, skip re-prefill
#   --kv-cache-dtype fp8       halve KV bytes/token -> ~2x sequences fit in the tight KV budget
#   --tensor-parallel-size 1   single H100, no sharding
#   --disable-log-requests     drop per-request logging overhead under load
#
# First levers to pull in Phase 6 if you miss the SLO (change ONE at a time):
#   KV util ~1.0 / preemptions>0 / queue building -> FP8 WEIGHTS:
#       MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8   (or add --quantization fp8)
#       weights ~61->~30GB frees ~30GB for KV -> much higher concurrency, and
#       H100 FP8 tensor cores speed prefill. RE-RUN EVALS to confirm quality survived.
#   decode latency (TPOT) spiking under load -> lower --max-num-batched-tokens (e.g. 4096)
#   KV still tight after FP8 -> lower --max-model-len toward real p99 len, or raise gpu-util to 0.94
#   throughput short but KV has headroom -> raise --max-num-seqs
# ---------------------------------------------------------------------------
