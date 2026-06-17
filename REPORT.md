# LLM inference + o11y — Report

Text-to-SQL PoC serving **Qwen3-30B-A3B-Instruct-2507** on 1× H100 80 GB, with a
LangGraph verify→revise agent, Prometheus/Grafana serving observability, and
Langfuse agent tracing. SLO: **P95 end-to-end *agent* latency < 5 s at ≥ 10 RPS**
(1 RPS = one full agent run = 2–3 vLLM calls) over a 5-minute window. All numbers
below are from the real 30B on the H100 (run 2026-06-17).

**Headline:** the SLO is **missed**, and the diagnosis is the interesting part — the
bottleneck is **not** the GPU (which sits ~6 % KV-utilized under load) but the agent
orchestration layer. A single config change cut P95 latency **7.7×** (68.9 s → 9.0 s)
and confirmed the root cause, though full SLO compliance needs a second iteration.

---

## 1. Serving configuration (Phase 1)

**Constraint that drives everything:** Qwen3-30B-A3B is an MoE — ~30 B total params
(≈ 61 GB in BF16) but only ~3 B active per token. On an 80 GB H100 the weights consume
most of VRAM, so *a priori* concurrency looked **KV-cache-bound, not compute-bound**.
The config trades context length and precision for KV headroom. (Phase 6 showed the
real bottleneck was upstream of vLLM entirely — see §6.)

| Flag | Value | Why (for *this* workload) |
|---|---|---|
| `--dtype` | `bfloat16` | Baseline precision; FP8 weights held as a Phase-6 lever. |
| `--max-model-len` | `8192` | Prompts (1.5–3 K) + short SQL output fit in 8 K; smaller context ⇒ more KV slots ⇒ more concurrency. |
| `--gpu-memory-utilization` | `0.92` | Claw back VRAM for KV without OOM risk on the 80 GB card. |
| `--max-num-seqs` | `96` | Cap concurrent sequences near the concurrency target. |
| `--max-num-batched-tokens` | `8192` | Per-step token budget; bounds how much a big prefill starves in-flight decode. |
| `--enable-chunked-prefill` | on | Slice 3 K-token prefills so decodes aren't blocked → protects TTFT/TPOT under load. |
| `--enable-prefix-caching` | on | The DB schema prefix repeats across the 2–3 calls per agent run → cache it, skip re-prefill. |
| `--kv-cache-dtype` | `fp8` | Halve KV bytes/token → ~2× sequences fit in the KV budget. |
| `--tensor-parallel-size` | `1` | Single H100, no sharding. |
| `--disable-log-requests` | on | Drop per-request logging overhead under load. |

Full script + per-flag rationale: `scripts/start_vllm.sh`. Manual sanity check passed
(model loads, returns sensible SQL for sample questions on `:8000`).

> **Note on two screenshots.** `screenshots/vllm_manual_query.png` (this Phase-1 manual
> query) and `screenshots/grafana_eval_run.png` (Phase 5, dashboard during the baseline
> eval) are absent: the single time-boxed H100 session was torn down to contain GPU cost
> before those two captures were taken, and the VM is gone. Both underlying runs did
> happen — the manual query is the `scripts/h100_runbook.sh` sanity step that gated the
> rest of the session, and the eval ran to the `results/eval_baseline.json` numbers below;
> the dashboard's under-load behavior is shown in `grafana_serving.png` / `grafana_before.png`.
> I'm flagging the gap rather than substituting a mislabeled image.

---

## 2. Observability dashboard (Phase 2)

Grafana dashboard **`vLLM Serving Health`** (`infra/grafana/provisioning/dashboards/serving.json`,
uid `vllm-serving`), built from vLLM's v1 `/metrics` (`vllm:` names — confirmed against
the live endpoint, no patch needed). Four at-a-glance SLO tiles (P95 e2e latency, RPS,
KV utilization, queue depth) over three sections:

- **Latency — "is it slow, and *where* in the lifecycle?"** e2e p50/p95/p99, a stacked
  queue → prefill → decode breakdown, plus TTFT and TPOT (inter-token).
- **Throughput.** Prompt vs generation tokens/s, request rate by finish reason,
  running-vs-waiting concurrency.
- **KV cache.** `gpu_cache_usage_perc` (warn/crit lines), prefix-cache hit rate, preemptions/s.

The panels visibly react under load — `screenshots/grafana_before.png` shows e2e latency
spiking to ~2 min during the overload runs **while TTFT (p95 89 ms), TPOT (p95 25 ms), and
KV (≈ 6 %) stay flat** — i.e. the dashboard answers its own question: the time is spent
*queueing*, not in prefill/decode. That single read is what drove the Phase-6 diagnosis.

---

## 3. Agent design (Phase 3)

LangGraph graph (`agent/graph.py`):

```
attach_schema → generate_sql → execute → verify ──ok──▶ END
                                  ▲          │
                                  └─ revise ◀┘ (not ok, iteration < MAX)
```

- **generate_sql / revise** call the model via `langchain_openai.ChatOpenAI`; SQL is
  pulled from the ```` ```sql ```` block.
- **verify** short-circuits a hard execution error (a crashed query never plausibly
  answers), otherwise asks the model for a strict `{ok, issue}` JSON judged against the
  question — flagging SQL errors, empty results when rows are implied, wrong column
  shape, or an ignored condition. Parsed defensively (default ok on bad JSON so a flaky
  reply can't trap the loop).
- **route_after_verify** ends on `verify_ok` or at `MAX_ITERATIONS = 3`, else revises.

Served at `POST /answer` (`agent/server.py`), returning final SQL, rows, iteration count,
and per-node `history`. **The loop genuinely fires:** in the 10-question H100 run, 2
questions (financial, toxicology) ran to `iterations = 3` — a real generate→verify→revise
waterfall.

---

## 4. Agent tracing (Phase 4)

Langfuse (self-hosted via `docker-compose.yml`, headless-init so no manual signup) captures
every run as a `generate_sql / verify / (revise)` span waterfall with per-call prompt,
response, latency, and token counts. Runs are tagged with `{phase, split, db}` — passed as
trace **metadata** (filterable in Phase 6).

- `screenshots/langfuse_trace.png` — the toxicology run: full
  attach_schema → generate → execute → verify → revise → execute → verify → revise → execute
  waterfall, with `iteration: 3`, `verify_ok: false`, and the `verify_issue` text visible.
- `screenshots/langfuse_tags.png` — trace list with metadata columns.

---

## 5. Baseline eval (Phase 5)

**Signal = execution accuracy.** `evals/run_eval.py` runs the agent's final SQL and the
gold SQL against the target DB and compares **canonicalized row sets** (rows sorted, cells
stringified, NULL→''), ignoring column-name case. It also re-runs the SQL emitted at *each*
attempt (from the trace `history`) to report the pass rate per iteration, with carry-forward
for early-terminating runs — this is what reveals whether the verify→revise loop earns its keep.

| Metric | Value |
|---|---|
| Overall execution accuracy (30 questions) | **23.3 %** |
| Pass rate @ iter 0 (generate only) | **26.7 %** |
| Pass rate @ iter 1 | 23.3 % |
| Pass rate @ iter 2 | 23.3 % |
| Avg iterations / question | 1.53 |

Result: `results/eval_baseline.json`.

> **The loop does *not* earn its keep here — it regresses quality.** Iter-0 (raw generate)
> is **26.7 %**; after revising it drops to **23.3 %**. The verifier occasionally rewrites a
> correct query into a wrong one — its precision on this model isn't high enough to fire only
> on genuinely-bad SQL. Net-negative on aggregate accuracy, even though it clearly helped the
> 2 individual cases that reached iter 3. See §7 for the fix.

---

## 6. Hitting the SLO (Phase 6)

**Baseline vs SLO (P95 agent e2e < 5 s @ ≥ 10 RPS).** Load via `load_test/driver.py`
(open-loop, coordinated-omission-safe; latencies in seconds, percentiles over successful
requests). The current config **misses badly**, and pushing past the target shows the wall:

| Offered RPS | OK / total | OK throughput | p95 e2e | Verdict |
|---|---|---|---|---|
| 8  | 627 / 720   | ~4.3 rps | 27.1 s | ❌ |
| **10** | 2582 / 3000 | ~7.2 rps | **68.9 s** | ❌ (14× over SLO) |
| 30 | 1483 / 5400 | ~6.2 rps | 101.8 s | ❌ collapse (73 % errors) |

True sustainable capacity is **~6–7 successful agent runs/s** — and offering *more* load just
adds errors and queue latency, never throughput.

**Diagnosis (metric-grounded).** During the 10-RPS run the Grafana dashboard showed vLLM
**idle**: KV utilization **6.4 %**, queue depth **0**, vLLM-internal e2e p95 **2.31 s**,
TTFT p95 **100 ms**, TPOT p95 **33 ms** — all green. So the GPU is *not* the constraint; the
68.9 s the client sees is time spent **queueing upstream of vLLM**. The open-loop driver
offers 10 rps but the pipeline clears only ~7, so a backlog builds and coordinated-omission
latency diverges (p50 54.7 s → max 110 s = linear queue growth). **Root cause:** the
`/answer` endpoint is a *synchronous* handler running in FastAPI's bounded threadpool; with
2–3 dependent LLM calls per run it caps at ~7 rps regardless of how idle the GPU is. (SQLite
ruled out — `execute_sql` opens read-only and catches all exceptions, so it returns `ok=false`
rather than throwing; the 500s come from the agent→vLLM client path.)

**Iteration log** — *saw X → hypothesized Y → changed Z → result W*:

1. **Saw** P95 68.9 s @ 10 rps with the GPU idle (KV 6 %, queue 0, TTFT/TPOT tiny).
   **Hypothesized** the bottleneck is agent-layer concurrency (sync endpoint, bounded
   threadpool), not vLLM. **Changed** `uvicorn … --workers 1 → 4`. **Result:** P95
   **68.9 s → 9.0 s (7.7×)**, P50 **54.7 s → 1.83 s (30×)**. The targeted metric moved
   decisively → **diagnosis confirmed: it was orchestration concurrency.** But the SLO is
   still missed (9.0 s > 5 s), OK-throughput stayed ~flat (~6 rps), and **errors rose
   (14 % → 17 %)** — the failure mode shifted from "everything queues 60 s" to "fast
   responses, ~17 % shed as connection errors."

| 10 RPS | Before (1 worker) | After (4 workers) |
|---|---|---|
| P50 e2e | 54.7 s | **1.83 s** |
| P95 e2e | 68.9 s | **9.0 s** |
| P99 e2e | 73.9 s | 12.6 s |
| Errors | 14 % | 17 % |

Before/after: `screenshots/grafana_before.png`, `screenshots/grafana_after.png`.

**Post-tuning quality (`results/eval_after_tuning.json`): 23.3 %, unchanged** — as expected,
a concurrency change doesn't touch model output, so the latency win cost no accuracy.

**Honest verdict: SLO missed.** The gap closed from **14× to 1.8×** and the root cause is
proven, but full compliance needs a second iteration (below). Notably, the *a-priori* "lever
#1" (FP8 weights for more KV) would have done **nothing** — KV was never the constraint. The
lesson: confirm the bottleneck on the dashboard before pulling the obvious lever.

---

## 7. Wrap-up

**Did the agent loop help?** No — it **regressed** accuracy (iter-0 26.7 % → iter-2 23.3 %).
The verify→revise loop rewrites some correct SQL into wrong SQL. It should be gated, not run
unconditionally.

**SLO verdict:** Missed. Baseline P95 68.9 s vs 5 s @ 10 rps; after the agent-concurrency fix,
9.0 s — a 7.7× improvement that confirms the diagnosis but still 1.8× over target. The GPU was
never the bottleneck (6 % KV under load).

**What I'd do with more time (specific):**
- **Iteration 2 (the real SLO fix):** make `/answer` `async` with an async LLM client so a
  single process handles many in-flight runs without threadpool limits, *and* fix the error
  path — the 149 HTTP + 58 client errors at 4 workers are now the limiter, likely outbound
  httpx connection-pool exhaustion to vLLM (raise `max_connections`). More workers alone hit
  diminishing returns and just trade queue latency for dropped requests.
- **Then, and only then, vLLM-side tuning:** once orchestration can actually feed it, KV will
  finally climb; FP8 weights / higher `--max-num-seqs` become relevant. There is huge GPU
  headroom (6 % KV) waiting behind the agent bottleneck.
- **Gate the revise loop** behind a SQL static check (EXPLAIN / column-existence lint) before
  spending a verify LLM call, and only revise on high-confidence verifier failures — to stop
  the loop from regressing quality.
- **Cache rendered schemas** server-side keyed by `db_id` (re-rendered per request today) to
  shave TTFT, and run the eval at higher concurrency to separate quality regressions from
  latency wins.
