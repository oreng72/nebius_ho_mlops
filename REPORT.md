# LLM inference + o11y — Report

Text-to-SQL PoC serving **Qwen3-30B-A3B-Instruct-2507** on 1× H100 80 GB, with a
LangGraph verify→revise agent, Prometheus/Grafana serving observability, and
Langfuse agent tracing. SLO: **P95 end-to-end agent latency < 5 s at ≥ 10 RPS**
(1 RPS = one full agent run = 2–3 vLLM calls) over a 5-minute window.

> Numbers marked `‹H100›` are collected on the GPU; everything else (config,
> architecture, eval methodology) was built and validated off-GPU against a
> Nebius Token Factory stand-in model.

---

## 1. Serving configuration (Phase 1)

**Constraint that drives everything:** Qwen3-30B-A3B is an MoE — ~30 B total
params (≈ 61 GB in BF16) but only ~3 B active per token. On an 80 GB H100 the
weights consume most of VRAM, so **concurrency is KV-cache-bound, not
compute-bound.** The config trades context length and precision for KV headroom.

| Flag | Value | Why (for *this* workload) |
|---|---|---|
| `--dtype` | `bfloat16` | Baseline precision; FP8 weights held as the first Phase-6 lever. |
| `--max-model-len` | `8192` | Prompts (1.5–3 K) + short SQL output fit in 8 K; smaller context ⇒ more KV slots ⇒ more concurrency. |
| `--gpu-memory-utilization` | `0.92` | Claw back VRAM for KV without OOM risk on the 80 GB card. |
| `--max-num-seqs` | `96` | Cap concurrent sequences near the real concurrency target (~20–30 vLLM req/s). |
| `--max-num-batched-tokens` | `8192` | Per-step token budget; bounds how much a big prefill can starve in-flight decode. |
| `--enable-chunked-prefill` | on | Slice 3 K-token prefills so decodes aren't blocked → protects TTFT/TPOT under load. |
| `--enable-prefix-caching` | on | The DB schema prefix repeats across the 2–3 calls per agent run → cache it, skip re-prefill. |
| `--kv-cache-dtype` | `fp8` | Halve KV bytes/token → ~2× the sequences fit in the tight KV budget. |
| `--tensor-parallel-size` | `1` | Single H100, no sharding. |
| `--disable-log-requests` | on | Drop per-request logging overhead under load. |

Full script + rationale: `scripts/start_vllm.sh`.

---

## 2. Observability dashboard (Phase 2)

Grafana dashboard `vLLM Serving Health` (`infra/grafana/provisioning/dashboards/serving.json`),
built from vLLM's `/metrics`, readable cold. Four at-a-glance SLO tiles
(P95 e2e latency, RPS, KV utilization, queue depth) over three sections:

- **Latency — "is it slow, and *where* in the lifecycle?"** e2e p50/p95/p99,
  a stacked queue → prefill → decode breakdown, plus TTFT and TPOT (inter-token).
- **Throughput.** Prompt vs generation tokens/s, request rate by finish reason,
  and running-vs-waiting concurrency.
- **KV cache — "headroom, or about to evict?"** `gpu_cache_usage_perc` (with
  warn/crit threshold lines), prefix-cache hit rate, and preemptions/s.

Validated off-GPU: provisions cleanly, scrape target wired, PromQL parses.
`‹H100›` screenshot of panels reacting under load: `screenshots/grafana_serving.png`.

---

## 3. Agent design (Phase 3)

LangGraph graph (`agent/graph.py`):

```
attach_schema → generate_sql → execute → verify ──ok──▶ END
                                  ▲          │
                                  └─ revise ◀┘ (not ok, iteration < MAX)
```

- **generate_sql / revise** call the model via `langchain_openai.ChatOpenAI`; SQL
  is pulled from the ```` ```sql ```` block.
- **verify** short-circuits a hard execution error (a crashed query never plausibly
  answers), otherwise asks the model for a strict `{ok, issue}` JSON judged against
  the question — flagging SQL errors, empty results when rows are implied, wrong
  column shape, or an ignored condition. Parsed defensively (default ok on bad JSON
  so a flaky reply can't trap the loop).
- **route_after_verify** ends on `verify_ok` or at `MAX_ITERATIONS = 3`, else revises.

Prompts in `agent/prompts.py`. Served at `POST /answer` (`agent/server.py`), which
returns the final SQL, rows, iteration count, and the per-node `history`.
The loop genuinely fires: in the off-GPU 10-question run, 2 questions triggered a
revise (a real `generate → verify → revise` waterfall in Langfuse).

---

## 4. Agent tracing (Phase 4)

Langfuse (self-hosted via `docker-compose.yml`, headless-init so no manual signup)
captures every run as a `generate_sql / verify / (revise)` span waterfall with
per-call prompt, response, latency, and token counts. Traces are tagged with
`{phase, split, db}` metadata for Phase-6 filtering.
`‹H100›` `screenshots/langfuse_trace.png` (a revise waterfall) and
`screenshots/langfuse_tags.png` (trace list with tags).

---

## 5. Baseline eval (Phase 5)

**Signal = execution accuracy.** `evals/run_eval.py` runs the agent's final SQL
and the gold SQL against the target DB and compares **canonicalized row sets**
(rows sorted, cells stringified, NULL→''), ignoring column-name case. It also
re-runs the SQL emitted at *each* attempt (from the trace `history`) to report the
**pass rate at each iteration**, with carry-forward for runs that terminate early —
this is what reveals whether the verify→revise loop earns its keep.

| Metric | Value |
|---|---|
| Overall execution accuracy | `‹H100›` |
| Pass rate @ iter 0 (generate only) | `‹H100›` |
| Pass rate @ iter 1 | `‹H100›` |
| Pass rate @ iter 2 | `‹H100›` |
| Avg iterations / question | `‹H100›` |

Result: `results/eval_baseline.json`. Grafana during the run:
`screenshots/grafana_eval_run.png`.

> Commentary `‹H100›`: if iter-0 ≈ iter-2, the loop is doing nothing; if iter-2 is
> meaningfully higher, the architecture adds measurable value.

---

## 6. Hitting the SLO (Phase 6)

Baseline vs SLO (P95 < 5 s @ ≥ 10 RPS): `‹H100›`.

**Iteration log** — *saw X → hypothesized Y → changed Z → result W*:

1. `‹H100›` (lever #1 candidate: FP8 weights — frees ~30 GB for KV → higher concurrency)
2. `‹H100›`
3. `‹H100›`

Final config result: `‹H100›`. Post-tuning quality: `results/eval_after_tuning.json`
(`‹H100›` — did accuracy survive the tuning?). Before/after:
`screenshots/grafana_before.png`, `screenshots/grafana_after.png`.

---

## 7. Wrap-up

**Did the agent loop help?** `‹H100›` — cite the iter-0 vs iter-2 pass rates above.

**SLO verdict:** `‹H100›` — hit, or missed with the gap quantified and diagnosed.

**What I'd do with more time (specific):**
- Cache rendered schemas server-side keyed by `db_id` (today the schema is
  re-rendered per request) to cut a measurable chunk off TTFT under load.
- Add a SQL static-check (EXPLAIN / column-existence lint) *before* execution so
  obvious errors are caught without spending a verify LLM call.
- Make the verifier emit a structured failure category and route revises with
  category-specific prompts, instead of one generic revise prompt.
- Run the eval at higher concurrency to separate quality regressions from latency
  wins, and add a small held-out set to guard against prompt overfitting.
