"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    Strategy: the agent's `history` records the SQL it emitted at each attempt
    (generate_sql, then each revise). We re-run gold and every attempt locally
    (cheap SQLite) and compare canonicalized rows, so we can report not just
    final accuracy but the accuracy *at each iteration*.
    """
    db_id = question["db_id"]
    q_text = question["question"]
    gold_sql = question.get("gold_sql") or question.get("SQL") or question.get("sql")

    # Gold must be runnable for the question to be gradable.
    g_ok, gold_rows, g_err = run_sql(db_id, gold_sql) if gold_sql else (False, None, "no gold sql")
    if not g_ok:
        return {"db_id": db_id, "question": q_text, "skipped": True,
                "skip_reason": f"gold SQL not runnable: {g_err}"}

    try:
        resp = httpx.post(agent_url, json={"question": q_text, "db": db_id}, timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001 - any agent failure means we can't grade this row
        return {"db_id": db_id, "question": q_text, "skipped": True,
                "skip_reason": f"agent call failed: {type(e).__name__}: {e}"}

    history = data.get("history", [])
    candidate_sqls = [h["sql"] for h in history
                      if h.get("node") in ("generate_sql", "revise") and h.get("sql")]
    if not candidate_sqls and data.get("sql"):
        candidate_sqls = [data["sql"]]

    # Correctness of each attempt's SQL, in order.
    correct_by_attempt: list[bool] = []
    for sql in candidate_sqls:
        p_ok, p_rows, _ = run_sql(db_id, sql)
        correct_by_attempt.append(bool(p_ok and matches(gold_rows, p_rows)))

    final_sql = data.get("sql") or (candidate_sqls[-1] if candidate_sqls else "")
    fp_ok, fp_rows, _ = run_sql(db_id, final_sql)
    final_correct = bool(fp_ok and matches(gold_rows, fp_rows))

    return {
        "db_id": db_id,
        "question": q_text,
        "gold_sql": gold_sql,
        "final_sql": final_sql,
        "iterations": data.get("iterations", len(candidate_sqls)),
        "agent_ok": data.get("ok"),
        "agent_error": data.get("error"),
        "candidate_sqls": candidate_sqls,
        "correct_by_attempt": correct_by_attempt,
        "final_correct": final_correct,
        "skipped": False,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    graded = [r for r in results if not r.get("skipped")]
    n_total = len(results)
    n_graded = len(graded)
    if n_graded == 0:
        return {"n_total": n_total, "n_graded": 0, "skipped": n_total,
                "overall_accuracy": 0.0, "pass_rate_by_iteration": [], "avg_iterations": 0}

    # Width = most attempts any question took; shorter ones carry forward their
    # last attempt (the agent stopped emitting after terminating early).
    n_iter = max((len(r["correct_by_attempt"]) for r in graded), default=1) or 1

    pass_at = [0] * n_iter
    for r in graded:
        cba = r["correct_by_attempt"] or [r["final_correct"]]
        for k in range(n_iter):
            if (cba[k] if k < len(cba) else cba[-1]):
                pass_at[k] += 1

    overall = sum(1 for r in graded if r["final_correct"]) / n_graded
    avg_iters = sum(r.get("iterations", len(r["correct_by_attempt"])) for r in graded) / n_graded

    return {
        "n_total": n_total,
        "n_graded": n_graded,
        "skipped": n_total - n_graded,
        "overall_accuracy": round(overall, 4),
        "pass_rate_by_iteration": [round(c / n_graded, 4) for c in pass_at],
        "avg_iterations": round(avg_iters, 2),
        "max_iterations_observed": n_iter,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="only first N questions (smoke test)")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
