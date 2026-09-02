"""
evals/latency_app_eval.py — Operational Latency Evaluation Suite (with TTFT & SLO Budgets)

Operational evaluation: LATENCY (with time-to-first-token).

Unlike quality evals (correctness, faithfulness, toxicity, leakage, scope),
latency needs no golden dataset and no LLM judge. It is a deterministic measurement:
run the pipeline N times, collect a distribution, and report percentiles against
a budget (SLO) -- not against a ground truth.

Two latency numbers matter, and they answer different questions:
  - END-TO-END total : how long until the FULL answer is ready
  - TTFT (perceived) : how long until the user sees the FIRST token stream in

Key ideas encoded below:
  - perf_counter, not time()          (right clock for elapsed time)
  - many samples -> percentiles       (p95/p99 tail, not misleading mean)
  - discard warmup                    (cold start poisons the stats)
  - decompose the pipeline            (retrieval + generation, + TTFT)
  - log answer length                 (latency couples to output length)
  - SLO budgets (PASS/FAIL)           (latency numbers need targets to evaluate against)

Run from project root:
    python evals/latency_app_eval.py
"""

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve
from src.generator import generate_stream_from_context

# CONFIGURATION & SLO BUDGETS
GOLDEN_DATASET = PROJECT_ROOT / "goldens" / "quality_golden_dataset.json"
TOP_K = 3

REPEATS = 2           # Measured runs per question -> total samples = len(QUESTIONS) * REPEATS
WARMUP_RUNS = 1       # Throwaway calls before measuring (cold start)

MEASURE_TTFT = True   # Stream generation and clock time-to-first-token (perceived latency)

# Service Level Objectives (SLOs / Budgets in milliseconds)
SLO_P95_MS = 5000        # End-to-end full answer p95 under 5.0s (5000ms)
SLO_TTFT_P95_MS = 2500   # Perceived first visible token p95 under 2.5s (2500ms)


def run_stages_streaming(question: str, k: int = TOP_K):
    """
    Stage-level streaming evaluation:
    1. Times retrieval (embedding + vector DB search)
    2. Streams generation and records time-to-first-token (TTFT)
    
    Returns:
        answer (str), stage_timings (dict in ms)
    """
    t0 = time.perf_counter()
    retrieved_docs = retrieve(question, k=k)
    context_texts = [doc.page_content for doc in retrieved_docs]
    t1 = time.perf_counter()

    first_token_t = None
    pieces = []
    
    stream = generate_stream_from_context(question, context_texts)
    for piece in stream:
        if first_token_t is None and piece.strip():
            first_token_t = time.perf_counter()   # Clock the first visible token chunk
        pieces.append(piece)
    t2 = time.perf_counter()

    answer = "".join(pieces)
    ttft_ms = (first_token_t - t0) * 1000.0 if first_token_t else float("nan")
    
    return answer, {
        "retrieval": (t1 - t0) * 1000.0,
        "generation": (t2 - t1) * 1000.0,
        "ttft": ttft_ms,
    }


def percentile(values: List[float], p: float) -> float:
    """Linear-interpolation percentile calculation."""
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    s = sorted(clean)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def benchmark():
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = [q["question"] for q in dataset["questions"]]

    # Warmup: run and DISCARD cold start stats
    print(f"Warming up ({WARMUP_RUNS} runs, discarded for cold start)...")
    for i in range(WARMUP_RUNS):
        run_stages_streaming(questions[i % len(questions)])
    print("Warmup completed.\n")

    total_ms, retrieval_ms, generation_ms, ttft_ms = [], [], [], []
    answer_lengths = []

    # Measured runs: repeat questions REPEATS times
    print(f"Measuring {len(questions)} benchmark questions ({REPEATS} repeats per question)...")
    
    for q_idx, question in enumerate(questions, 1):
        print(f"[{q_idx}/{len(questions)}] Benchmarking: '{question[:45]}...'")
        for r in range(REPEATS):
            start = time.perf_counter()
            answer, stage = run_stages_streaming(question)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            total_ms.append(elapsed_ms)
            retrieval_ms.append(stage["retrieval"])
            generation_ms.append(stage["generation"])
            ttft_ms.append(stage["ttft"])
            answer_lengths.append(len(answer or ""))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }


def summarize(samples: List[float]) -> Dict[str, Any]:
    clean = [s for s in samples if not math.isnan(s)]
    if not clean:
        return {"n": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def print_row(label: str, s: Dict[str, Any]):
    print(f"{label:<12} | n={s['n']:<3} "
          f"mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
          f"p95={s['p95']:7.1f}  p99={s['p99']:7.1f}  "
          f"min={s['min']:7.1f}  max={s['max']:7.1f}")


def slo_line(label: str, p95: float, budget: float):
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(f"SLO Target: {label:<22} p95 <= {budget:>5.0f} ms  ->  Actual p95 = {p95:7.1f} ms   [{verdict}]")


def report(results: Dict[str, Any]):
    print("\n" + "=" * 82)
    print("OPERATIONAL LATENCY BENCHMARK REPORT (milliseconds)")
    print("=" * 82)
    print(f"{'Stage':<12} | {'n':<5} {'Mean':>11} {'P50':>11} "
          f"{'P95':>11} {'P99':>11} {'Min':>11} {'Max':>11}")
    print("-" * 82)

    total = summarize(results["total"])
    print_row("end-to-end", total)
    if results["ttft"]:
        print_row("ttft", summarize(results["ttft"]))   # Perceived: query -> first token
    if results["retrieval"]:
        print_row("retrieval", summarize(results["retrieval"]))
        print_row("generation", summarize(results["generation"]))

    avg_len = sum(results["answer_len"]) / len(results["answer_len"]) if results["answer_len"] else 0
    print("-" * 82)
    print(f"avg answer length: {avg_len:.0f} chars (latency couples to output token length)")

    # SLO Verdicts
    print("=" * 82)
    slo_line("full answer", total["p95"], SLO_P95_MS)
    if results["ttft"]:
        ttft_summary = summarize(results["ttft"])
        slo_line("first token (perceived)", ttft_summary["p95"], SLO_TTFT_P95_MS)
    print("=" * 82)


if __name__ == "__main__":
    benchmark_results = benchmark()
    report(benchmark_results)
