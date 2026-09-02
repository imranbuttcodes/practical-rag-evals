"""
evals/reliability_app_eval.py — Operational Reliability Evaluation Suite

Operational evaluation: RELIABILITY (Failure Categorization, Timeout & Retry Rates, SLO Budgets).

Unlike quality evals (correctness, faithfulness, toxicity, leakage, scope),
reliability needs no LLM judge. It is a deterministic measurement:
runs the pipeline across diverse benchmark queries, tracks stage completion,
categorizes specific failure modes (API errors, rate limits, retrieval failures,
timeouts, parsing errors), and reports overall success, error, and timeout rates
against target SLO reliability budgets.

Run from project root:
    python evals/reliability_app_eval.py
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

# CONFIGURATION & SLO RELIABILITY BUDGETS
GOLDEN_DATASET = PROJECT_ROOT / "goldens" / "quality_golden_dataset.json"
TOP_K = 3

REPEATS = 2                 # Runs per question to increase sample count (N)
MAX_TIMEOUT_SEC = 10.0      # Maximum allowed timeout threshold per request (seconds)

# Service Level Objectives (SLOs / Reliability Budgets)
SLO_SUCCESS_RATE_PCT = 95.0   # Target: Success rate >= 95.0%
SLO_TIMEOUT_RATE_PCT = 2.0    # Target: Timeout rate <= 2.0%


def run_pipeline_with_reliability_tracking(question: str, timeout_sec: float = MAX_TIMEOUT_SEC) -> Dict[str, Any]:
    """
    Executes a single RAG pipeline turn with timeout and failure categorization.
    
    Returns:
        Dict containing success status, latency, response length, and categorized failure type.
    """
    start_time = time.perf_counter()
    failure_category = None
    retries_required = 0

    try:
        # Step 1: Retrieval Stage
        retriever_start = time.perf_counter()
        retrieved_docs = retrieve(question, k=TOP_K)
        retriever_time = time.perf_counter() - retriever_start

        if not retrieved_docs:
            return {
                "success": False,
                "latency_sec": time.perf_counter() - start_time,
                "failure_category": "Retrieval Failure (Empty Context)",
                "retries": retries_required,
                "response": ""
            }

        context_texts = [doc.page_content for doc in retrieved_docs]

        # Step 2: Generation Stage with Timeout Checking
        stream = generate_stream_from_context(question, context_texts)
        pieces = []

        for piece in stream:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_sec:
                return {
                    "success": False,
                    "latency_sec": elapsed,
                    "failure_category": "Timeout Exceeded",
                    "retries": retries_required,
                    "response": ""
                }
            pieces.append(piece)

        response_text = "".join(pieces)
        total_time = time.perf_counter() - start_time

        if not response_text.strip():
            return {
                "success": False,
                "latency_sec": total_time,
                "failure_category": "Parser / Empty Response Error",
                "retries": retries_required,
                "response": ""
            }

        return {
            "success": True,
            "latency_sec": total_time,
            "failure_category": None,
            "retries": retries_required,
            "response": response_text
        }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        err_msg = str(e).lower()

        if "rate" in err_msg or "429" in err_msg:
            category = "Rate Limit Error (429)"
        elif "connection" in err_msg or "500" in err_msg or "503" in err_msg:
            category = "LLM API Network/Server Error"
        elif "timeout" in err_msg:
            category = "Timeout Exceeded"
        else:
            category = f"Internal Exception ({type(e).__name__})"

        return {
            "success": False,
            "latency_sec": elapsed,
            "failure_category": category,
            "retries": retries_required,
            "response": ""
        }


def benchmark_reliability() -> Dict[str, Any]:
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = [q["question"] for q in dataset["questions"]]
    total_samples = len(questions) * REPEATS

    print(f"Benchmarking operational reliability across {len(questions)} queries ({REPEATS} repeats, N={total_samples})...")

    results = []
    failure_counts: Dict[str, int] = {
        "LLM API Network/Server Error": 0,
        "Rate Limit Error (429)": 0,
        "Retrieval Failure (Empty Context)": 0,
        "Timeout Exceeded": 0,
        "Parser / Empty Response Error": 0,
        "Internal Exception": 0
    }

    sample_counter = 0
    for q_idx, question in enumerate(questions, 1):
        for r in range(REPEATS):
            sample_counter += 1
            res = run_pipeline_with_reliability_tracking(question, timeout_sec=MAX_TIMEOUT_SEC)
            results.append(res)

            status_str = "SUCCESS" if res["success"] else f"FAILED [{res['failure_category']}]"
            print(f"[{sample_counter}/{total_samples}] Query: '{question[:35]}...' | Status: {status_str} | Latency: {res['latency_sec']:.2f}s")

            if not res["success"]:
                category = res["failure_category"]
                matched = False
                for key in failure_counts:
                    if key in category:
                        failure_counts[key] += 1
                        matched = True
                        break
                if not matched:
                    failure_counts["Internal Exception"] += 1

    return {
        "samples": results,
        "total_n": total_samples,
        "failure_counts": failure_counts
    }


def slo_line(label: str, actual: float, target: float, is_lower_better: bool = False):
    if is_lower_better:
        verdict = "PASS" if actual <= target else "FAIL"
        comp_symbol = "<="
    else:
        verdict = "PASS" if actual >= target else "FAIL"
        comp_symbol = ">="

    print(f"SLO Target: {label:<22} {comp_symbol} {target:>5.1f}%  ->  Actual = {actual:>5.1f}%   [{verdict}]")


def report(data: Dict[str, Any]):
    samples = data["samples"]
    total_n = data["total_n"]
    failure_counts = data["failure_counts"]

    successes = sum(1 for s in samples if s["success"])
    failures = total_n - successes

    success_rate_pct = (successes / total_n) * 100.0 if total_n > 0 else 0.0
    error_rate_pct = (failures / total_n) * 100.0 if total_n > 0 else 0.0

    timeout_count = failure_counts.get("Timeout Exceeded", 0)
    timeout_rate_pct = (timeout_count / total_n) * 100.0 if total_n > 0 else 0.0

    retries_count = sum(s["retries"] for s in samples)
    retry_rate_pct = (retries_count / total_n) * 100.0 if total_n > 0 else 0.0

    print("\n" + "=" * 82)
    print("OPERATIONAL RELIABILITY BENCHMARK REPORT")
    print("=" * 82)
    print(f"Total Requests (n):      {total_n}")
    print(f"Successful Requests:     {successes} ({success_rate_pct:.1f}%)")
    print(f"Failed Requests:         {failures} ({error_rate_pct:.1f}%)")
    print(f"Timeout Failures:        {timeout_count} ({timeout_rate_pct:.1f}%)")
    print("-" * 82)

    print("DETAILED FAILURE MODE CATEGORIZATION:")
    print(f"{'Failure Category':<40} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 82)
    for category, count in failure_counts.items():
        pct = (count / total_n) * 100.0 if total_n > 0 else 0.0
        print(f"{category:<40} | {count:<8} | {pct:>8.1f}%")
    print("-" * 82)

    # SLO RELIABILITY VERDICTS
    print("=" * 82)
    slo_line("success rate", success_rate_pct, SLO_SUCCESS_RATE_PCT, is_lower_better=False)
    slo_line("timeout rate", timeout_rate_pct, SLO_TIMEOUT_RATE_PCT, is_lower_better=True)
    print("=" * 82)


if __name__ == "__main__":
    benchmark_data = benchmark_reliability()
    report(benchmark_data)
