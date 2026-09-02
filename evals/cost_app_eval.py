"""
evals/cost_app_eval.py — Operational Cost Evaluation Suite

Operational evaluation: COST (Input vs Output Token Pricing & SLO Budgets).

Unlike quality evals (correctness, faithfulness, toxicity, leakage, scope),
cost evaluation needs no LLM judge. It is a deterministic measurement:
tracks input (prompt) and output (completion) token consumption per query,
applies the exact LLM provider pricing rates, and reports cost distributions
(P50, P90, P95) against a target SLO cost budget.

OpenRouter DeepSeek V3 Pricing Rates:
  - Input Token Price : $0.2574 per 1,000,000 tokens
  - Output Token Price: $1.0290 per 1,000,000 tokens

Run from project root:
    python evals/cost_app_eval.py
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
from src.generator import generate_from_context_with_metadata

# CONFIGURATION & PRICING (OpenRouter DeepSeek V3 Rates)
GOLDEN_DATASET = PROJECT_ROOT / "goldens" / "quality_golden_dataset.json"
TOP_K = 3

INPUT_PRICE_PER_1M = 0.2574    # USD per 1M prompt tokens
OUTPUT_PRICE_PER_1M = 1.0290   # USD per 1M completion tokens
USD_TO_PKR = 277.55             # Currency conversion rate for PKR (Rs)

# Service Level Objective (SLO Cost Budget per query)
SLO_COST_P95_USD = 0.0025      # Target: P95 query cost under $0.0025 USD (~Rs 0.694 PKR)


def calculate_query_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate monetary cost in USD for a single query turn."""
    input_cost = (prompt_tokens / 1_000_000.0) * INPUT_PRICE_PER_1M
    output_cost = (completion_tokens / 1_000_000.0) * OUTPUT_PRICE_PER_1M
    return input_cost + output_cost


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


def benchmark_cost():
    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    questions = [q["question"] for q in dataset["questions"]]
    total_queries = len(questions)

    print(f"Measuring operational token costs for {total_queries} benchmark queries...")

    prompt_token_counts = []
    completion_token_counts = []
    total_token_counts = []
    query_costs_usd = []
    query_costs_pkr = []

    for q_idx, question in enumerate(questions, 1):
        # 1. Retrieve Context
        retrieved_docs = retrieve(question, k=TOP_K)
        context_texts = [doc.page_content for doc in retrieved_docs]

        # 2. Invoke LLM and extract official usage_metadata from LLM API response
        meta_result = generate_from_context_with_metadata(question, context_texts)
        prompt_tokens = meta_result["input_tokens"]
        completion_tokens = meta_result["output_tokens"]
        total_tokens = meta_result["total_tokens"] or (prompt_tokens + completion_tokens)

        cost_usd = calculate_query_cost(prompt_tokens, completion_tokens)
        cost_pkr = cost_usd * USD_TO_PKR

        prompt_token_counts.append(prompt_tokens)
        completion_token_counts.append(completion_tokens)
        total_token_counts.append(total_tokens)
        query_costs_usd.append(cost_usd)
        query_costs_pkr.append(cost_pkr)

        print(
            f"[{q_idx}/{total_queries}] Query: '{question[:35]}...' | "
            f"Input Tokens: {prompt_tokens} | "
            f"Output Tokens: {completion_tokens} | "
            f"Cost: ${cost_usd:.5f} (PKR Rs {cost_pkr:.3f})"
        )

    return {
        "prompt_tokens": prompt_token_counts,
        "completion_tokens": completion_token_counts,
        "total_tokens": total_token_counts,
        "costs_usd": query_costs_usd,
        "costs_pkr": query_costs_pkr,
    }


def summarize(samples: List[float]) -> Dict[str, Any]:
    clean = [s for s in samples if not math.isnan(s)]
    if not clean:
        return {"n": 0, "mean": 0, "p50": 0, "p90": 0, "p95": 0, "min": 0, "max": 0}
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "min": min(clean),
        "max": max(clean),
    }


def print_row(label: str, s: Dict[str, Any], fmt_spec: str = ".5f"):
    print(
        f"{label:<20} | n={s['n']:<3} "
        f"mean={s['mean']:{fmt_spec}}  p50={s['p50']:{fmt_spec}}  "
        f"p90={s['p90']:{fmt_spec}}  p95={s['p95']:{fmt_spec}}  "
        f"min={s['min']:{fmt_spec}}  max={s['max']:{fmt_spec}}"
    )


def slo_line(label: str, p95: float, budget: float):
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(f"SLO Target: {label:<22} p95 <= ${budget:.5f}  ->  Actual p95 = ${p95:.5f}   [{verdict}]")


def report(results: Dict[str, Any]):
    total_prompt_tokens = sum(results["prompt_tokens"])
    total_completion_tokens = sum(results["completion_tokens"])
    total_all_tokens = sum(results["total_tokens"])

    total_suite_cost_usd = sum(results["costs_usd"])
    total_suite_cost_pkr = sum(results["costs_pkr"])

    cost_usd_stats = summarize(results["costs_usd"])
    cost_pkr_stats = summarize(results["costs_pkr"])
    prompt_tok_stats = summarize(results["prompt_tokens"])
    comp_tok_stats = summarize(results["completion_tokens"])

    print("\n" + "=" * 90)
    print("OPERATIONAL COST & TOKEN BENCHMARK REPORT")
    print("=" * 90)
    print("Model: DeepSeek V3 (OpenRouter Rates: $0.2574/1M Input, $1.0290/1M Output)")
    print("-" * 90)
    print(f"{'Category':<20} | {'n':<5} {'Mean':>11} {'P50':>11} {'P90':>11} {'P95':>11} {'Min':>11} {'Max':>11}")
    print("-" * 90)

    print_row("Prompt Tokens", prompt_tok_stats, "7.1f")
    print_row("Completion Tokens", comp_tok_stats, "7.1f")
    print("-" * 90)
    print_row("Query Cost ($ USD)", cost_usd_stats, "9.5f")
    print_row("Query Cost (PKR Rs)", cost_pkr_stats, "7.3f")
    print("-" * 90)

    print("SUITE AGGREGATE SUMMARY:")
    print(f"  Total Prompt Tokens:     {total_prompt_tokens:,} tokens")
    print(f"  Total Completion Tokens: {total_completion_tokens:,} tokens")
    print(f"  Total Tokens Consumed:   {total_all_tokens:,} tokens")
    print(f"  Total Suite Cost:        ${total_suite_cost_usd:.5f} USD (PKR Rs {total_suite_cost_pkr:.2f})")
    print(f"  Mean Cost per Query:     ${cost_usd_stats['mean']:.5f} USD (PKR Rs {cost_pkr_stats['mean']:.3f})")

    # SLO COST VERDICT
    print("=" * 90)
    slo_line("query cost ($ USD)", cost_usd_stats["p95"], SLO_COST_P95_USD)
    print("=" * 90)


if __name__ == "__main__":
    cost_results = benchmark_cost()
    report(cost_results)
