"""
evals/rag_triad_eval.py — End-to-End RAG Triad Pipeline Evaluation

The RAG Triad evaluates the 3 essential quality legs of a RAG application:
  1. Context Relevance  (ContextualRelevancyMetric): Is retrieved context relevant to the Query?
  2. Faithfulness       (FaithfulnessMetric): Is the Answer grounded in the Context (no hallucinations)?
  3. Answer Relevance   (AnswerRelevancyMetric): Is the Answer relevant to the Query?

Run from project root:
    python evals/rag_triad_eval.py
"""

import json
from pathlib import Path
import sys

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import generate_answer
from evals.judge import judge

# PATHS & CONFIGURATION
GOLDEN_DATASET = PROJECT_ROOT / "goldens" / "faithfulness_golden_dataset.json"
TOP_K = 2


from evals.judge import judge

# LOAD GOLDEN DATASET & RUN END-TO-END RAG PIPELINE
with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)

test_cases = []

print(f"Running End-to-End RAG Pipeline for {len(dataset['questions'])} questions...\n")

for item in dataset["questions"]:
    question = item["question"]
    expected_output = item["expected_output"]

    # End-to-End Pipeline Execution: Retriever -> Generator
    rag_result = generate_answer(question, k=TOP_K)
    actual_output = rag_result["answer"]
    retrieval_context = rag_result["retrieval_context"]

    # Build LLMTestCase for RAG Triad Evaluation
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    test_cases.append(test_case)

# RAG TRIAD METRICS DEFINITION

# 1. Context Relevance: Is retrieved context relevant to Query?
context_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
)

# 2. Faithfulness: Is Answer grounded in Context (no hallucinations)?
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
)

# 3. Answer Relevance: Is Answer relevant to Query?
answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
)

if __name__ == "__main__":
    print("Evaluating RAG Triad Metrics (Context Relevance, Faithfulness, Answer Relevance)...\n")
    evaluate(
        test_cases=test_cases,
        metrics=[
            context_relevancy_metric,
            faithfulness_metric,
            answer_relevancy_metric,
        ],
    )
