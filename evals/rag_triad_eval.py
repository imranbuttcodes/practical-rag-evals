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
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_deepseek import ChatDeepSeek

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import generate_answer

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_EVAL_MODEL = os.getenv("DEEPSEEK_EVAL_MODEL", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY is not set in the .env file.")

# PATHS & CONFIGURATION
GOLDEN_DATASET = PROJECT_ROOT / "goldens" / "faithfulness_golden_dataset.json"
TOP_K = 2


# DEEPSEEK JUDGE MODEL
class DeepSeekJudge(DeepEvalBaseLLM):

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = ChatDeepSeek(
            model=model_name,
            api_key=DEEPSEEK_API_KEY,
            temperature=0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return self.model_name


judge = DeepSeekJudge(DEEPSEEK_EVAL_MODEL)

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
