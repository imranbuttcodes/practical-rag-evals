import json
from pathlib import Path
import sys

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import generate_from_context, generate_answer
from evals.judge import judge

# PATHS & CONFIGURATION
GOLDEN_DATASET = PROJECT_ROOT / "goldens" / "faithfulness_golden_dataset.json"

# Set ISOLATED_TESTING = True to test generator in isolation with ground-truth ideal_context
# Set ISOLATED_TESTING = False for end-to-end pipeline testing (retriever + generator)
ISOLATED_TESTING = False
TOP_K = 2


from evals.judge import judge

# LOAD GOLDEN DATASET & BUILD TEST CASES
with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)

test_cases = []

mode_str = "ISOLATED GENERATOR (ideal_context)" if ISOLATED_TESTING else "END-TO-END PIPELINE (retriever + generator)"
print(f"Running evaluation in mode: [{mode_str}] for {len(dataset['questions'])} test cases...\n")

for item in dataset["questions"]:
    question = item["question"]
    expected_output = item["expected_output"]

    if ISOLATED_TESTING:
        # Isolated Generator Mode: Bypass retriever and supply ground-truth ideal_context directly
        retrieval_context = item.get("ideal_context", [])
        actual_output = generate_from_context(question, retrieval_context)
    else:
        # End-to-End Mode: Retrieve context dynamically and generate answer
        gen_result = generate_answer(question, k=TOP_K)
        actual_output = gen_result["answer"]
        retrieval_context = gen_result["retrieval_context"]

    # Build DeepEval LLMTestCase for Generator Evaluation
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    test_cases.append(test_case)

# METRICS DEFINITION
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    model=judge,
)

if __name__ == "__main__":
    evaluate(
        test_cases=test_cases,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
        ],
    )
