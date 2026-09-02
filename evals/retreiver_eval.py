import json
from pathlib import Path
import sys

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve
from evals.judge import judge

GOLDEN_DATASET = PROJECT_ROOT / "goldens" / "retriever_golden_dataset.json"

with open(
    GOLDEN_DATASET,
    "r",
    encoding="utf-8",
) as f:

    dataset = json.load(f)



TOP_K = 5



test_cases = []


for item in dataset["questions"]:

    question = item["question"]

    expected_output = item["expected_output"]


    retrieved_docs = retrieve(
        question,
        k=TOP_K,
    )


    # --------------------------------------------------------
    # Convert LangChain Documents into DeepEval context
    # --------------------------------------------------------

    retrieval_context = [
        doc.page_content
        for doc in retrieved_docs
    ]


    # --------------------------------------------------------
    # Create DeepEval test case
    # --------------------------------------------------------

    test_case = LLMTestCase(

        input=question,

        # We are evaluating the RETRIEVER,
        # not the answer generator.
        actual_output="",

        # Golden / ideal answer.
        expected_output=expected_output,

        # Actual chunks returned by YOUR retriever.
        retrieval_context=retrieval_context,
    )


    test_cases.append(test_case)



contextual_recall = ContextualRecallMetric(

    threshold=0.7,

    include_reason=True,

    model=judge,
)



contextual_precision = ContextualPrecisionMetric(

    threshold=0.7,

    include_reason=True,

    model=judge,
)



if __name__ == "__main__":

    evaluate(

        test_cases=test_cases,

        metrics=[
            contextual_recall,
            contextual_precision,
        ],
    )