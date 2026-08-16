import json
import os
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM

from langchain_deepseek import ChatDeepSeek

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve


load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_EVAL_MODEL = os.getenv(
    "DEEPSEEK_EVAL_MODEL",
    "deepseek-chat",
)

if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set in the .env file."
    )


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_DATASET = (
    PROJECT_ROOT
    / "goldens"
    / "retriever_golden_dataset.json"
)



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



judge = DeepSeekJudge(
    DEEPSEEK_EVAL_MODEL
)



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