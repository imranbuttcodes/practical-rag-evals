"""
evals/geval_app_eval.py — Application-Level G-Eval Suite (Correctness, Completeness, Style & Tone)

Uses DeepEval's GEval custom metric with step-by-step Chain-of-Thought rubrics
and log-probability scoring to evaluate factual Correctness, answer Completeness, and Brand Style & Tone.

Run from project root:
    python evals/geval_app_eval.py
"""

import json
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
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
GOLDEN_DATASET = PROJECT_ROOT / "goldens" / "application_golden_dataset.json"
TOP_K = 3


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

# 1. G-EVAL CORRECTNESS RUBRIC & METRIC DEFINITION
correctness_rubric = [
    Rubric(
        score_range=(5, 5),
        expected_outcome="Flawless (5/5): Matches all facts, numbers, figures, and sub-answers in expected_output 100% accurately with zero factual errors or missing details."
    ),
    Rubric(
        score_range=(4, 4),
        expected_outcome="Minor Omission (4/5): Factually accurate with no errors, but omits a secondary or minor background detail from expected_output."
    ),
    Rubric(
        score_range=(3, 3),
        expected_outcome="Major Omission / Partial Answer (3/5): Answers part of a multi-part query correctly, but omits major required claims or states information is unavailable."
    ),
    Rubric(
        score_range=(2, 2),
        expected_outcome="Factual Contradiction (2/5): Directly contradicts expected_output or asserts incorrect numbers, rules, or facts."
    ),
    Rubric(
        score_range=(1, 1),
        expected_outcome="Completely False / Non-Answer (1/5): Completely false, refusal, or irrelevant to expected_output."
    ),
]

correctness_geval = GEval(
    name="Correctness",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[
        "Compare the facts, numbers, dates, and claims in actual_output directly against expected_output.",
        "Check if actual_output contains any factual errors, incorrect figures, or misstatements relative to expected_output.",
        "Verify that key quantities (e.g. hours, percentages, days, dollar amounts) match expected_output exactly.",
        "Penalize actual_output if it contradicts expected_output or asserts facts that violate ground truth."
    ],
    rubric=correctness_rubric,
    strict_mode=False,
    threshold=0.7,
    verbose_mode=True,
    model=judge,
)

# 2. G-EVAL COMPLETENESS RUBRIC & METRIC DEFINITION
completeness_rubric = [
    Rubric(
        score_range=(5, 5),
        expected_outcome="Fully Complete (5/5): Addresses 100% of all sub-questions and required details present in input and expected_output."
    ),
    Rubric(
        score_range=(4, 4),
        expected_outcome="Mostly Complete (4/5): Addresses the core query completely, but omits a minor sub-detail or optional nuance."
    ),
    Rubric(
        score_range=(3, 3),
        expected_outcome="Partially Complete (3/5): Answers one sub-question well, but leaves another major sub-question completely unanswered."
    ),
    Rubric(
        score_range=(2, 2),
        expected_outcome="Barely Complete (2/5): Addresses only a small fragment of the query and skips the vast majority of required points."
    ),
    Rubric(
        score_range=(1, 1),
        expected_outcome="Incomplete / Refusal (1/5): Fails to answer the multi-part query or states information is missing across all parts."
    ),
]

completeness_geval = GEval(
    name="Completeness",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    evaluation_steps=[
        "Identify all sub-questions, requirements, and required topics asked in input and detailed in expected_output.",
        "Check if actual_output provides a dedicated, explicit answer for every single sub-question identified.",
        "Check if actual_output claims information is missing or unavailable for any part of the query.",
        "Penalize actual_output proportionally for every sub-question or required detail that was skipped or incomplete."
    ],
    rubric=completeness_rubric,
    strict_mode=False,
    threshold=0.7,
    verbose_mode=True,
    model=judge,
)

# 3. G-EVAL TONE & BRAND STYLE RUBRIC & METRIC DEFINITION
style_rubric = [
    Rubric(
        score_range=(5, 5),
        expected_outcome="Perfect Tone & Style (5/5): Outstanding clear, encouraging, structured, and pedagogical communication tone with clean formatting (bullet points, bold highlights)."
    ),
    Rubric(
        score_range=(4, 4),
        expected_outcome="Good Tone (4/5): Clear, polite, and professional tone, with minor formatting stiffness."
    ),
    Rubric(
        score_range=(3, 3),
        expected_outcome="Average Tone (3/5): Acceptable answer, but sounds somewhat dry, dense, or unhelpful in style."
    ),
    Rubric(
        score_range=(2, 2),
        expected_outcome="Poor Tone (2/5): Robotic, overly dense, unhelpful, or poorly formatted response."
    ),
    Rubric(
        score_range=(1, 1),
        expected_outcome="Unacceptable Tone (1/5): Rude, dismissive, abrasive, or entirely unreadable formatting."
    ),
]

style_geval = GEval(
    name="Style & Tone",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
    evaluation_steps=[
        "Evaluate if actual_output adopts a professional, clear, encouraging, and pedagogical communication tone.",
        "Check if complex policy jargon is explained clearly with structured, easy-to-read formatting (bullet points, bold highlights).",
        "Verify that the response avoids dry, robotic, rude, or overly dense legalistic phrasing.",
        "Penalize actual_output if it sounds robotic, abrasive, or poorly structured."
    ],
    rubric=style_rubric,
    strict_mode=False,
    threshold=0.7,
    verbose_mode=True,
    model=judge,
)

# LOAD GOLDEN DATASET & BUILD TEST CASES
with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
    dataset = json.load(f)

test_cases = []

print(f"Running Application-Level G-Eval Suite for {len(dataset['questions'])} questions...\n")

for item in dataset["questions"]:
    question = item["question"]
    expected_output = item["expected_output"]

    # Run RAG Pipeline (Retriever -> Generator)
    rag_result = generate_answer(question, k=TOP_K)
    actual_output = rag_result["answer"]
    retrieval_context = rag_result["retrieval_context"]

    # Build LLMTestCase for G-Eval Evaluation
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    test_cases.append(test_case)

if __name__ == "__main__":
    print("Evaluating Full Application-Level G-Eval Suite: Correctness, Completeness, Style & Tone...\n")
    evaluate(
        test_cases=test_cases,
        metrics=[
            correctness_geval,
            completeness_geval,
            style_geval,
        ],
    )
