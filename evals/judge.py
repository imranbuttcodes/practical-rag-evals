"""
evals/judge.py — Centralized DeepSeek LLM Judge module for DeepEval evaluation metrics

Provides a single authoritative DeepSeekJudge implementation and shared `judge` instance across all component, pipeline, and application evaluation suites.
"""

import os
from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_deepseek import ChatDeepSeek

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_EVAL_MODEL = os.getenv("DEEPSEEK_EVAL_MODEL", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY is not set in the .env file.")


class DeepSeekJudge(DeepEvalBaseLLM):

    def __init__(self, model_name: str = DEEPSEEK_EVAL_MODEL):
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


# Default shared judge instance
judge = DeepSeekJudge()
