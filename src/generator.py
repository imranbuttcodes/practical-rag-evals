import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
# from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve

load_dotenv()

# System prompt forcing grounded answers based on retrieved context
SYSTEM_PROMPT = """You are a helpful and accurate assistant. 
Answer the user's question based strictly on the provided context below. 
If the context does not contain enough information to answer the question, state clearly that you do not have enough information based on the provided documents. Do not guess or make up facts.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

def get_llm():
    # --- DeepSeek Model (Temporarily Commented Out) ---
    # api_key = os.getenv("DEEPSEEK_API_KEY")
    # if not api_key:
    #     raise RuntimeError("DEEPSEEK_API_KEY is not set in the .env file.")
    # return ChatDeepSeek(
    #     model='deepseek-chat',
    #     api_key=api_key,
    #     temperature=0,
    # )

    # --- Groq Model (Temporary Active Model) ---
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the .env file.")
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0,
    )

def generate_from_context(question: str, context_texts: List[str]) -> str:
    """
    Isolated Generator step:
    Accepts explicit context strings (e.g. ideal_context from golden dataset)
    and returns the generated answer string directly.
    """
    formatted_context = "\n\n---\n\n".join(context_texts)
    chain = PROMPT | get_llm()

    response = chain.invoke({
        "context": formatted_context,
        "question": question
    })

    return response.content


def generate_answer(question: str, k: int = 4) -> Dict[str, Any]:
    """
    RAG Pipeline generation step:
    1. Retrieves top-k documents for the given question.
    2. Formats documents into prompt context.
    3. Invokes LLM to generate grounded response.

    Returns:
        dict: {
            "question": str,
            "answer": str,
            "retrieved_docs": List[Document],
            "retrieval_context": List[str]
        }
    """
    retrieved_docs: List[Document] = retrieve(question, k=k)
    context_texts = [doc.page_content for doc in retrieved_docs]
    answer = generate_from_context(question, context_texts)

    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "retrieval_context": context_texts
    }


if __name__ == "__main__":
    test_question = "Summerize the Policy paper?"
    print(f"Question: {test_question}\n")
    
    result = generate_answer(test_question, k=3)
    print("Generated Answer:")
    print(result["answer"])
    print("\nRetrieved Context Count:", len(result["retrieved_docs"]))

    print(result['retrieval_context'])
