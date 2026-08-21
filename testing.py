from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
)




while True:
    user_input = input("Enter your question (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    for chunk in llm.stream(user_input):
        print(chunk.content, end='', flush=True)

    print()
    