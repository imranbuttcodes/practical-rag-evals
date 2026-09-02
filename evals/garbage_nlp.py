# import time

# start = time.perf_counter()

# # Code whose execution time we want to measure
# total = 0
# for i in range(10000000):
#     total += i

# end = time.perf_counter()

# elapsed = end - start

# print("Total:", total)
# print("Execution time:", elapsed, "seconds")

from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model='openai/gpt-oss-120b',
    api_key=os.getenv('GROQ_API_KEY')
)

print(llm.invoke('what is AI in one word?'))